#!/usr/bin/env python3
"""
Evaluation via a 1–3 LLM "committee" using a unified JSON JUDGE_PROMPT.

This script replaces the two-step prompts with a single structured
evaluator prompt (provided by the user) that returns a strict JSON schema.

It supports 1, 2, or 3 judge models via --judges (space‑separated).

Outputs:
- per‑item committee judgments (committee_judgments.json)
- aggregate summary (committee_summary.json + .csv)

Env vars (set only the ones you use):
  OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY

Examples:
# Single evaluator
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir ./runs/openai-gpt-3.5-turbo \
  --judges meta-llama/Meta-Llama-3-8B-Instruct

# Double evaluators (majority/median aggregation)
python eval_committee_v2.py \
  --results ./runs/x/results.jsonl \
  --outdir ./runs/x \
  --judges openai/gpt-4o-mini gemini/gemini-2.5-flash-lite

# Triple evaluators
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir ./runs/openai-gpt-3.5-turbo \
  --judges openai/gpt-3.5-turbo gemini/gemini-2.5-flash-lite meta-llama/Meta-Llama-3-8B-Instruct \
    -v \
  --checkpoint-every 10 \
  --log-every 5 \
  --max-tokens 256 \
  --temperature 1.0 \
  --resume

python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir ./runs/openai-gpt-3.5-turbo \
  --judges meta-llama/Meta-Llama-3-8B-Instruct \
  -v \
  --checkpoint-every 10 \
  --log-every 5

python eval_committee_v2.py \
  --results ./runs/deepseek_coder_allcats/results.jsonl \
  --outdir  ./runs/deepseek_coder_allcats \
  --no-llm
  
"""
from __future__ import annotations

import logging
import signal
from pathlib import Path

import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------- User-specified unified judge prompt ----------

JUDGE_PROMPT = """You are a careful *evaluator* of clarifying-question quality and recovery.
You are given:
1) The ORIGINAL coding problem description.
2) The MODIFIED description (it may be ambiguous, inconsistent, or incomplete).
3) A MODEL RESPONSE (which may contain questions and/or code).

Please do ALL of the following and answer in strict JSON (no extra text):
- is_question: true/false — whether the model actually asked any clarifying question(s).
- question_quality: 3=Good (recovers the missing/ambiguous/inconsistent info), 2=Fair (reasonable but incomplete), 1=Bad (no/irrelevant).
- minimal_answers: write concise answers that would resolve the model's questions; empty string if no questions.
- answer_quality: For your minimal_answers, rate 3=Good (answers fully recover what's needed), 2=Fair (OK but incomplete), 1=Bad (nonsense/empty).
- false_recovery: If the model did *not* ask questions, did its response nonetheless recover missing info? true/false.
- reasoning: 1-2 sentence justification.

Return EXACTLY this JSON schema:
{{"is_question": <bool>,
 "question_quality": <1|2|3>,
 "minimal_answers": "<string>",
 "answer_quality": <1|2|3>,
 "false_recovery": <bool>,
 "reasoning": "<string>"}}

ORIGINAL:
{original}

MODIFIED:
{modified}

MODEL RESPONSE:
{response}
"""

# ---------- I/O helpers ----------

def setup_logging(verbosity: int) -> None:
    """verbosity: 0=WARNING, 1=INFO, 2+=DEBUG"""
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def safe_write_json(path: str, obj: Any) -> None:
    """Atomic-ish write: write to .tmp then replace."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(p)

def append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------- Utility: robust JSON extraction ----------

def try_parse_json(obj: str) -> Optional[Dict[str, Any]]:
    """Try json.loads; if that fails, attempt to extract first {...} block."""
    try:
        return json.loads(obj)
    except Exception:
        pass
    # greedy find first JSON object
    start = obj.find("{")
    end = obj.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = obj[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


# ---------- Aggregation helpers ----------

def majority_vote_int(values: List[int]) -> int:
    if not values:
        return 1
    try:
        return statistics.mode(values)
    except statistics.StatisticsError:
        return int(round(statistics.median(values)))


def majority_vote_bool(values: List[bool]) -> bool:
    if not values:
        return False
    true_count = sum(1 for v in values if v)
    return true_count >= math.ceil(len(values) / 2)


# ---------- Judge adapters (OpenAI, Gemini, OpenRouter, HF) ----------

class Judge:
    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 256):
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.provider, self.model_name = self._parse_model(model_id)
        self._init_backend()

    @staticmethod
    def _parse_model(model_id: str) -> Tuple[str, str]:
        if model_id.startswith("openai/"):
            return "openai", model_id.split("/", 1)[1]
        if model_id.startswith("gemini/"):
            return "gemini", model_id.split("/", 1)[1]
        if model_id.startswith("openrouter/"):
            return "openrouter", model_id.split("/", 1)[1]
        if model_id.startswith("hf/"):
            return "hf", model_id.split("/", 1)[1]
        if "/" in model_id:
            return "hf", model_id  # plain HF repo id
        return "openai", model_id

    def _init_backend(self) -> None:
        if self.provider == "openai":
            from openai import OpenAI  # type: ignore
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set but an OpenAI judge was requested.")
            self._client = OpenAI()
        elif self.provider == "gemini":
            import google.generativeai as genai  # type: ignore
            if not os.getenv("GEMINI_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY not set but a Gemini judge was requested.")
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self._client = genai.GenerativeModel(self.model_name)
        elif self.provider == "openrouter":
            from openai import OpenAI  # type: ignore
            if not os.getenv("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY not set but an OpenRouter judge was requested.")
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        elif self.provider == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
            tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=True)
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                device_map="auto",
                return_full_text=False,
            )
        else:
            raise ValueError(f"Unknown provider for judge: {self.provider}")

    @staticmethod
    def build_prompt(original: str, modified: str, response: str) -> str:
        return JUDGE_PROMPT.format(original=original, modified=modified, response=response)

    def _chat(self, prompt: str) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9,
            )
            return resp.choices[0].message.content or ""
        if self.provider == "gemini":
            resp = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.9,
                },
            )
            return getattr(resp, "text", "") or ""
        if self.provider == "openrouter":
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9,
                extra_body={"include_reasoning": True},
            )
            return resp.choices[0].message.content or ""
        if self.provider == "hf":
            outs = self._pipe(prompt, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0, top_p=0.9)
            return outs[0]["generated_text"]
        raise RuntimeError("Unsupported provider")

    def judge_once(self, original: str, modified: str, response: str) -> Dict[str, Any]:
        try:
            raw = self._chat(self.build_prompt(original, modified, response))
        except Exception as e:
            logging.warning(f"Judge '{self.model_id}' failed: {e}")
            return {
                "is_question": False,
                "question_quality": 1,
                "minimal_answers": "",
                "answer_quality": 1, 
                "false_recovery": False,
                "reasoning": f"judge error: {e}", 
                "_raw": "", 
                "_error": str(e),
            }
        parsed = try_parse_json(raw) or {}

        raw = self._chat(self.build_prompt(original, modified, response))
        parsed = try_parse_json(raw) or {}
        # Coerce types / ranges robustly
        def as_bool(x, default=False):
            if isinstance(x, bool): return x
            if isinstance(x, str):
                xl = x.lower()
                if xl in ("true", "yes", "y", "1"): return True
                if xl in ("false", "no", "n", "0"): return False
            return default
        def as_int(x, default=1):
            try:
                v = int(x)
                if v in (1,2,3): return v
                return default
            except Exception:
                return default
        out = {
            "is_question": as_bool(parsed.get("is_question"), False),
            "question_quality": as_int(parsed.get("question_quality"), 1),
            "minimal_answers": str(parsed.get("minimal_answers", "")),
            "answer_quality": as_int(parsed.get("answer_quality"), 1),
            "false_recovery": as_bool(parsed.get("false_recovery"), False),
            "reasoning": str(parsed.get("reasoning", "")),
            "_raw": raw,
        }
        return out

def _recompute_counters(per_item: List[PerItemJudgment]) -> Dict[str, int]:
    c = dict(num_items=0, num_with_questions=0, num_good=0, num_acceptable=0, num_nonq=0, num_false_recovery=0)
    for pi in per_item:
        c["num_items"] += 1
        if pi.final_is_question:
            c["num_with_questions"] += 1
            if (pi.final_question_quality or 1) >= 2:
                c["num_acceptable"] += 1
            if (pi.final_question_quality or 1) == 3:
                c["num_good"] += 1
        else:
            c["num_nonq"] += 1
            if bool(pi.final_false_recovery):
                c["num_false_recovery"] += 1
    return c


# ---------- Data structures ----------

@dataclass
class PerItemJudgment:
    record_id: str
    committee_is_question: List[bool]
    committee_question_quality: List[int]
    committee_minimal_answers: List[str]
    committee_answer_quality: List[int]
    committee_false_recovery: List[bool]
    committee_reasoning: List[str]
    final_is_question: bool
    final_question_quality: Optional[int]
    final_answer_quality: Optional[int]
    final_false_recovery: Optional[bool]

def save_partial_outputs(outdir: str,
                         per_item: List[PerItemJudgment],
                         counters: Dict[str, int],
                         judge_ids: List[str],
                         temperature: float,
                         max_tokens: int,
                         partial_reason: Optional[str] = None) -> None:
    """Save whatever we have so far (committee_judgments.json + summary.json)."""
    ensure_outdir(outdir)

    per_item_path = os.path.join(outdir, "committee_judgments.json")
    safe_write_json(per_item_path, [dataclasses.asdict(pi) for pi in per_item])

    num_items = counters.get("num_items", 0)
    num_with_questions = counters.get("num_with_questions", 0)
    num_nonq = counters.get("num_nonq", 0)
    num_good = counters.get("num_good", 0)
    num_acceptable = counters.get("num_acceptable", 0)
    num_false_recovery = counters.get("num_false_recovery", 0)

    summary = {
        "items": num_items,
        "with_questions": num_with_questions,
        "non_questions": num_nonq,
        "communication_rate": (num_with_questions / num_items) if num_items else 0.0,
        "good_question_rate": (num_good / num_items) if num_items else 0.0,
        "acceptable_question_rate": (num_acceptable / num_items) if num_items else 0.0,
        "false_recovery_rate": (num_false_recovery / num_nonq) if num_nonq else 0.0,
        "judges": judge_ids,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": int(time.time()),
        "partial": True if partial_reason else False,
        "partial_reason": partial_reason,
    }
    summary_path = os.path.join(outdir, "committee_summary.json")
    safe_write_json(summary_path, summary)

    logging.info("Partial outputs saved to:")
    logging.info(f"  {per_item_path}")
    logging.info(f"  {summary_path}")

# ---------- Main evaluation ----------

def evaluate(results_path: str, outdir: str, judge_ids: List[str], temperature: float, max_tokens: int, limit: Optional[int],
             checkpoint_every: int = 10, log_every: int = 1, stream_jsonl: bool = True, resume: bool = False) -> None:
    
    existing_per_item: List[PerItemJudgment] = []
    processed_ids = set()
    per_item: List[PerItemJudgment] = []
    counters = dict(num_items=0, num_with_questions=0, num_good=0, num_acceptable=0, num_nonq=0, num_false_recovery=0)

    per_item_path = os.path.join(outdir, "committee_judgments.json")
    jsonl_path = os.path.join(outdir, "committee_judgments.jsonl")

    if resume and os.path.exists(per_item_path):
        try:
            with open(per_item_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # load into PerItemJudgment objects
            for d in data:
                pi = PerItemJudgment(**d)
                existing_per_item.append(pi)
                processed_ids.add(pi.record_id)
            counters = _recompute_counters(existing_per_item)
            per_item.extend(existing_per_item)
            logging.info(f"Resuming with {len(processed_ids)} items already done.")
        except Exception as e:
            logging.warning(f"Resume requested but failed to load existing per-item file: {e}")

    rows = read_jsonl(results_path)
    # Skip anything already processed (record_id heuristic mirrors creation)
    def _rid(r, idx):
        return r.get("record_id", r.get("task_id", f"item{idx}"))

    rows_to_run = []
    for idx, r in enumerate(rows, start=1):
        rid = _rid(r, idx)
        if rid not in processed_ids:
            rows_to_run.append((idx, r))
    total = len(rows_to_run)

    judges = [Judge(mid, temperature=temperature, max_tokens=max_tokens) for mid in judge_ids]

    # jsonl stream: truncate only if not resuming
    if stream_jsonl:
        if resume and os.path.exists(jsonl_path):
            pass  # append
        else:
            open(jsonl_path, "w").close()
    ensure_outdir(outdir)
    rows = read_jsonl(results_path)
    if limit:
        rows = rows[:limit]

    judges = [Judge(mid, temperature=temperature, max_tokens=max_tokens) for mid in judge_ids]

    per_item: List[PerItemJudgment] = []

    # aggregate counters
    counters = dict(num_items=0, num_with_questions=0, num_good=0, num_acceptable=0, num_nonq=0, num_false_recovery=0)

    # streaming file path
    jsonl_path = os.path.join(outdir, "committee_judgments.jsonl")
    if stream_jsonl:
        # fresh file if exists
        open(jsonl_path, "w").close()

    # graceful interrupt to trigger partial save
    def _handle_interrupt(sig, frame):
        logging.warning("Received interrupt, saving partial results...")
        save_partial_outputs(outdir, per_item, counters, judge_ids, temperature, max_tokens, partial_reason="interrupt")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_interrupt)

    try:
        total = len(rows)
        for idx, r in rows_to_run:
            counters["num_items"] += 1

            original = (
                r.get("original_text")
                or r.get("original")
                or r.get("prompt_original")
                or ""
            )
            modified = (
                r.get("prompt_text")
                or r.get("prompt_final")
                or r.get("modified")
                or ""
            )
            response = r.get("generated_text") or r.get("output_text") or r.get("response") or ""

            com_is_q: List[bool] = []
            com_qqual: List[int] = []
            com_mins: List[str] = []
            com_aqual: List[int] = []
            com_frec: List[bool] = []
            com_reas: List[str] = []

            # judge calls wrapped individually (judge_once already logs on failure)
            for j in judges:
                out = j.judge_once(original, modified, response)
                com_is_q.append(bool(out["is_question"]))
                com_qqual.append(int(out["question_quality"]))
                com_mins.append(str(out["minimal_answers"]))
                com_aqual.append(int(out["answer_quality"]))
                com_frec.append(bool(out["false_recovery"]))
                com_reas.append(str(out["reasoning"]))

            # Aggregate (unchanged)
            final_is_q = majority_vote_bool(com_is_q)
            final_qqual = majority_vote_int(com_qqual) if final_is_q else None
            final_aqual = majority_vote_int(com_aqual) if final_is_q else None
            final_frec = (majority_vote_bool(com_frec) if not final_is_q else None)

            if final_is_q:
                counters["num_with_questions"] += 1
                if final_qqual == 3:
                    counters["num_good"] += 1
                    counters["num_acceptable"] += 1
                elif final_qqual == 2:
                    counters["num_acceptable"] += 1
            else:
                counters["num_nonq"] += 1
                if final_frec:
                    counters["num_false_recovery"] += 1

            pi = PerItemJudgment(
                record_id=r.get("record_id", r.get("task_id", f"item{idx}")),
                committee_is_question=com_is_q,
                committee_question_quality=com_qqual,
                committee_minimal_answers=com_mins,
                committee_answer_quality=com_aqual,
                committee_false_recovery=com_frec,
                committee_reasoning=com_reas,
                final_is_question=final_is_q,
                final_question_quality=final_qqual,
                final_answer_quality=final_aqual,
                final_false_recovery=final_frec,
            )
            per_item.append(pi)

            # Stream to JSONL and periodic checkpoints
            if stream_jsonl:
                append_jsonl(jsonl_path, dataclasses.asdict(pi))

            if checkpoint_every and (idx % checkpoint_every == 0):
                logging.info(f"Checkpoint at {idx}/{total} items...")
                safe_write_json(os.path.join(outdir, "committee_judgments.json"),
                                [dataclasses.asdict(x) for x in per_item])

            # progress log
            if log_every and (idx % log_every == 0):
                if final_is_q:
                    logging.info(f"[{idx}/{total}] {pi.record_id}: is_q=1, qqual={final_qqual}, aqual={final_aqual}")
                else:
                    logging.info(f"[{idx}/{total}] {pi.record_id}: is_q=0, false_recovery={bool(final_frec)}")

        # ---------- compute + save final outputs (your original block, slightly refactored) ----------
        communication_rate = (counters["num_with_questions"] / counters["num_items"]) if counters["num_items"] else 0.0
        good_question_rate = (counters["num_good"] / counters["num_items"]) if counters["num_items"] else 0.0
        acceptable_question_rate = (counters["num_acceptable"] / counters["num_items"]) if counters["num_items"] else 0.0
        false_recovery_rate = (counters["num_false_recovery"] / counters["num_nonq"]) if counters["num_nonq"] else 0.0

        summary = {
            "items": counters["num_items"],
            "with_questions": counters["num_with_questions"],
            "non_questions": counters["num_nonq"],
            "communication_rate": communication_rate,
            "good_question_rate": good_question_rate,
            "acceptable_question_rate": acceptable_question_rate,
            "false_recovery_rate": false_recovery_rate,
            "judges": judge_ids,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": int(time.time()),
        }

        per_item_path = os.path.join(outdir, "committee_judgments.json")
        safe_write_json(per_item_path, [dataclasses.asdict(pi) for pi in per_item])

        summary_path = os.path.join(outdir, "committee_summary.json")
        safe_write_json(summary_path, summary)

        csv_path = os.path.join(outdir, "committee_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["items", "with_questions", "non_questions", "communication_rate", "good_question_rate", "acceptable_question_rate", "false_recovery_rate", "judges", "temperature", "max_tokens"])
            w.writerow([
                summary["items"], summary["with_questions"], summary["non_questions"],
                f"{summary['communication_rate']:.6f}", f"{summary['good_question_rate']:.6f}", f"{summary['acceptable_question_rate']:.6f}", f"{summary['false_recovery_rate']:.6f}",
                " ".join(judge_ids), temperature, max_tokens,
            ])

        logging.info("Saved:")
        logging.info(f"  {per_item_path}")
        logging.info(f"  {summary_path}")
        logging.info(f"  {csv_path}")
        if stream_jsonl:
            logging.info(f"  {jsonl_path}")

    except Exception as e:
        logging.exception("Fatal error during evaluation.")
        save_partial_outputs(outdir, per_item, counters, judge_ids, temperature, max_tokens, partial_reason=str(e))
        # Re-raise so call sites/CI still see a failure code if desired.
        raise


# ---------- CLI ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to results.jsonl")
    p.add_argument("--outdir", required=True, help="Directory to save committee outputs")
    p.add_argument("--judges", nargs="+", default=[], help="1–3 judge model ids (e.g., gemini/gemini-2.5-flash-lite openai/gpt-3.5-turbo meta-llama/Llama-3.1-8B)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help="Debug: skip judge calls; compute only communication rate from saved flags (is_question/extracted_questions)")
    p.add_argument("--verbose", "-v", action="count", default=0, help="-v for INFO, -vv for DEBUG")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Write a JSON checkpoint every N items")
    p.add_argument("--log-every", type=int, default=1, help="Log progress every N items")
    p.add_argument("--no-stream-jsonl", action="store_true", help="Disable streaming committee_judgments.jsonl")
    p.add_argument("--resume", action="store_true", help="Resume from existing per-item outputs in outdir")
    args = p.parse_args(argv)
    if len(args.judges) == 0 and not args.no_llm:
        p.error("--judges required unless --no-llm is set")
    if len(args.judges) > 3:
        p.error("Provide at most 3 judges")
    return args


def main():
    args = parse_args()
    setup_logging(args.verbose)
    if args.no_llm:
        rows = read_jsonl(args.results)
        if args.limit:
            rows = rows[: args.limit]
        items = len(rows)
        with_q = 0
        for r in rows:
            is_q = bool(r.get("is_question")) or (isinstance(r.get("extracted_questions"), list) and len(r.get("extracted_questions")) > 0)
            if is_q: with_q += 1
        nonq = items - with_q
        ensure_outdir(args.outdir)
        summary = {
            "items": items,
            "with_questions": with_q,
            "non_questions": nonq,
            "communication_rate": (with_q / items) if items else 0.0,
            "good_question_rate": None,
            "acceptable_question_rate": None,
            "false_recovery_rate": None,
            "judges": [],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timestamp": int(time.time()),
        }
        with open(os.path.join(args.outdir, "committee_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, indent=2))
        return

    evaluate(
        results_path=args.results,
        outdir=args.outdir,
        judge_ids=args.judges,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        stream_jsonl=not args.no_stream_jsonl,
    )


if __name__ == "__main__":
    main()
