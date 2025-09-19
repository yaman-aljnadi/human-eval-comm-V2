#!/usr/bin/env python3
"""
eval_committee.py — 3‑LLM committee evaluator for HumanEvalComm runs.

Reads a results.jsonl (or .jsonl.gz) produced by make_dataset_v2.py, asks a
committee of 3 LLM judges to rate each model response, and computes the paper’s
metrics:

- Communication Rate (from logged is_question)
- Good Question Rate (committee label == 3)
- Good Answer Rate (questions)      -> from committee's "answer_quality" == 3
- Acceptable Answer Rate (questions) -> from committee's "answer_quality" in {2,3}
- False Recovery Rate (non-questions)-> among non-question responses, committee says
                                        the model's response recovered missing info

The script also writes back a per-item augmentation JSONL with the committee’s
votes and a summary JSON.

Backends supported for judges:
  * OpenAI (env: OPENAI_API_KEY)             e.g. --judges openai/gpt-4o-mini openai/gpt-4o-mini openai/gpt-4o-mini
  * OpenRouter (env: OPENROUTER_API_KEY)     e.g. --judges openrouter/deepseek/deepseek-r1:free openrouter/google/gemini-2.0-flash-thinking-exp:free openrouter/openai/gpt-4o-mini
  * Gemini via google-generativeai (env: GEMINI_API_KEY) e.g. --judges gemini/gemini-2.0-flash-lite gemini/gemini-2.0-flash-lite gemini/gemini-2.0-flash-lite

Usage:
  python eval_committee.py \
    --results ./runs/deepseek_coder_allcats/results.jsonl \
    --outdir  ./runs/deepseek_coder_allcats \
    --judges openai/gpt-4o-mini openrouter/deepseek/deepseek-r1:free gemini/gemini-2.0-flash-lite \
    --max-tokens 512 --temperature 0.2

Notes:
- We fetch the ORIGINAL & MODIFIED problem text from the public dataset so judges
  can check “recovery” precisely.
- If you already captured evaluator outputs, this script can be run in "dry"
  mode (--no-llm) to only recompute aggregate metrics from stored labels.

Output files:
  outdir/
    committee.jsonl         # per-item augmentation (record_id keyed)
    committee_summary.json  # corpus-level metrics + per-category breakdown
"""

from __future__ import annotations
import argparse
import gzip
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------- Small utils ----------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            for ln in f:
                if not ln.strip():
                    continue
                out.append(json.loads(ln.decode("utf-8")))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                out.append(json.loads(ln))
    return out

def atomic_write(path: str, data: str, binary: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    mode = "wb" if binary else "w"
    with open(tmp, mode) as f:
        if binary:
            f.write(data)
        else:
            f.write(data)
    os.replace(tmp, path)

def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Dataset helpers ----------

@dataclass
class SourceItem:
    task_id: str
    category: str
    modified_problem: str
    original_problem: str

def load_hec_index(split: str = "train") -> Dict[Tuple[str, str], SourceItem]:
    """
    Build (task_id, category) -> SourceItem from the huggingface dataset.
    Requires: datasets
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("pip install datasets") from e

    ds = load_dataset("jie-jw-wu/HumanEvalComm", split=split)
    index: Dict[Tuple[str, str], SourceItem] = {}
    for rec in ds:
        task_id = rec.get("task_id")
        cat = rec.get("variant") or rec.get("category") or rec.get("clarification_category")
        modified = rec.get("problem_modified") or rec.get("problem") or ""
        original = rec.get("problem_original") or rec.get("original_problem") or rec.get("missing_information") or ""
        if not task_id or not cat:
            # Be permissive; skip if missing keys.
            continue
        index[(str(task_id), str(cat))] = SourceItem(
            task_id=str(task_id),
            category=str(cat),
            modified_problem=str(modified),
            original_problem=str(original),
        )
    return index

# ---------- Judge backends ----------

class JudgeClient:
    def __init__(self, model_spec: str, temperature: float = 0.2, max_tokens: int = 512):
        """
        model_spec formats:
          - "openai/<model>"
          - "openrouter/<provider>/<model>" OR "openrouter/<model>"
          - "gemini/<model>"
        """
        self.spec = model_spec.strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        if self.spec.startswith("openai/"):
            from openai import OpenAI
            self.kind = "openai"
            self.model = self.spec.split("/", 1)[1]
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.spec.startswith("openrouter/"):
            from openai import OpenAI
            self.kind = "openrouter"
            self.model = self.spec.split("/", 1)[1]  # allow provider/model or just model
            self.client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        elif self.spec.startswith("gemini/"):
            import google.generativeai as genai
            self.kind = "gemini"
            self.model = self.spec.split("/", 1)[1]
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            genai.configure(api_key=gemini_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unrecognized judge model spec: {self.spec}")

    def chat(self, prompt: str) -> str:
        if self.kind in ("openai", "openrouter"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        elif self.kind == "gemini":
            resp = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            try:
                return (resp.text or "").strip()
            except Exception:
                return ""
        else:
            raise RuntimeError("Unknown backend")

# ---------- Prompts ----------

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
{"is_question": <bool>,
 "question_quality": <1|2|3>,
 "minimal_answers": "<string>",
 "answer_quality": <1|2|3>,
 "false_recovery": <bool>,
 "reasoning": "<string>"}

ORIGINAL:
{original}

MODIFIED:
{modified}

MODEL RESPONSE:
{response}
"""

def parse_safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # Try to extract the last JSON object
        import re
        m = re.search(r"\{.*\}\s*$", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # Fallback default
    return {
        "is_question": False,
        "question_quality": 1,
        "minimal_answers": "",
        "answer_quality": 1,
        "false_recovery": False,
        "reasoning": "parse_error",
    }

# ---------- Voting ----------

def majority_vote(ints: List[int], default: int = 1) -> int:
    from collections import Counter
    cnt = Counter(ints)
    value, _ = cnt.most_common(1)[0]
    return value if value in (1,2,3) else default

def bool_vote(bools: List[bool]) -> bool:
    return sum(1 for b in bools if b) >= 2

# ---------- Main pipeline ----------

def run_committee(
    results_path: str,
    outdir: str,
    judges: List[str],
    split: str = "train",
    temperature: float = 0.2,
    max_tokens: int = 512,
    dry: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:

    rows = load_jsonl(results_path)
    if limit is not None:
        rows = rows[:limit]

    # Build index to ORIGINAL/MODIFIED text
    hec = load_hec_index(split=split)

    # Init judges
    judge_clients = []
    if not dry:
        for spec in judges:
            judge_clients.append(JudgeClient(spec, temperature=temperature, max_tokens=max_tokens))

    # Outputs
    aug_rows: List[Dict[str, Any]] = []

    # Running tallies
    total = 0
    non_code = 0  # equals "is_question" logged by generator
    gq_good = 0
    ans_good = 0
    ans_acc = 0
    nq_total = 0
    false_recovery_count = 0

    # Category breakdowns
    by_cat = {}

    for r in rows:
        total += 1
        task_id = str(r.get("task_id") or r.get("problem_id") or "")
        category = str(r.get("category") or "")
        key = (task_id, category)
        src = hec.get(key)

        original = src.original_problem if src else ""
        modified = src.modified_problem if src else (r.get("prompt_text") or "")
        response = r.get("generated_text") or ""

        was_question = bool(r.get("is_question", False))
        if was_question:
            non_code += 1

        # If dry, reuse logged signals only (no LLM calls)
        judge_votes: List[Dict[str, Any]] = []
        if dry:
            vote = {
                "is_question": was_question,
                "question_quality": 3 if was_question else 1,
                "minimal_answers": "" if not was_question else "N/A (dry)",
                "answer_quality": 2 if was_question else 1,
                "false_recovery": False,
                "reasoning": "dry mode",
            }
            judge_votes = [vote, vote, vote]
        else:
            # Query each judge
            for jc in judge_clients:
                prompt = JUDGE_PROMPT.format(original=original, modified=modified, response=response)
                text = jc.chat(prompt)
                judge_votes.append(parse_safe_json(text))

        # Aggregate
        q_quals = [int(v.get("question_quality", 1)) for v in judge_votes]
        ans_quals = [int(v.get("answer_quality", 1)) for v in judge_votes]
        false_recs = [bool(v.get("false_recovery", False)) for v in judge_votes]
        is_q_flags = [bool(v.get("is_question", False)) for v in judge_votes]

        committee_is_question = bool_vote(is_q_flags)
        committee_q_quality = majority_vote(q_quals, default=1)
        committee_ans_quality = majority_vote(ans_quals, default=1)
        committee_false_recovery = bool_vote(false_recs)

        # Metrics accumulation
        if committee_q_quality == 3:
            gq_good += 1
        if committee_is_question:
            if committee_ans_quality == 3:
                ans_good += 1
            if committee_ans_quality in (2,3):
                ans_acc += 1
        else:
            nq_total += 1
            if committee_false_recovery:
                false_recovery_count += 1

        # Per-category
        if category not in by_cat:
            by_cat[category] = {"n": 0, "gq_good": 0, "nq_total": 0, "false_rec": 0, "ans_good": 0, "ans_acc": 0, "comm": 0}
        by_cat[category]["n"] += 1
        by_cat[category]["comm"] += 1 if committee_is_question else 0
        by_cat[category]["gq_good"] += 1 if committee_q_quality == 3 else 0
        if committee_is_question:
            by_cat[category]["ans_good"] += 1 if committee_ans_quality == 3 else 0
            by_cat[category]["ans_acc"] += 1 if committee_ans_quality in (2,3) else 0
        else:
            by_cat[category]["nq_total"] += 1
            by_cat[category]["false_rec"] += 1 if committee_false_recovery else 0

        aug = {
            "record_id": r.get("record_id"),
            "task_id": task_id,
            "category": category,
            "model_name": r.get("model_name"),
            "seed": r.get("seed"),
            "is_question_logged": was_question,
            "committee_is_question": committee_is_question,
            "committee_question_quality": committee_q_quality,
            "committee_answer_quality": committee_ans_quality,
            "committee_false_recovery": committee_false_recovery,
            "judges": judge_votes,
        }
        aug_rows.append(aug)

    # Corpus-level metrics (paper-aligned)
    comm_rate = (non_code / total) if total else 0.0  # based on logged is_question (paper’s def)
    good_q_rate = (gq_good / total) if total else 0.0
    good_ans_rate_questions = (ans_good / non_code) if non_code else 0.0
    acceptable_ans_rate_questions = (ans_acc / non_code) if non_code else 0.0
    false_recovery_rate = (false_recovery_count / nq_total) if nq_total else 0.0

    # Build per-category view
    cat_view = {}
    for cat, d in by_cat.items():
        n = d["n"]
        comm = (d["comm"] / n) if n else 0.0
        gq = (d["gq_good"] / n) if n else 0.0
        ga = (d["ans_good"] / d["comm"]) if d["comm"] else 0.0
        aa = (d["ans_acc"] / d["comm"]) if d["comm"] else 0.0
        fr = (d["false_rec"] / d["nq_total"]) if d["nq_total"] else 0.0
        cat_view[cat] = {
            "n": n, "communication_rate": comm, "good_question_rate": gq,
            "good_answer_rate_questions": ga, "acceptable_answer_rate_questions": aa,
            "false_recovery_rate_non_questions": fr,
        }

    summary = {
        "total": total,
        "communication_rate": comm_rate,
        "good_question_rate": good_q_rate,
        "good_answer_rate_questions": good_ans_rate_questions,
        "acceptable_answer_rate_questions": acceptable_ans_rate_questions,
        "false_recovery_rate_non_questions": false_recovery_rate,
        "by_category": cat_view,
        "judges": judges,
        "results_path": results_path,
        "created_utc": int(time.time()),
    }

    # Write outputs
    append_jsonl(os.path.join(outdir, "committee.jsonl"), aug_rows)
    atomic_write(os.path.join(outdir, "committee_summary.json"), json.dumps(summary, indent=2))

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.jsonl or .jsonl.gz from make_dataset_v2.py")
    ap.add_argument("--outdir", required=True, help="Where to write committee.jsonl and committee_summary.json")
    ap.add_argument("--judges", nargs=3, metavar=("J1","J2","J3"),
                    help="Three judge model specs, e.g. openai/gpt-4o-mini openrouter/deepseek/deepseek-r1:free gemini/gemini-2.0-flash-lite")
    ap.add_argument("--split", default="train")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Only score first N rows (debug)")
    ap.add_argument("--no-llm", action="store_true", help="Dry mode: no LLM calls; derive metrics from logs only")
    args = ap.parse_args()

    if not args.no_llm and (not args.judges or len(args.judges) != 3):
        ap.error("Please supply exactly three --judges (or use --no-llm).")

    summary = run_committee(
        results_path=args.results,
        outdir=args.outdir,
        judges=args.judges or [],
        split=args.split,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        dry=bool(args.no_llm),
        limit=args.limit,
    )

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
