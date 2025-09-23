#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------- I/O helpers ----------

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
                # best‑effort tolerance; you can also raise
                continue
    return rows


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
        # default: treat as HF repo id if there's a slash, otherwise try OpenAI
        if "/" in model_id:
            return "hf", model_id
        return "openai", model_id

    def _init_backend(self) -> None:
        if self.provider == "openai":
            from openai import OpenAI  # type: ignore
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set but an OpenAI judge was requested.")
            self._client = OpenAI()
        elif self.provider == "gemini":
            import google.generativeai as genai  # type: ignore
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set but a Gemini judge was requested.")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model_name)
        elif self.provider == "openrouter":
            from openai import OpenAI  # type: ignore
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set but an OpenRouter judge was requested.")
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
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

    # ---- prompts ----
    @staticmethod
    def _prompt_for_questions(modified_problem: str, questions: List[str]) -> str:
        # Aligned to the 1..3 scale used in the paper; keep the response machine‑parsable.
        qlist = "\n".join(f"- {q}" for q in questions) if questions else "(none)"
        return (
            "You are grading the QUALITY of clarifying questions for a coding task.\n"
            "Scale: 3=Good (insightful and directly recover missing/ambiguous requirements);\n"
            "2=Fair (partially helpful but incomplete); 1=Bad/None (no questions or irrelevant).\n\n"
            f"Modified problem description (possibly ambiguous/incomplete/inconsistent):\n{modified_problem}\n\n"
            f"Clarifying questions to grade:\n{qlist}\n\n"
            "Reply ONLY in this strict JSON: {\"quality\": <1|2|3>, \"rationale\": \"short reason\"}"
        )

    @staticmethod
    def _prompt_for_recovery(modified_problem: str, answer_text: str) -> str:
        return (
            "Judge whether the following NON‑question answer text nonetheless *recovers the missing or ambiguous "
            "requirements* implied by the modified problem description.\n"
            "Respond in strict JSON with {\"recovered\": true|false, \"rationale\": \"short\"}.\n\n"
            f"Modified problem description:\n{modified_problem}\n\n"
            f"Model answer (no questions asked):\n{answer_text}"
        )

    # ---- inference ----
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
            # For HF we just do single‑shot generation; many HF models are not chat‑tuned.
            outs = self._pipe(prompt, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0, top_p=0.9)
            return outs[0]["generated_text"]
        raise RuntimeError("Unsupported provider")

    def grade_questions(self, modified_problem: str, questions: List[str]) -> Tuple[int, str]:
        raw = self._chat(self._prompt_for_questions(modified_problem, questions))
        # Extract a single integer 1..3 from JSON-ish output robustly
        try:
            j = json.loads(raw)
            q = int(j.get("quality", 0))
            if q not in (1, 2, 3):
                raise ValueError
            rationale = str(j.get("rationale", ""))
            return q, rationale
        except Exception:
            # Fallback: find first digit 1..3 in the text
            for ch in raw:
                if ch in "123":
                    return int(ch), raw
            return 1, raw  # conservative

    def grade_recovery(self, modified_problem: str, answer_text: str) -> Tuple[bool, str]:
        raw = self._chat(self._prompt_for_recovery(modified_problem, answer_text))
        try:
            j = json.loads(raw)
            rec = bool(j.get("recovered", False))
            rationale = str(j.get("rationale", ""))
            return rec, rationale
        except Exception:
            text = raw.lower()
            if "true" in text and "false" not in text:
                return True, raw
            if "false" in text and "true" not in text:
                return False, raw
            return False, raw


# ---------- Aggregation ----------

def majority_vote_int(values: List[int]) -> int:
    if not values:
        return 1
    try:
        return statistics.mode(values)
    except statistics.StatisticsError:
        # tie -> median (rounded)
        return int(round(statistics.median(values)))


def majority_vote_bool(values: List[bool]) -> bool:
    if not values:
        return False
    true_count = sum(1 for v in values if v)
    return true_count >= math.ceil(len(values) / 2)


# ---------- Main evaluation ----------

@dataclass
class PerItemJudgment:
    record_id: str
    is_question: bool
    question_count: int
    committee_quality_labels: List[int]
    final_quality_label: Optional[int]  # only for question items
    committee_recovered: List[bool]
    final_recovered: Optional[bool]     # only for non‑question items


def evaluate(results_path: str, outdir: str, judge_ids: List[str], temperature: float, max_tokens: int, limit: Optional[int]) -> None:
    ensure_outdir(outdir)
    rows = read_jsonl(results_path)
    if limit:
        rows = rows[:limit]

    judges = [Judge(mid, temperature=temperature, max_tokens=max_tokens) for mid in judge_ids]

    per_item: List[PerItemJudgment] = []

    num_with_questions = 0
    num_items = 0

    # for rates
    num_good = 0  # quality==3 among *question items*
    num_acceptable = 0  # quality in {2,3} among *question items*

    # For non‑questions
    num_nonq = 0
    num_false_recovery = 0

    for r in rows:
        num_items += 1
        is_q = bool(r.get("is_question")) or (isinstance(r.get("extracted_questions"), list) and len(r.get("extracted_questions")) > 0)
        q_list = list(r.get("extracted_questions", []) or [])
        modified_problem = r.get("prompt_text") or r.get("prompt_final") or ""
        answer_text = r.get("generated_text", "")

        if is_q:
            num_with_questions += 1
            labels = []
            for j in judges:
                q, _why = j.grade_questions(modified_problem, q_list)
                labels.append(int(q))
            final_q = majority_vote_int(labels)
            if final_q == 3:
                num_good += 1
                num_acceptable += 1
            elif final_q == 2:
                num_acceptable += 1
            per_item.append(PerItemJudgment(
                record_id=r.get("record_id", r.get("task_id", f"item{num_items}")),
                is_question=True,
                question_count=len(q_list),
                committee_quality_labels=labels,
                final_quality_label=final_q,
                committee_recovered=[],
                final_recovered=None,
            ))
        else:
            num_nonq += 1
            votes = []
            for j in judges:
                rec, _why = j.grade_recovery(modified_problem, answer_text)
                votes.append(bool(rec))
            final_rec = majority_vote_bool(votes)
            if final_rec:
                num_false_recovery += 1
            per_item.append(PerItemJudgment(
                record_id=r.get("record_id", r.get("task_id", f"item{num_items}")),
                is_question=False,
                question_count=0,
                committee_quality_labels=[],
                final_quality_label=None,
                committee_recovered=votes,
                final_recovered=final_rec,
            ))

    # --- compute aggregate metrics ---
    communication_rate = (num_with_questions / num_items) if num_items else 0.0
    good_question_rate = (num_good / num_items) if num_items else 0.0
    acceptable_question_rate = (num_acceptable / num_items) if num_items else 0.0
    false_recovery_rate = (num_false_recovery / num_nonq) if num_nonq else 0.0

    summary = {
        "items": num_items,
        "with_questions": num_with_questions,
        "non_questions": num_nonq,
        "communication_rate": communication_rate,
        "good_question_rate": good_question_rate,
        "acceptable_question_rate": acceptable_question_rate,
        "false_recovery_rate": false_recovery_rate,
        "judges": judge_ids,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": int(time.time()),
    }

    # Save outputs
    # 1) Per‑item JSON
    per_item_path = os.path.join(outdir, "committee_judgments.json")
    with open(per_item_path, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(pi) for pi in per_item], f, ensure_ascii=False, indent=2)

    # 2) Summary JSON
    summary_path = os.path.join(outdir, "committee_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 3) Summary CSV (handy for spreadsheets)
    csv_path = os.path.join(outdir, "committee_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["items", "with_questions", "non_questions", "communication_rate", "good_question_rate", "acceptable_question_rate", "false_recovery_rate", "judges", "temperature", "max_tokens"])
        w.writerow([
            summary["items"], summary["with_questions"], summary["non_questions"],
            f"{summary['communication_rate']:.6f}", f"{summary['good_question_rate']:.6f}", f"{summary['acceptable_question_rate']:.6f}", f"{summary['false_recovery_rate']:.6f}",
            " ".join(judge_ids), temperature, max_tokens,
        ])

    print("\nSaved:")
    print(" ", per_item_path)
    print(" ", summary_path)
    print(" ", csv_path)


# ---------- CLI ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to results.jsonl produced by make_dataset_v2.py")
    p.add_argument("--outdir", required=True, help="Directory to save committee outputs")
    p.add_argument("--judges", nargs="+", default=[], help="1–3 judge model ids (e.g., gemini/gemini-2.5-flash-lite openai/gpt-3.5-turbo meta-llama/Llama-3.1-8B)")
    p.add_argument("--split", default="train")  # reserved; not used but kept for parity
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help="Debug: don’t call judges; compute only communication rate from saved flags")
    args = p.parse_args(argv)
    if len(args.judges) == 0 and not args.no_llm:
        p.error("--judges required unless --no-llm is set")
    if len(args.judges) > 3:
        p.error("Provide at most 3 judges")
    return args


def main():
    args = parse_args()

    if args.no_llm:
        # Fast path: compute simple rates only, no committee calls
        rows = read_jsonl(args.results)
        if args.limit:
            rows = rows[: args.limit]
        items = len(rows)
        with_q = sum(1 for r in rows if bool(r.get("is_question")) or (isinstance(r.get("extracted_questions"), list) and len(r.get("extracted_questions")) > 0))
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
    )


if __name__ == "__main__":
    main()
