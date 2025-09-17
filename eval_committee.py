#!/usr/bin/env python3
"""
eval_committee.py — Separate evaluation with a 3-judge committee.

What it does now:
  - Loads one or more results.jsonl files (from make_dataset_v2.py or your proto)
  - For each (task, category, generated_text) runs 3 judges:
      * HeuristicJudge (built-in, no API)
      * (optional) OpenAIJudge (stub)
      * (optional) HFJudge (stub)
  - Aggregates per-item scores by *average* (or majority if you prefer labels)
  - Reports Communication Rate and a proxy 'Good Question Rate' under committee voting

You can later:
  - Swap judges to real LLMs (ChatGPT, Claude, etc.)
  - Change aggregation from average to majority vote, median, weighted, etc.

Usage:
  python eval_committee.py \
    --inputs ./runs/deepseek_coder_allcats/results.jsonl ./runs/codellama_allcats/results.jsonl \
    --committee heuristic heuristic heuristic \
    --out ./runs/eval_committee_summary.json
"""
import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------- utilities -----------------
CODE_FENCE_RE = re.compile(r"```")
DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
QMARK_RE = re.compile(r"\?")

def is_code_like(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True
    if DEF_RE.search(text):
        return True
    return False

def count_questions(text: str) -> int:
    # crude but serviceable
    return len(QMARK_RE.findall(text))

# ----------------- judge API -----------------
class Judge:
    """
    A judge returns:
      - label: one of {"questions", "code", "mixed/invalid"}
      - score_good_question: float in [0,1] estimating quality of clarifying questions (1=good, 0=bad)
    """
    def judge(self, prompt_text: str, generated_text: str) -> Tuple[str, float]:
        raise NotImplementedError

class HeuristicJudge(Judge):
    """
    Offline, no-API judge.
      - If code detected -> label 'code', score_good_question = 0
      - Else if at least one '?' -> 'questions', score_good_question = min(1, n_q/3)
      - Else -> 'mixed/invalid', 0
    """
    def judge(self, prompt_text: str, generated_text: str) -> Tuple[str, float]:
        if is_code_like(generated_text):
            return "code", 0.0
        n_q = count_questions(generated_text)
        if n_q > 0:
            # crude idea: up to 3 reasonable questions ~ 1.0
            return "questions", min(1.0, n_q / 3.0)
        return "mixed/invalid", 0.0

class OpenAIJudge(Judge):
    """
    Stub: fill in your OpenAI call and map response to (label, score).
    Keep deterministic formatting so you can parse easily.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    def judge(self, prompt_text: str, generated_text: str) -> Tuple[str, float]:
        # TODO: implement OpenAI call
        raise NotImplementedError("Wire your OpenAI judging here")

class HFJudge(Judge):
    """
    Stub: fill in a Hugging Face pipeline/model call and parse output to (label, score).
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        # TODO: lazy-load a text-generation or text-classification pipeline here
    def judge(self, prompt_text: str, generated_text: str) -> Tuple[str, float]:
        # TODO: implement HF call
        raise NotImplementedError("Wire your HF judging here")

# ----------------- aggregation -----------------
def aggregate_committee(votes: List[Tuple[str, float]]) -> Tuple[str, float]:
    """
    votes: list of (label, score)
    Returns:
      label_majority, score_avg
    """
    # majority on labels
    counts: Dict[str, int] = {}
    for lab, _ in votes:
        counts[lab] = counts.get(lab, 0) + 1
    label_majority = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    # average score
    score_avg = sum(s for _, s in votes) / len(votes) if votes else 0.0
    return label_majority, score_avg

# ----------------- IO -----------------
def load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows

# ----------------- main -----------------
def build_committee(names: List[str]) -> List[Judge]:
    out: List[Judge] = []
    for n in names:
        key = n.lower()
        if key == "heuristic":
            out.append(HeuristicJudge())
        elif key.startswith("openai"):
            # e.g., "openai:gpt-4o-mini"
            parts = n.split(":", 1)
            out.append(OpenAIJudge(model=parts[1] if len(parts) > 1 else "gpt-4o-mini"))
        elif key.startswith("hf"):
            # e.g., "hf:meta-llama/Llama-3.1-8B-Instruct"
            parts = n.split(":", 1)
            if len(parts) == 1:
                raise ValueError("HF judge requires model name, e.g., hf:meta-llama/Llama-3.1")
            out.append(HFJudge(model_name=parts[1]))
        else:
            raise ValueError(f"Unknown judge spec: {n}")
    if len(out) != 3:
        raise ValueError("Please specify exactly 3 judges (e.g., heuristic heuristic heuristic)")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more results.jsonl files")
    ap.add_argument("--committee", nargs=3, required=True,
                    help="Exactly 3 judges, e.g., heuristic heuristic heuristic "
                         "or openai:gpt-4o-mini hf:meta-llama/Llama-3.1 heuristic")
    ap.add_argument("--out", required=True, help="Path to write summary JSON")
    args = ap.parse_args()

    rows = load_jsonl(args.inputs)
    judges = build_committee(args.committee)

    n = len(rows)
    if n == 0:
        raise SystemExit("No rows to evaluate")

    comm_labels = 0
    good_q_sum = 0.0

    by_cat: Dict[str, Dict[str, float]] = {}  # category -> {comm_rate, goodq_avg, count}
    cat_counts: Dict[str, int] = {}

    for r in rows:
        prompt_text = r.get("prompt_text") or r.get("prompt_modified") or ""
        gen_text = r.get("generated_text", "")
        category = r.get("category") or r.get("prompt_field") or "NA"

        votes = [j.judge(prompt_text, gen_text) for j in judges]
        maj_label, score_avg = aggregate_committee(votes)

        if maj_label == "questions":
            comm_labels += 1
        good_q_sum += score_avg

        # per-category
        cat_counts[category] = cat_counts.get(category, 0) + 1
        bucket = by_cat.setdefault(category, dict(comm=0.0, goodq=0.0))
        # we'll fill later after totals known

    total = float(n)
    comm_rate = comm_labels / total
    goodq_avg = good_q_sum / total

    # per-category finalize
    # we need pass 2 to compute per-cat sums:
    comm_counts: Dict[str, int] = {k: 0 for k in cat_counts}
    goodq_sums: Dict[str, float] = {k: 0.0 for k in cat_counts}

    for r in rows:
        prompt_text = r.get("prompt_text") or r.get("prompt_modified") or ""
        gen_text = r.get("generated_text", "")
        category = r.get("category") or r.get("prompt_field") or "NA"

        votes = [j.judge(prompt_text, gen_text) for j in judges]
        maj_label, score_avg = aggregate_committee(votes)

        if maj_label == "questions":
            comm_counts[category] += 1
        goodq_sums[category] += score_avg

    for cat, cnt in cat_counts.items():
        by_cat[cat]["comm"] = (comm_counts[cat] / cnt) if cnt else 0.0
        by_cat[cat]["goodq"] = (goodq_sums[cat] / cnt) if cnt else 0.0

    summary = {
        "num_items": n,
        "committee": args.committee,
        "communication_rate_committee": comm_rate,   # % labeled 'questions' by majority
        "good_question_avg_committee": goodq_avg,    # average of numeric judge scores
        "per_category": by_cat,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Committee Summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
