#!/usr/bin/env python3
"""
make_dataset_v2.py — HumanEvalComm runner with bias-aware logging.

What it does
============
• Iterates a dataset of items and, for each requested category (e.g., 1a, 2ac),
  generates a model response using a prompt template (paper-style by default).
• Extracts both code and questions (from NON-code regions), even for mixed outputs.
• Records rich metadata to enable later judge/eval (committee) analysis.
• Writes a JSONL of item-level rows and a summary.json with run stats.

Assumptions about dataset
=========================
The loader searches for a "modified description" string per category in this order:
  1) item[f"prompt{cat}"]
  2) item[cat]
  3) item["prompts"][cat]            (if a nested "prompts" dict exists)
  4) item["variants"][cat]           (if a nested "variants" dict exists)

Additionally, it looks for:
  • "task_id" (string) — optional; else a sequential id is used
  • "entry_point" (string) — optional

Usage example
=============
python make_dataset_v2.py \
  --dataset ./humanevalcomm.jsonl \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --categories 1a 1c 1p 2ac 2ap 2cp 3acp \
  --max-new-tokens 256 --temperature 1 --top-p 0.95 \
  --outdir ./runs/deepseek_coder_allcats --seed 123

Outputs (in --outdir)
=====================
  items.jsonl          # one JSON record per (item, category)
  summary.json         # aggregate stats and model/tokenizer metadata

"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# ----------------------------
# Prompt template (paper-like)
# ----------------------------
DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert software developer who writes high quality code. "
    "With below information, please either generate Python3 code (Respond directly with code only with markdown), "
    "or ask clarifying questions:\n\n{problem}"
)

# ----------------------------
# Simple dataset loader
# ----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def resolve_modified_description(item: Dict[str, Any], cat: str) -> Optional[str]:
    # Try common keys: prompt{cat}, {cat}, prompts[cat], variants[cat]
    keys = [f"prompt{cat}", cat]
    for k in keys:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]

    for parent in ("prompts", "variants"):
        if parent in item and isinstance(item[parent], dict):
            v = item[parent].get(cat)
            if isinstance(v, str) and v.strip():
                return v
    return None

# ----------------------------
# Regex / heuristics
# ----------------------------
CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
DEF_RE = re.compile(r"^\s*(def|class)\s+\w+\s*\(", re.MULTILINE)

INTERROGATIVE_START_RE = re.compile(
    r"(?i)^\s*(what|which|how|why|when|where|who|whom|whose|should|do|does|did|is|are|was|were|can|could|would|may|must|will)\b"
)

def contains_code_block(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True
    return DEF_RE.search(text) is not None

def extract_code(text: str) -> Optional[str]:
    """
    Returns the FIRST fenced code block if any (including fences).
    """
    lines = text.splitlines(keepends=False)
    out = []
    capturing = False
    for line in lines:
        if line.strip().startswith("```"):
            if not capturing:
                capturing = True
                out.append(line)
            else:
                out.append(line)
                return "\n".join(out)
        elif capturing:
            out.append(line)
    return None

def _strip_fenced_code(text: str) -> str:
    cleaned = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            cleaned.append(line)
    return "\n".join(cleaned)

def extract_questions_sane(text: str) -> List[str]:
    """
    Extract question-like lines from NON-code regions.
    Heuristics:
      - line ends with '?' OR
      - starts with a common interrogative (even without '?')
    Deduplicated (case/space-insensitive).
    """
    non_code = _strip_fenced_code(text)
    qs = []
    for raw in non_code.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith("?") or INTERROGATIVE_START_RE.match(line):
            qs.append(line)
    # deduplicate lightly
    seen = set()
    dedup = []
    for q in qs:
        k = re.sub(r"\s+", " ", q.lower())
        if k not in seen:
            seen.add(k)
            dedup.append(q)
    return dedup

def count_code_fences(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith("```"))

# ----------------------------
# Token counting helpers
# ----------------------------
def token_count(tok, text: str) -> int:
    try:
        return len(tok(text).input_ids)
    except Exception:
        return 0

# ----------------------------
# Seeding helpers
# ----------------------------
def set_all_seeds(seed: int):
    import random, numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)

# ----------------------------
# Data classes
# ----------------------------
@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.0

@dataclass
class ItemOut:
    task_id: str
    category: str                  # e.g., 1a, 2ac, ...
    entry_point: Optional[str]
    prompt_field: str              # e.g., "prompt1a"
    prompt_text: str               # the selected modified description
    prompt_final: str              # resolved prompt after {problem} substitution
    model_name: str
    gen_raw: Dict[str, Any]        # raw pipeline return (slimmed)
    generated_text: str
    contains_code: bool
    extracted_code: Optional[str]
    extracted_questions: List[str]
    asked_question: bool
    has_backticks: bool
    num_code_fences: int
    num_questions: int
    response_mode: str             # code_only | question_only | mixed | other
    num_tokens_prompt: int
    num_tokens_generated: int
    latency_sec: float
    seed: int

# ----------------------------
# HF model prep & generation
# ----------------------------
def prepare_generator(model_name: str, seed: int = 0):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        trust_remote_code=True,
    )

    # Create a torch.Generator for reproducible sampling
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
        g = torch.Generator(device="cuda" if device == 0 else "cpu")
        g.manual_seed(seed)
    except Exception:
        g = None

    return gen_pipe, tok, g, model

def generate_one(gen_pipe, prompt: str, cfg: GenConfig, generator=None) -> Dict[str, Any]:
    out = gen_pipe(
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        repetition_penalty=cfg.repetition_penalty,
        generator=generator,
        return_full_text=True,
    )[0]
    return out

# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL.")
    p.add_argument("--model", type=str, required=True, help="HF model repo id or path.")
    p.add_argument("--categories", nargs="+", required=True,
                   help="List of categories to run, e.g. 1a 1c 1p 2ac 2ap 2cp 3acp")
    p.add_argument("--prompt-template", type=str, default=None,
                   help="Optional path to a prompt template file with '{problem}'.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--do-sample", action="store_true", default=True)
    p.add_argument("--no-sample", dest="do_sample", action="store_false")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, required=True)
    return p.parse_args()

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Prompt template setup
    if args.prompt_template and os.path.exists(args.prompt_template):
        with open(args.prompt_template, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    # Seeding
    set_all_seeds(args.seed)

    # Load model
    gen_pipe, tok, torch_generator, model = prepare_generator(args.model, seed=args.seed)

    # Load data
    data = load_jsonl(args.dataset)

    # Gen config
    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
    )

    rows: List[Dict[str, Any]] = []

    # In case dataset lacks explicit IDs
    def _item_id(i, item):
        return str(item.get("task_id") or item.get("id") or f"idx_{i:05d}")

    t_all0 = time.time()
    for i, item in enumerate(data):
        for cat in args.categories:
            problem = resolve_modified_description(item, cat)
            if problem is None:
                # skip gracefully if category not present in this item
                continue

            # Compose final prompt
            try:
                final_prompt = prompt_template.format(problem=problem)
            except KeyError as e:
                print(f"[WARN] Missing key in prompt template: {e}. Falling back to default.", file=sys.stderr)
                final_prompt = DEFAULT_PROMPT_TEMPLATE.format(problem=problem)

            # Generate
            t0 = time.time()
            gen = generate_one(gen_pipe, final_prompt, cfg, generator=torch_generator)
            dt = time.time() - t0

            # Normalize output dict a bit
            text = gen.get("generated_text", "")
            gen_slim = {
                k: gen.get(k)
                for k in ("generated_text", "details", "index")
                if k in gen
            }

            # Analysis
            has_code = contains_code_block(text)
            code = extract_code(text) if has_code else None
            questions = extract_questions_sane(text)
            asked_question = len(questions) > 0
            num_q = len(questions)
            num_fences = count_code_fences(text)
            has_backticks = ("```" in text)

            # response mode
            if has_code and not asked_question:
                response_mode = "code_only"
            elif asked_question and not has_code:
                response_mode = "question_only"
            elif has_code and asked_question:
                response_mode = "mixed"
            else:
                response_mode = "other"

            n_tok_prompt = token_count(tok, final_prompt)
            n_tok_total = token_count(tok, text)
            n_tok_gen = max(0, n_tok_total - n_tok_prompt)

            task_id = _item_id(i, item)
            entry_point = item.get("entry_point")

            rec = ItemOut(
                task_id=task_id,
                category=cat,
                entry_point=entry_point,
                prompt_field=f"prompt{cat}",
                prompt_text=problem,
                prompt_final=final_prompt,
                model_name=args.model,
                gen_raw=gen_slim,
                generated_text=text,
                contains_code=has_code,
                extracted_code=code,
                extracted_questions=questions,
                asked_question=asked_question,
                has_backticks=has_backticks,
                num_code_fences=num_fences,
                num_questions=num_q,
                response_mode=response_mode,
                num_tokens_prompt=n_tok_prompt,
                num_tokens_generated=n_tok_gen,
                latency_sec=dt,
                seed=args.seed,
            )
            rows.append(asdict(rec))

    total_sec = time.time() - t_all0

    # Write items.jsonl
    items_path = os.path.join(args.outdir, "items.jsonl")
    with open(items_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary stats
    asked_any = sum(1 for r in rows if r.get("asked_question"))
    comm_rate_strict = asked_any / max(1, len(rows))

    non_code = sum(1 for r in rows if not r.get("contains_code"))
    mixed_rate = sum(1 for r in rows if r.get("response_mode") == "mixed") / max(1, len(rows))

    # Model meta
    model_meta = {
        "model_name": getattr(model, "name_or_path", args.model),
        "model_type": getattr(getattr(model, "config", None), "__class__", type("x", (), {})).__name__,
        "tokenizer_name": getattr(getattr(gen_pipe, "tokenizer", None), "name_or_path", "unknown"),
        "pad_token_id": getattr(gen_pipe.tokenizer, "pad_token_id", None) if getattr(gen_pipe, "tokenizer", None) else None,
        "eos_token_id": getattr(gen_pipe.tokenizer, "eos_token_id", None) if getattr(gen_pipe, "tokenizer", None) else None,
        "inference_kwargs": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
    }

    summary = {
        "total": len(rows),
        "communication_rate_strict": comm_rate_strict,
        "mixed_output_rate": mixed_rate,
        "non_code_responses": non_code,
        "runtime_sec": total_sec,
        "categories": args.categories,
        "model_meta": model_meta,
    }

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWrote:\n  {items_path}\n  {os.path.join(args.outdir, 'summary.json')}")

if __name__ == "__main__":
    main()
