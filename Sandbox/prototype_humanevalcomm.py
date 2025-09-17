#!/usr/bin/env python3
"""
HumanEvalComm prototype runner (Option 1: evaluation reliability)

- Loads jie-jw-wu/HumanEvalComm
- Selects prompt variant fields (prompt1a, prompt1c, prompt1p, 2ac, 2ap, 2cp, 3acp)
- Runs HuggingFace causal LMs to generate an initial response per item
- Computes Communication Rate (non-code responses), as per paper definition
- Saves per-item outputs and quick summary for future judging

USAGE (example):
  python prototype_humanevalcomm.py \
      --model deepseek-ai/deepseek-coder-6.7b-instruct \
      --categories 1a 1c 1p \
      --max-new-tokens 256 \
      --temperature 1.0 \
      --top-p 0.9 \
      --limit 20 \
      --outdir ./runs/deepseek-coder-6.7b_1a1c1p
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Lazy imports so the script prints helpful errors if libs are missing
def _lazy_imports():
    try:
        import torch  # noqa: F401
    except Exception as e:
        print("ERROR: PyTorch is required. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        raise
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa: F401
    except Exception:
        print("ERROR: transformers is required. Install with: pip install transformers accelerate bitsandbytes", file=sys.stderr)
        raise
    try:
        from datasets import load_dataset  # noqa: F401
    except Exception:
        print("ERROR: datasets is required. Install with: pip install datasets", file=sys.stderr)
        raise

@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None  # not all models support `stopping_criteria` in pipeline

@dataclass
class ItemResult:
    task_id: str
    category: Optional[str]          # now set to the prompt key used (e.g., "prompt1a")
    entry_point: Optional[str]
    prompt_modified: str
    prompt_original: Optional[str]
    model_name: str
    seed: int
    gen: Dict[str, Any]              # raw HF output dict (without "generated_text" key)
    generated_text: str              # single source of truth for the text output
    contains_code: bool              # heuristic
    extracted_code: Optional[str]    # parsed from output
    extracted_questions: List[str]   # parsed from output
    latency_sec: float

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
Q_SENT_RE = re.compile(r"[^.?!]*\?")
PROMPT_VARIANTS = ["1a", "1c", "1p", "2ac", "2ap", "2cp", "3acp"]

def contains_code_block(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True
    # If no backticks, try to detect obvious Python function definitions
    return DEF_RE.search(text) is not None

def extract_code(text: str) -> Optional[str]:
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: grab lines around a def
    if DEF_RE.search(text):
        lines = text.splitlines()
        # simple heuristic: return contiguous lines from first 'def' onward
        for i, ln in enumerate(lines):
            if DEF_RE.match(ln):
                return "\n".join(lines[i:]).strip()
    return None

def extract_questions(text: str) -> List[str]:
    # Very light heuristic: split by '?' and re-attach '?'
    chunks = [c.strip() for c in text.split('?') if c.strip()]
    qs = []
    for c in chunks:
        if 'def ' in c or 'return ' in c or '```' in c:
            continue
        qs.append(c + '?')
    # Also find inline question-like sentences
    for m in Q_SENT_RE.finditer(text):
        q = m.group(0).strip()
        if q and q not in qs and '```' not in q:
            qs.append(q)
    # de-duplicate preserving order
    seen = set()
    dedup = []
    for q in qs:
        k = q.strip()
        if k and k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup

def _select_prompt_field(item: Dict[str, Any], requested_categories: Optional[List[str]]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns: (prompt_text, category_label, prompt_original)

    - If requested_categories is given, try keys 'prompt{cat}' in order.
    - Otherwise, prefer 'prompt' if present; else fall back to any existing variant in PROMPT_VARIANTS.
    - category_label is the key name used (e.g., 'prompt1a') so it's never None when a variant is chosen.
    """
    # If the dataset offers an explicit original/description field, keep it if available.
    prompt_original = item.get("original_prompt") or item.get("original_description") or item.get("original") or None

    candidate_keys: List[str] = []
    if requested_categories:
        candidate_keys.extend([f"prompt{c}" for c in requested_categories])
    # General fallback order: 'prompt' then all known variants
    candidate_keys.append("prompt")
    candidate_keys.extend([f"prompt{v}" for v in PROMPT_VARIANTS])

    for k in candidate_keys:
        txt = item.get(k)
        if isinstance(txt, str) and txt.strip():
            return txt, (k if k != "prompt" else "prompt"), prompt_original

    # Final fallback: empty string
    return "", None, prompt_original

def build_initial_prompt(item: Dict[str, Any], requested_categories: Optional[List[str]]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Returns: (final_prompt, category_label, entry_point, prompt_original)
    """
    # Common HumanEval-like keys
    entry_point = item.get("entry_point") or item.get("entrypoint") or None

    # Choose which prompt text to feed the model and set the category label to that field name
    prompt_text, category_label, prompt_original = _select_prompt_field(item, requested_categories)

    system_header = (
        "You are evaluating a possibly inconsistent, ambiguous, or incomplete programming problem.\n"
        "Either:\n"
        "  (A) Ask clarifying questions (ONLY questions, no code),\n"
        "or\n"
        "  (B) Return Python3 code only (in a fenced block), if you are certain the requirements are fully clear.\n"
        "Respond with ONLY one of the two (questions OR code)."
    )
    final_prompt = f"{system_header}\n\n### Problem:\n{prompt_text}"
    return final_prompt, category_label, entry_point, prompt_original

def load_dataset_filtered(categories: Optional[List[str]] = None, split: str = "train"):
    """
    HumanEvalComm doesn't expose a 'category' column; it exposes multiple prompt fields instead
    (e.g., 'prompt1a', 'prompt1c', ...). So we generally don't filter rows here.
    We keep this function in case a future variant adds a category-like column.
    """
    from datasets import load_dataset
    ds = load_dataset("jie-jw-wu/HumanEvalComm", split=split)

    # If the dataset *did* have a category-like column, we could filter here; otherwise we skip filtering.
    if categories:
        # Try a few likely field names for category
        key = None
        for k in ["category", "clarification_type", "type"]:
            if k in ds.features:
                key = k
                break
        if key is not None:
            ds = ds.filter(lambda x: x.get(key) in categories)
        else:
            print("[info] No explicit category field in dataset; will select among prompt variants per item.")
    return ds

def prepare_generator(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Prefer 8-bit if bitsandbytes is available; otherwise fallback to fp16/bf16 auto
    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    try:
        load_kwargs.update(dict(load_in_8bit=True))
    except Exception:
        pass
    try:
        load_kwargs.update(dict(torch_dtype="auto"))
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        return_full_text=True,
    )
    return gen_pipe, tok

def generate_one(gen_pipe, prompt: str, cfg: GenConfig) -> Dict[str, Any]:
    kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        repetition_penalty=cfg.repetition_penalty,
        pad_token_id=gen_pipe.tokenizer.pad_token_id,
        eos_token_id=gen_pipe.tokenizer.eos_token_id,
    )
    return gen_pipe(prompt, **kwargs)[0]

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "by_item"), exist_ok=True)

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    _lazy_imports()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model hub id, e.g., deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--categories", nargs="*", default=None, help="Prompt variants to prefer: 1a 1c 1p 2ac 2cp 2ap 3acp. Default: try 'prompt', then all variants")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="Debug: stop after N items")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()
    ensure_outdir(args.outdir)

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        repetition_penalty=args.repetition_penalty,
    )

    # Save run config
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[load] dataset (split={args.split}) ...")
    ds = load_dataset_filtered(categories=args.categories, split=args.split)
    print(f"[load] {len(ds)} items after filtering")

    print(f"[model] loading {args.model} ...")
    gen_pipe, tok = prepare_generator(args.model)

    rows: List[Dict[str, Any]] = []
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    t0_all = time.time()

    for i in range(n):
        rec = ds[i]
        final_prompt, category_label, entry_point, prompt_orig = build_initial_prompt(rec, args.categories)

        t0 = time.time()
        gen = generate_one(gen_pipe, final_prompt, cfg)
        dt = time.time() - t0

        text = gen.get("generated_text", "") or str(gen)

        # Keep only ONE copy of generated text: top-level only
        gen_slim = dict(gen)
        if "generated_text" in gen_slim:
            try:
                del gen_slim["generated_text"]
            except Exception:
                pass

        has_code = contains_code_block(text)
        code = extract_code(text) if has_code else None
        questions = [] if has_code else extract_questions(text)

        ir = ItemResult(
            task_id=rec.get("task_id") or rec.get("problem_id") or f"idx_{i}",
            category=category_label,          # e.g., "prompt1a" or "prompt"
            entry_point=entry_point,
            prompt_modified=final_prompt,
            prompt_original=prompt_orig,
            model_name=args.model,
            seed=args.seed,
            gen=gen_slim,                     # raw dict without duplicate text
            generated_text=text,              # single authoritative text field
            contains_code=has_code,
            extracted_code=code,
            extracted_questions=questions,
            latency_sec=dt,
        )
        row = asdict(ir)
        rows.append(row)

        # Per-item JSON for easy debugging
        with open(os.path.join(args.outdir, "by_item", f"{ir.task_id}.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[{i+1}/{n}] time={dt:.2f}s cat={category_label} code={has_code} q={len(questions)}")

    total_sec = time.time() - t0_all

    # Quick metrics
    total = len(rows)
    non_code = sum(1 for r in rows if not r["contains_code"])
    communication_rate = non_code / total if total > 0 else 0.0

    summary = {
        "model": args.model,
        "count": total,
        "communication_rate": communication_rate,
        "non_code_responses": non_code,
        "runtime_sec": total_sec,
        "categories": args.categories or "AUTO",
    }
    print("\n=== Quick Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save results
    save_jsonl(os.path.join(args.outdir, "results.jsonl"), rows)
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional: write a CSV with just the essentials
    try:
        import pandas as pd
        df = pd.DataFrame([
            {
                "task_id": r["task_id"],
                "category": r["category"],
                "contains_code": r["contains_code"],
                "n_questions": len(r["extracted_questions"]),
                "latency_sec": r["latency_sec"],
            }
            for r in rows
        ])
        df.to_csv(os.path.join(args.outdir, "results_light.csv"), index=False)
    except Exception:
        pass

    # Stubs for future Option-1 reliability work:
    with open(os.path.join(args.outdir, "NEXT_STEPS.md"), "w", encoding="utf-8") as f:
        f.write(
"""# Next Steps (Option 1: Evaluation Reliability)
- Add a judging step that labels each response as:
  - Questions only (Good/Fair/Bad)
  - Code only
  - Mixed/Invalid (both present) -> count as judge error
- Implement a multi-LLM committee (3+ different models) and take majority vote.
- Add simple rules:
  - If triple backticks or `def` present -> treat as code present.
  - If any '?' sentences and *no* backticks/def -> treat as questions present.
- Create a small human-labeled set (e.g., 60 items) to estimate judge accuracy and calibrate thresholds.
- Compute metrics:
  - Communication Rate = % non-code initial responses
  - Good Question Rate = % questions labeled 'Good' by judge(s)
  - (Later) Pass@1, Test Pass Rate by executing generated code against ground-truth tests.
"""
        )

if __name__ == "__main__":
    main()
