#!/usr/bin/env python3
"""
make_dataset_v2.py — HumanEvalComm runner that iterates ALL categories and uses the paper's prompt.

Fixes:
  1) Iterates every requested category field per example (e.g., prompt1a, 1c, 1p, 2ac, 2ap, 2cp, 3acp).
  2) Uses the paper's prompt template verbatim.
  3) Leaves *evaluation* to a separate script (see eval_committee.py).

Usage example:
  python make_dataset_v2.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --categories 1a 1c 1p 2ac 2ap 2cp 3acp \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.95 \
    --outdir ./runs/deepseek_coder_allcats

Outputs:
  outdir/
    run_config.json
    results.jsonl          # 1 line per (task_id, category)
    by_item/<taskid>__<cat>.json
    summary.json
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# ---------- tiny utils ----------
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
Q_SENT_RE = re.compile(r"[^.?!]*\?")

PROMPT_VARIANTS_ALL = ["1a", "1c", "1p", "2ac", "2ap", "2cp", "3acp"]

PAPER_PROMPT_TEMPLATE = (
    "You are an expert software developer who writes high quality code. "
    "With below information, please either generate Python3 code (Respond directly with code only with markdown), "
    "or ask clarifying questions:\n\n{problem}"
)

@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0

@dataclass
class ItemOut:
    task_id: str
    category: str                  # e.g., 1a, 2ac, ...
    entry_point: Optional[str]
    prompt_field: str              # e.g., "prompt1a"
    prompt_text: str               # the selected modified description
    prompt_final: str              # paper prompt after {problem} substitution
    model_name: str
    gen_raw: Dict[str, Any]        # HF output dict (minus generated_text)
    generated_text: str
    contains_code: bool
    extracted_code: Optional[str]
    extracted_questions: List[str]
    latency_sec: float
    seed: int

def contains_code_block(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True
    return DEF_RE.search(text) is not None

def extract_code(text: str) -> Optional[str]:
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if DEF_RE.search(text):
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if DEF_RE.match(ln):
                return "\n".join(lines[i:]).strip()
    return None

def extract_questions(text: str) -> List[str]:
    chunks = [c.strip() for c in text.split('?') if c.strip()]
    qs = []
    for c in chunks:
        if 'def ' in c or 'return ' in c or '```' in c:
            continue
        qs.append(c + '?')
    for m in Q_SENT_RE.finditer(text):
        q = m.group(0).strip()
        if q and q not in qs and '```' not in q:
            qs.append(q)
    # de-dup
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def lazy_imports():
    try:
        import torch  # noqa
    except Exception:
        print("ERROR: PyTorch is required. pip install torch --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        raise
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa
    except Exception:
        print("ERROR: transformers is required. pip install transformers accelerate bitsandbytes", file=sys.stderr)
        raise
    try:
        from datasets import load_dataset  # noqa
    except Exception:
        print("ERROR: datasets is required. pip install datasets", file=sys.stderr)
        raise

def prepare_generator(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
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
    return gen_pipe

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
    lazy_imports()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--categories", nargs="*", default=PROMPT_VARIANTS_ALL,
                        help="Which prompt variants to include. Default: all (1a 1c 1p 2ac 2ap 2cp 3acp)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of base problems (before category expansion)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        repetition_penalty=args.repetition_penalty,
    )
    ensure_outdir(args.outdir)
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    from datasets import load_dataset
    ds = load_dataset("jie-jw-wu/HumanEvalComm", split=args.split)
    n_base = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[load] base items: {n_base} (of {len(ds)}) ; categories per item: {args.categories}")

    gen_pipe = prepare_generator(args.model)

    rows: List[Dict[str, Any]] = []
    t0_all = time.time()

    for i in range(n_base):
        item = ds[i]
        entry_point = item.get("entry_point")
        task_id = item.get("task_id") or item.get("problem_id") or f"idx_{i}"

        for cat in args.categories:
            field = f"prompt{cat}"
            problem = item.get(field)
            if not isinstance(problem, str) or not problem.strip():
                # Some combos (2ac/2cp/2ap/3acp) may be missing for certain tasks; skip gracefully.
                continue

            final_prompt = PAPER_PROMPT_TEMPLATE.format(problem=problem.strip())

            t0 = time.time()
            gen = generate_one(gen_pipe, final_prompt, cfg)
            dt = time.time() - t0

            text = gen.get("generated_text", "") or str(gen)
            gen_slim = dict(gen)
            gen_slim.pop("generated_text", None)

            has_code = contains_code_block(text)
            code = extract_code(text) if has_code else None
            questions = [] if has_code else extract_questions(text)

            rec = ItemOut(
                task_id=task_id,
                category=cat,
                entry_point=entry_point,
                prompt_field=field,
                prompt_text=problem,
                prompt_final=final_prompt,
                model_name=args.model,
                gen_raw=gen_slim,
                generated_text=text,
                contains_code=has_code,
                extracted_code=code,
                extracted_questions=questions,
                latency_sec=dt,
                seed=args.seed,
            )
            row = asdict(rec)
            rows.append(row)

            # Per item+cat
            fn = f"{task_id}__{cat}.json"
            with open(os.path.join(args.outdir, "by_item", fn), "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)

        if (i + 1) % 10 == 0 or (i + 1) == n_base:
            print(f"[{i+1}/{n_base}] expanded rows so far: {len(rows)}")

    total_sec = time.time() - t0_all

    # Quick metrics
    total = len(rows)
    non_code = sum(1 for r in rows if not r["contains_code"])
    comm_rate = (non_code / total) if total else 0.0

    save_jsonl(os.path.join(args.outdir, "results.jsonl"), rows)
    summary = {
        "model": args.model,
        "base_items": n_base,
        "expanded_rows": total,
        "communication_rate": comm_rate,
        "non_code_responses": non_code,
        "runtime_sec": total_sec,
        "categories": args.categories,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
