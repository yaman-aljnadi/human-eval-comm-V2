#!/usr/bin/env python3
"""
make_dataset_v2.py — HumanEvalComm runner with robust, schema-stable saving.

What’s new vs your version:
  - Strong IDs: record_id = "<task_id>::<cat>::<model>::seed<seed>"
  - Provenance: model, sampling params, prompt hash, dataset name/version
  - Rich per-item fields for later committee aggregation:
      * is_question (bool), question_count, first_question, question_chars
      * code_detected_method ("fenced" | "def-scan" | "none")
      * timings + token settings
  - Atomic writes + --append support for results.jsonl (+ optional .gz)
  - Optional Parquet export for analysis (pandas/pyarrow if available)
  - Summary saved with exact counts + schema version
  - Everything under outdir/{by_item,artifacts} with a run_manifest.json

NOTE: This script still *generates* model outputs. Your separate script can
ingest results.jsonl (or Parquet) to run the 3-model committee & add ratings.

Usage example:
  python make_dataset_v2.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --categories 1a 1c 1p 2ac 2ap 2cp 3acp \
    --max-new-tokens 256 --temperature 1.0 --top-p 0.95 \
    --outdir ./runs/deepseek_coder_allcats

  python make_dataset_v2.py \
    --model meta-llama/Llama-3.1-8B \
    --categories 1a 1c 1p 2ac 2ap 2cp 3acp \
    --max-new-tokens 256 --temperature 1.0 --top-p 0.95 \
    --outdir ./runs/meta-llama/Llama-3.1-8B

    python make_dataset_v2.py \
    --model gemini-2.5-flash-lite \
    --categories 1a 1c 1p 2ac 2ap 2cp 3acp \
    --max-new-tokens 256 --temperature 1.0 --top-p 0.95 \
    --outdir ./runs/gemini-2.5-flash-lite
    
    python make_dataset_v2.py \
        --model deepseek/deepseek-r1:free \
        --categories 1a 1c 1p 2ac 2ap 2cp 3acp \
        --max-new-tokens 256 --temperature 1.0 --top-p 0.95 \
        --outdir ./runs/openrouter/deepseek-r1-free

Outputs:
  outdir/
    run_manifest.json            # run config + dataset provenance + schema ver
    results.jsonl[.gz]           # one line per (task_id, category)
    results.parquet              # optional if --parquet
    by_item/<record_id>.json     # per-sample artifact
    summary.json                 # quick aggregates
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import google.generativeai as genai

import random
import numpy as np

# ---------- regexes & helpers ----------
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
Q_SENT_RE = re.compile(r"[^.?!]*\?")

PROMPT_VARIANTS_ALL = ["1a", "1c", "1p", "2ac", "2ap", "2cp", "3acp"]

PAPER_PROMPT_TEMPLATE = (
    "You are an expert software developer who writes high quality code. "
    "With below information, please either generate Python3 code (Respond directly with code only with markdown), "
    "or ask clarifying questions:\n\n{problem}"
)



SCHEMA_VERSION = "comm-v2.1"
# ---------- seeding ----------
def set_all_seeds(seed: int):
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

# ---------- small utils ----------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_dirs(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "by_item"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "artifacts"), exist_ok=True)

def atomic_write(path: str, data: str, mode: str = "w", encoding: str = "utf-8"):
    """Atomic write that also ensures the parent directory exists."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp, mode, encoding=encoding) as f:
        f.write(data)
    os.replace(tmp, path)

def append_jsonl(path: str, rows: List[Dict[str, Any]], gzip: bool = False):
    if gzip:
        import gzip as _gzip
        mode = "ab"
        with _gzip.open(path, mode) as f:
            for r in rows:
                line = (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")
                f.write(line)
    else:
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

def detect_code_and_questions(text: str):
    """Return: contains_code(bool), extracted_code(str|None),
               questions(list[str]), code_detected_method(str)"""
    method = "none"
    m = CODE_FENCE_RE.search(text)
    if m:
        method = "fenced"
        return True, m.group(1).strip(), [], method
    if DEF_RE.search(text):
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if DEF_RE.match(ln):
                method = "def-scan"
                return True, "\n".join(lines[i:]).strip(), [], method
    # No code detected => try extract questions (filter out obvious code-y lines)
    chunks = [c.strip() for c in text.split('?') if c.strip()]
    qs = []
    for c in chunks:
        if 'def ' in c or 'return ' in c or '```' in c:
            continue
        qs.append(c + '?')
    for m in Q_SENT_RE.finditer(text):
        q = m.group(0).strip()
        if q and '```' not in q and q not in qs:
            qs.append(q)
    # de-dup
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return False, None, out, method

# ---------- HF generation ----------
def lazy_imports():
    try:
        import torch  # noqa: F401
    except Exception:
        print("ERROR: PyTorch is required.", file=sys.stderr)
        raise
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa: F401
    except Exception:
        print("ERROR: transformers is required.", file=sys.stderr)
        raise
    try:
        from datasets import load_dataset  # noqa: F401
    except Exception:
        print("ERROR: datasets is required.", file=sys.stderr)
        raise

def prepare_generator(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    # If available, these can improve memory/throughput; ignore if unsupported
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

def prepare_generator_gemini(model_name: str, cfg) -> Any:
    """
    Returns a callable with a HF-like interface:
      gen_fn(final_prompt, **ignored) -> {"generated_text": str, "gen_raw": dict}
    """
    if genai is None:
        raise RuntimeError("google-generativeai is not installed. pip install google-generativeai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is not set.")
    genai.configure(api_key=api_key)

    # Build the model
    model = genai.GenerativeModel(model_name)

    # Map your config to Gemini generation_config
    generation_config = {
        "max_output_tokens": int(getattr(cfg, "max_new_tokens", 256)),
        "temperature": float(getattr(cfg, "temperature", 1.0)),
        "top_p": float(getattr(cfg, "top_p", 0.9)),
        # You can add "top_k": N if you want; not in your CLI today
        # "response_mime_type": "text/plain",
    }

    def gen_fn(prompt_text: str, **_):
        t0 = time.time()
        resp = model.generate_content(
            prompt_text,
            generation_config=generation_config,
        )
        dt = time.time() - t0

        # Pull a plain text; fall back defensively
        try:
            text = resp.text or ""
        except Exception:
            text = ""

        # Make a slim raw dict for parity with your HF "gen_raw"
        try:
            usage = getattr(resp, "usage_metadata", None)
            candidates = getattr(resp, "candidates", None)
            gen_raw = {
                "usage_metadata": getattr(usage, "__dict__", usage) if usage else None,
                "num_candidates": len(candidates) if candidates else 0,
                "finish_reasons": [getattr(c, "finish_reason", None) for c in (candidates or [])],
            }
        except Exception:
            gen_raw = {}

        # Mirror HF pipeline return shape minimally
        return [{"generated_text": text, "gen_raw": gen_raw, "latency_sec": dt}]

    # Attach attrs used in your HF path to keep calls uniform
    gen_fn.tokenizer = type("T", (), {"pad_token_id": None, "eos_token_id": None})()
    return gen_fn


def prepare_generator_openrouter(model_name: str, cfg) -> Any:
    """
    Returns a callable with an HF-like interface:
      gen_fn(final_prompt, **ignored) -> [{"generated_text": str, "gen_raw": dict, "latency_sec": float}]
    Requires:
      - pip install openai>=1.0.0
      - env: OPENROUTER_API_KEY
      - optional env: OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required for OpenRouter. pip install openai") from e

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY env var is not set for OpenRouter.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Map your config to OpenAI-style params (sane defaults)
    generation_kwargs = {
        "temperature": float(getattr(cfg, "temperature", 1.0)),     # steadier default
        "top_p": float(getattr(cfg, "top_p", 0.9)),
        # OpenRouter uses max_tokens (new tokens). Give reasoning models room.
        "max_tokens": int(getattr(cfg, "max_new_tokens", 256)),
    }

    # Optional attribution headers for openrouter.ai rankings
    extra_headers = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_name = os.getenv("OPENROUTER_SITE_NAME")
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if site_name:
        extra_headers["X-Title"] = site_name

    def _extract_text_from_message(msg) -> str:
        """Robustly extract visible text from an OpenRouter message object."""
        # Case 1: plain string content
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content:
            return content

        # Case 2: list of parts (content array)
        if isinstance(content, list):
            parts = []
            for p in content:
                p_type = getattr(p, "type", None) if hasattr(p, "type") else (p.get("type") if isinstance(p, dict) else None)
                p_text = getattr(p, "text", None) if hasattr(p, "text") else (p.get("text") if isinstance(p, dict) else None)
                # Prefer standard textual parts; some providers label final text as "output_text"
                if p_text and (p_type in (None, "text", "output_text")):
                    parts.append(p_text)
            if parts:
                return "".join(parts)

        # Case 3: last resort—if a provider exposed a 'reasoning' field and content is empty
        reasoning = getattr(msg, "reasoning", None)
        if isinstance(reasoning, str) and reasoning:
            return reasoning

        return ""

    def gen_fn(prompt_text: str, **_):
        t0 = time.time()
        # IMPORTANT: Ask for less visible 'thinking' and exclude it from the response.
        # Many reasoning models (e.g., deepseek-r1) honor this on OpenRouter.
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            reasoning={"exclude": True, "effort": "low"},
            extra_headers=extra_headers or None,
            **generation_kwargs,
        )
        latency = time.time() - t0

        # Defensive extraction
        try:
            msg = resp.choices[0].message
        except Exception:
            msg = None

        text = _extract_text_from_message(msg) if msg is not None else ""

        # Slim raw for parity with your HF dict
        try:
            usage = getattr(resp, "usage", None)
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            gen_raw = {
                "usage": usage.__dict__ if getattr(usage, "__dict__", None) else dict(usage or {}),
                "finish_reason": finish_reason,
                "model": getattr(resp, "model", None),
                "id": getattr(resp, "id", None),
            }
        except Exception:
            gen_raw = {}

        return [{"generated_text": text, "gen_raw": gen_raw, "latency_sec": latency}]

    # Tokenizer placeholders to align with HF path
    gen_fn.tokenizer = type("T", (), {"pad_token_id": None, "eos_token_id": None})()
    return gen_fn

# ---------- data classes ----------
@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0

@dataclass
class ItemRow:
    # Identifiers
    record_id: str
    task_id: str
    category: str
    entry_point: Optional[str]

    # Prompt provenance
    prompt_field: str
    prompt_text: str
    prompt_final: str
    prompt_sha256: str

    # Model provenance
    model_name: str
    seed: int
    gen_params: Dict[str, Any]

    # Raw output
    generated_text: str
    gen_raw: Dict[str, Any]  # slimmed HF dict (no generated_text)

    # Parsed output
    contains_code: bool
    code_detected_method: str
    extracted_code: Optional[str]
    is_question: bool
    extracted_questions: List[str]
    question_count: int
    first_question: Optional[str]
    question_chars: int

    # Timings
    latency_sec: float

    # Placeholders reserved for committee/evaluator later
    committee_label: Optional[str] = None          # e.g., "code" | "question" | "tie" | "inconclusive"
    evaluator_quality: Optional[int] = None        # 1/2/3 (paper), to be filled later
    evaluator_answers: Optional[str] = None        # filled later

# ---------- main ----------
def main():
    # For Gemini rate limits 
    REQUESTS_PER_MINUTE = 15
    SECONDS_PER_REQUEST = 60.0 / REQUESTS_PER_MINUTE
    last_request_time = 0.0

    lazy_imports()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--categories", nargs="*", default=PROMPT_VARIANTS_ALL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--outdir", required=True)

    # Saving options
    parser.add_argument("--append", action="store_true", help="Append to results.jsonl(.gz) instead of overwriting.")
    parser.add_argument("--gzip", action="store_true", help="Write results.jsonl.gz instead of results.jsonl.")
    parser.add_argument("--parquet", action="store_true", help="Also write results.parquet if pandas/pyarrow available.")

    args = parser.parse_args()
    set_all_seeds(args.seed)

    # Provenance + schema
    ensure_dirs(args.outdir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "jie-jw-wu/HumanEvalComm",
        "split": args.split,
        "categories": args.categories,
        "run_id": uuid.uuid4().hex,
        "model": args.model,
        "seed": args.seed,
        "gen_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": True,
        },
        "created_utc": int(time.time()),
    }
    atomic_write(os.path.join(args.outdir, "run_manifest.json"), json.dumps(manifest, indent=2))

    # Save initial config too (back-compat with your layout)
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset("jie-jw-wu/HumanEvalComm", split=args.split)
    n_base = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[load] base items: {n_base} (of {len(ds)}) ; categories per item: {args.categories}")

    # Prepare generator
    use_gemini = args.model.strip().lower().startswith("gemini")
    use_openrouter = (":" in args.model)

    if use_gemini:
        gen_pipe = prepare_generator_gemini(args.model, cfg=None)
    elif use_openrouter:
        gen_pipe = prepare_generator_openrouter(args.model, cfg=None)
    else:
        gen_pipe = prepare_generator(args.model)

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        repetition_penalty=args.repetition_penalty,
    )

    if use_gemini:
        gen_pipe = prepare_generator_gemini(args.model, cfg)
    elif use_openrouter:
        gen_pipe = prepare_generator_openrouter(args.model, cfg)


    # Paths
    results_path = os.path.join(args.outdir, "results.jsonl.gz" if args.gzip else "results.jsonl")
    by_item_dir = os.path.join(args.outdir, "by_item")

    # If not appending, (over)write an empty file atomically to start fresh
    if not args.append and os.path.exists(results_path):
        os.remove(results_path)

    all_rows_for_parquet: List[Dict[str, Any]] = []
    t0_all = time.time()
    expanded = 0

    for i in range(n_base):
        item = ds[i]
        entry_point = item.get("entry_point")
        task_id = item.get("task_id") or item.get("problem_id") or f"idx_{i}"

        for cat in args.categories:
            field = f"prompt{cat}"
            problem = item.get(field)
            if not isinstance(problem, str) or not problem.strip():
                # Some combos (2ac/2cp/2ap/3acp) may be missing; skip gracefully
                continue

            final_prompt = PAPER_PROMPT_TEMPLATE.format(problem=problem.strip())
            prompt_hash = sha256_text(final_prompt)

            if use_gemini or use_openrouter:
                now = time.time()
                elapsed = now - last_request_time
                if elapsed < SECONDS_PER_REQUEST:
                    time.sleep(SECONDS_PER_REQUEST - elapsed)
                last_request_time = time.time()

                _out = gen_pipe(final_prompt)[0]
                out = {"generated_text": _out.get("generated_text", "")}
                out_slim = _out.get("gen_raw", {})
                dt = _out.get("latency_sec", None)
            else:
                t0 = time.time()
                _out = gen_pipe(
                    final_prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    do_sample=cfg.do_sample,
                    repetition_penalty=cfg.repetition_penalty,
                    pad_token_id=gen_pipe.tokenizer.pad_token_id,
                    eos_token_id=gen_pipe.tokenizer.eos_token_id,
                )[0]
                dt = time.time() - t0
                out = _out
                out_slim = dict(_out)
                out_slim.pop("generated_text", None)

            text = out.get("generated_text", "") or str(out)
            out_slim = dict(out)
            out_slim.pop("generated_text", None)

            has_code, code, questions, method = detect_code_and_questions(text)
            is_question = not has_code
            q_count = len(questions)
            first_q = questions[0] if q_count else None
            q_chars = sum(len(q) for q in questions)

            record_id = f"{task_id}::{cat}::{args.model}::seed{args.seed}"

            row = ItemRow(
                record_id=record_id,
                task_id=task_id,
                category=cat,
                entry_point=entry_point,
                prompt_field=field,
                prompt_text=problem,
                prompt_final=final_prompt,
                prompt_sha256=prompt_hash,
                model_name=args.model,
                seed=args.seed,
                gen_params=asdict(cfg),
                generated_text=text,
                gen_raw=out_slim,
                contains_code=has_code,
                code_detected_method=method,
                extracted_code=code,
                is_question=is_question,
                extracted_questions=questions,
                question_count=q_count,
                first_question=first_q,
                question_chars=q_chars,
                latency_sec=dt,
            )
            row_dict = asdict(row)

            # Per-item JSON (atomic)
            fn = f"{task_id}__{cat}.json"
            with open(os.path.join(args.outdir, "by_item", fn), "w", encoding="utf-8") as f:
                json.dump(row_dict, f, ensure_ascii=False, indent=2)

            # Append to JSONL (streaming)
            append_jsonl(results_path, [row_dict], gzip=args.gzip)

            # Keep for parquet
            all_rows_for_parquet.append(row_dict)
            expanded += 1

        if (i + 1) % 10 == 0 or (i + 1) == n_base:
            print(f"[{i+1}/{n_base}] expanded rows so far: {expanded}")

    total_sec = time.time() - t0_all

    # Quick metrics (paper: Communication Rate = non-code / total)
    total = expanded
    non_code = sum(1 for r in all_rows_for_parquet if r["is_question"])
    comm_rate = (non_code / total) if total else 0.0

    summary = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "base_items": n_base,
        "expanded_rows": total,
        "communication_rate": comm_rate,   # matches paper definition
        "non_code_responses": non_code,
        "runtime_sec": total_sec,
        "categories": args.categories,
        "results_file": os.path.basename(results_path),
    }
    atomic_write(os.path.join(args.outdir, "summary.json"), json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Optional Parquet
    if args.parquet:
        try:
            import pandas as pd
            df = pd.DataFrame(all_rows_for_parquet)
            df.to_parquet(os.path.join(args.outdir, "results.parquet"), index=False)
            print("[write] results.parquet")
        except Exception as e:
            print(f"[warn] parquet export skipped: {e}")

if __name__ == "__main__":
    main()
