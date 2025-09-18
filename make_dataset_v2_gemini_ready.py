#!/usr/bin/env python3
# make_dataset_v2_gemini_ready.py
import argparse, json, os, re, sys, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# ---------- unchanged helpers (code/question extraction) ----------
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
Q_SENT_RE = re.compile(r"[^.?!]*\?")

PROMPT_VARIANTS_ALL = ["1a","1c","1p","2ac","2ap","2cp","3acp"]
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
    repetition_penalty: float = 1.0  # not used by Gemini, kept for HF parity

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
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "by_item"), exist_ok=True)

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- backends ----------
def prepare_generator_hf(model_name: str):
    # Optional HF support so you can A/B vs Gemini if you want
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception:
        print("Install transformers to use --backend hf", file=sys.stderr)
        raise
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tok, device_map="auto", return_full_text=True)

def generate_one_hf(gen_pipe, prompt: str, cfg: GenConfig) -> Dict[str, Any]:
    out = gen_pipe(
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        repetition_penalty=cfg.repetition_penalty,
        pad_token_id=gen_pipe.tokenizer.pad_token_id,
        eos_token_id=gen_pipe.tokenizer.eos_token_id,
    )[0]
    # normalize to a shared shape
    return {"generated_text": out.get("generated_text",""), "_raw": out}

def prepare_generator_gemini(api_key: Optional[str] = None):
    try:
        from google import genai
    except Exception:
        print("Install google-genai to use --backend gemini (pip install google-genai)", file=sys.stderr)
        raise
    # If GEMINI_API_KEY env var is set, Client() will pick it up automatically.
    # You can also pass api_key explicitly.
    client = genai.Client(api_key=api_key)
    return client

def generate_one_gemini(client, model: str, prompt: str, cfg: GenConfig) -> Dict[str, Any]:
    """
    Uses non-streaming models.generate_content.
    Maps our knobs to Gemini's generation_config.
    """
    generation_config = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_output_tokens": cfg.max_new_tokens,
    }
    # Plain string works for `contents` in the new SDK.
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        generation_config=generation_config,
    )
    text = getattr(resp, "text", None)
    if text is None:
        # Some responses may come back in .candidates/.content.parts format
        try:
            # best-effort flatten
            text = "".join(p.text for p in resp.candidates[0].content.parts if hasattr(p, "text"))
        except Exception:
            text = ""
    return {"generated_text": text, "_raw": resp.to_dict() if hasattr(resp, "to_dict") else {}}

# ---------- main ----------
def main():
    import random
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gemini","hf"], default="gemini")
    ap.add_argument("--model", required=True, help="e.g., gemini-2.5-flash or an HF model id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--categories", nargs="*", default=PROMPT_VARIANTS_ALL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)  # HF only
    ap.add_argument("--outdir", required=True)
    # Optional: pass API key explicitly (otherwise use GEMINI_API_KEY env)
    ap.add_argument("--gemini-api-key", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
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

    ds = load_dataset("jie-jw-wu/HumanEvalComm", split=args.split)
    n_base = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[load] base items: {n_base} (of {len(ds)}) ; categories: {args.categories}")

    if args.backend == "gemini":
        client = prepare_generator_gemini(api_key=args.gemini_api_key)
        gen_ctx = ("gemini", client)  # tuple tag
    else:
        gen_pipe = prepare_generator_hf(args.model)
        gen_ctx = ("hf", gen_pipe)

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
                continue

            final_prompt = PAPER_PROMPT_TEMPLATE.format(problem=problem.strip())

            t0 = time.time()
            if gen_ctx[0] == "gemini":
                out = generate_one_gemini(gen_ctx[1], args.model, final_prompt, cfg)
            else:
                out = generate_one_hf(gen_ctx[1], final_prompt, cfg)
            dt = time.time() - t0

            text = out.get("generated_text", "") or ""
            has_code = contains_code_block(text)
            code = extract_code(text) if has_code else None
            questions = [] if has_code else extract_questions(text)

            rec = {
                "task_id": task_id,
                "category": cat,
                "entry_point": entry_point,
                "prompt_field": field,
                "prompt_text": problem,
                "prompt_final": final_prompt,
                "model_name": args.model,
                "backend": args.backend,
                "gen_raw": out.get("_raw", {}),
                "generated_text": text,
                "contains_code": has_code,
                "extracted_code": code,
                "extracted_questions": questions,
                "latency_sec": dt,
                "seed": args.seed,
            }
            rows.append(rec)

            fn = f"{task_id}__{cat}.json"
            with open(os.path.join(args.outdir, "by_item", fn), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

        if (i + 1) % 10 == 0 or (i + 1) == n_base:
            print(f"[{i+1}/{n_base}] expanded rows so far: {len(rows)}")

    total_sec = time.time() - t0_all
    total = len(rows)
    non_code = sum(1 for r in rows if not r["contains_code"])
    comm_rate = (non_code / total) if total else 0.0

    save_jsonl(os.path.join(args.outdir, "results.jsonl"), rows)
    summary = {
        "model": args.model,
        "backend": args.backend,
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
