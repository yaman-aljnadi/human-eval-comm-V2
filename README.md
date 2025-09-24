# HumanEvalComm‑V2: Committee‑based LLM Evaluation for Ambiguity, Inconsistency, and Incompleteness


HumanEvalComm‑V2 extends **HumanEvalComm** to make LLM‑as‑judge evaluation **more reliable and reproducible**.  
Instead of a single judge, HumanEvalComm‑V2 evaluates each item with a **committee of 3 LLM judges**, aggregates by **majority vote**, and stores **per‑judge labels** for auditability. It also ships **revised evaluation parameters**, and better schema validation.

---

## What's new
- **Committee‑based judging (3 LLMs):** reduces variance/bias vs. a single judge; ties are surfaced explicitly.
- **Revised evaluation parameters & schemas:** sturdier parsing/validation; consistent JSON outputs.
- **Fully reproducible runs:** fixed seeds, deterministic temperatures, and structured artifacts (per‑item + aggregate).

> Targeted at assessing LLMs’ **communication competence** on prompts with **ambiguity (a)**, **inconsistency (c)**, and **incompleteness (p)**—individually and in combination.

---

## Dataset & Task
Using the **HumanEvalComm** dataset (771 items) which modifies HumanEval problems to introduce requirement defects. For each item, a generator LLM must either **ask clarifying question(s)** or **directly produce code**. Judges assess whether questions were asked and how good they are.  
Dataset (original): https://huggingface.co/datasets/jie-jw-wu/HumanEvalComm

**Categories & counts**

| Category | a | c | p | Count |
|---|:--:|:--:|:--:|---:|
| 1a  | ✓ |   |   | 164 |
| 1c  |   | ✓ |   | 164 |
| 1p  |   |   | ✓ | 164 |
| 2ac | ✓ | ✓ |   | 162 |
| 2cp |   | ✓ | ✓ |  34 |
| 2ap | ✓ |   | ✓ |  74 |
| 3apc| ✓ | ✓ | ✓ |   9 |
| **Total** |  |  |  | **771** |

---

## Benchmarked generator models (examples)
- `meta-llama/Meta-Llama-3-13B-Instruct`
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `gemini/gemini-2.5-flash-lite`
- `openai/gpt-3.5-turbo`

## LLM judges (committee)
- `openai/gpt-3.5-turbo`
- `gemini/gemini-2.5-flash-lite`
- `deepseek-ai/deepseek-coder-6.7b-instruct`

Aggregation: **simple majority** for Booleans and **mode** for ordinal labels (1–3). Ties: resolved in favor of the superior model;.

---

## Installation

**Requirements**
- Python **≥ 3.9** (repo originally tested with **3.12.11**)
- Recommended: virtual environment

```bash
pip install -r requirements.txt
```

**Environment variables (API keys)**

Linux/macOS:
```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

Windows (PowerShell):
```powershell
setx OPENAI_API_KEY "..."
setx GEMINI_API_KEY "..."
setx OPENROUTER_API_KEY "..."
```


---

## Quick start example

### 1) Generate model responses
```bash
python make_dataset_v2.py \
  --model gemini-2.5-flash-lite \
  --categories 1a 1c 1p 2ac 2ap 2cp 3apc \
  --max-new-tokens 256 \
  --temperature 1.0 \
  --top-p 0.95 \
  --outdir ./runs/gemini-2.5-flash-lite
```

This produces artifacts like:
```
runs/gemini-2.5-flash-lite/
├── run_manifest.json
├── results.jsonl
├── summary.json
└── by_item/
```

### 2) Evaluate with 1–3 judges example
**Single evaluator**
```bash
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir  ./runs/openai-gpt-3.5-turbo \
  --judges openai/gpt-3.5-turbo \
  -v \
  --checkpoint-every 10 \
  --log-every 5 \
  --max-tokens 256 \
  --temperature 1.0 \
  --resume
```

**Two evaluators (majority/median)**
```bash
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir  ./runs/openai-gpt-3.5-turbo \
  --judges openai/gpt-3.5-turbo gemini/gemini-2.5-flash-lite \
  -v \
  --checkpoint-every 10 \
  --log-every 5 \
  --max-tokens 256 \
  --temperature 1.0 \
  --resume
```

**Three evaluators (committee)**
```bash
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir  ./runs/openai-gpt-3.5-turbo \
  --judges openai/gpt-3.5-turbo gemini/gemini-2.5-flash-lite deepseek-ai/deepseek-coder-6.7b-instruct \
  -v \
  --checkpoint-every 10 \
  --log-every 5 \
  --max-tokens 256 \
  --temperature 1.0 \
  --resume
```

**Outputs**
- `committee_judgments.json` — per‑item, per‑judge JSON judgments
- `committee_summary.json` / `.csv` — aggregate metrics (+ tie counts)

---

## Metrics

Let each item *i* have final (aggregated) labels:
- `is_questionᵢ ∈ {0,1}` — whether the response asked clarifying question(s)
- `QQᵢ ∈ {1,2,3}` — question quality (1=Bad, 2=Fair, 3=Good)
- `FRᵢ ∈ {0,1}` — false recovery (no questions, but still “recovered” missing info)

Report:

- **Communication Rate (CR)** — share asking questions
- **Good Question Rate (GQR‑all)** — share with `QQᵢ=3` over all items
- **Good Question Rate (GQR‑asked)** — share with `QQᵢ=3` conditioned on asking
- **False Recovery Rate (FRR‑all / FRR‑noQ)** — spurious recovery without questions


---

## Data schemas

**Per‑response record (`ItemRow`)**
```json
{
  # Identifiers
  "record_id": "string",               // unique key, e.g. task_id::model::seed
  "task_id": "string",
  "category": "1a|1c|1p|2ac|2cp|2ap|3apc",
  "entry_point": "string|null",

  # Prompt provenance
  "prompt_field": "string",
  "prompt_text": "string",
  "prompt_final": "string",
  "prompt_sha256": "string",

  # Model provenance
  "model_name": "string",
  "seed": 0,
  "gen_params": {"temperature": 1.0, "top_p": 0.9},

  # Raw output
  "generated_text": "string",
  "gen_raw": {"...": "HF dict minus generated_text"},

  # Parsed output
  "contains_code": true,
  "code_detected_method": "regex|parser|none",
  "extracted_code": "string|null",
  "is_question": false,
  "extracted_questions": ["string", "..."],
  "question_count": 0,
  "first_question": "string|null",
  "question_chars": 0,

  # Timings
  "latency_sec": 0.0,

  # Evaluation
  "committee_label": "string|null",
  "evaluator_quality": 1,
  "evaluator_answers": "string|null"
}
```

**Per‑item committee judgments (`PerItemJudgment`)**
```json
{
  "record_id": "string",

  # Committee votes
  "committee_is_question": [true, false, true],
  "committee_question_quality": [1, 2, 3],
  "committee_minimal_answers": ["yes", "no", "maybe"],
  "committee_answer_quality": [1, 2, 3],
  "committee_false_recovery": [true, false, false],
  "committee_reasoning": ["...", "...", "..."],

  # Final aggregation
  "final_is_question": true,
  "final_question_quality": 3,
  "final_answer_quality": 2,
  "final_false_recovery": false
}
```

---

## Preliminary results (example, all categories, N=771)
| Model | CR ↑ | GQR‑all ↑ | GQR‑asked ↑ | FRR‑all ↓ | FRR‑noQ ↓ |
|---|---:|---:|---:|---:|---:|
| Meta‑Llama‑3‑13B‑Instruct | 38.5% | 17.6% | 45.8% | 6.5% | 10.5% |
| deepseek‑coder‑6.7b‑instruct | 35.1% | 25.6% | 72.7% | 20.9% | 32.2% |
| gemini‑2.5‑flash‑lite | 36.7% | 33.3% | 90.8% | 25.9% | 41.0% |
| openai‑gpt‑3.5‑turbo | 30.5% | 28.5% | 93.6% | 35.0% | 50.4% |

> Report point estimates with 95% CIs in papers; per‑judge rates and ties are available for auditability.

---

## Prompts (for transparency)

**Generation prompt (Python 3)**
```
You are an expert software developer who writes high quality code. With below information,
please either generate Python3 code (Respond directly with code only with markdown),
or ask clarifying questions:

{problem}
```

**Judge prompt (strict JSON output)**
```
You are a careful evaluator of clarifying-question quality and recovery.
You are given:
1) ORIGINAL problem
2) MODIFIED problem (may be ambiguous/inconsistent/incomplete)
3) MODEL RESPONSE (questions and/or code)

Return exactly:
{"is_question": <bool>,
 "question_quality": <1|2|3>,
 "minimal_answers": "<string>",
 "answer_quality": <1|2|3>,
 "false_recovery": <bool>,
 "reasoning": "<string>"}
```

---

## Reproducibility notes
- Max tokens: **256**, temperature: **1.0** (for both generation and judging)
- Deterministic seeds; strict schema validation
- Committee size/config and aggregation rule configurable (default: 3, majority/mode)
- Hardware used for sample runs: Ubuntu 20.04.5 LTS, Xeon Silver 4214 @ 2.20GHz, Tesla V100 32GB

---

## Citing
If you use this repo, please cite the original HumanEvalComm work and this V2 extension.

```text
Wu, Jie JW, and Fatemeh H. Fard. HumanEvalComm: Benchmarking the Communication
Competence of Code Generation for LLMs and LLM Agents. arXiv:2406.00215.
```

```text
Aljnadi, Yaman. HumanEvalComm‑V2: Committee‑based LLM Evaluation for Ambiguity,
Inconsistency, and Incompleteness. 2025.
```

---

## Acknowledgments
- HumanEvalComm authors & dataset maintainers
- Model providers: OpenAI, Google, DeepSeek
- Community contributors and reviewers

## License
Add your license here (e.g., MIT or Apache‑2.0).
