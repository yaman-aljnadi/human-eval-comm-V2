# HumanEvalComm-V2

HumanEvalComm-V2 is an extension of [HumanEvalComm](https://github.com/jie-jw-wu/human-eval-comm), a benchmark for evaluating the **communication competence of code generation models**.  

This version addresses limitations in evaluator reliability by:
- Updating the grading criteria for clarifying questions and recovery.
- Introducing a **committee of 3 LLM judges** instead of a single evaluator.  
This reduces bias and increases robustness in evaluation.

The benchmark builds on Wu & Fard’s *HumanEvalComm* paper.

---

## 📌 Motivation & Contribution

The original HumanEvalComm highlighted the importance of clarifying questions in code generation. However, it relied on **a single LLM evaluator**, which sometimes produced inconsistent or biased judgments.  

**HumanEvalComm-V2** improves on this by:
1. **Refined evaluation criteria** — better guidelines for detecting clarifying questions and assessing recovery quality.  
2. **Committee-based evaluation** — multiple LLM judges (1–3) evaluate responses, and their judgments are aggregated via majority vote/median.  

This yields more stable and trustworthy evaluation metrics.

---

## 📂 Repository Overview

- **`make_dataset_v2.py`**  
  Generates datasets from HumanEvalComm tasks.  
  Features:
  - Stable schema with strong IDs and provenance.
  - Detects clarifying questions vs. direct code.
  - Saves outputs as `results.jsonl`, `results.parquet` (optional), and per-sample artifacts.
  - Compatible with Hugging Face, Gemini, OpenRouter, and OpenAI APIs.  

- **`eval_committee_v2.py`**  
  Runs evaluations using 1–3 LLM judges.  
  Features:
  - Unified JSON schema for judgments:
    ```json
    {
      "is_question": true,
      "question_quality": 3,
      "minimal_answers": "Clarifies the missing info",
      "answer_quality": 2,
      "false_recovery": false,
      "reasoning": "Model asked a relevant question."
    }
    ```
  - Aggregates judgments into committee summaries.
  - Supports checkpointing, resuming, and configurable verbosity.

---
## 🖥️ Resources Used
* Operating System: Ubuntu 20.04.5 LTS
* CPU: Intel(R) Xeon(R) Silver 4214 @ 2.20GHz
* GPU: NVIDIA Tesla V100-PCIE-32GB


## ⚙️ Installation

Requirements:
- Python ≥ 3.9
- Recommended: virtual environment

Note! python 3.12.11 was used in the setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Setting up the API keys for LLM models
```bash
Required API keys (set as environment variables):

# Linux Based System:
export OPENAI_API_KEY='#' 
export GEMINI_API_KEY='#' 
export OPENROUTER_API_KEY='#' 

# Windows based System:
set OPENAI_API_KEY='#'
set GEMINI_API_KEY='#'
set OPENROUTER_API_KEY='#'

```

## ⚙️ Usage 
1. Generate dataset example
``` bash
    python make_dataset_v2.py \
    --model gemini-2.5-flash-lite \
    --categories 1a 1c 1p 2ac 2ap 2cp 3acp \
    --max-new-tokens 256 --temperature 1.0 --top-p 0.95 \
    --outdir ./runs/gemini-2.5-flash-lite
```
This produces:
``` bash
runs/gemini-2.5-flash-lite/
├── run_manifest.json
├── results.jsonl
├── summary.json
├── by_item/...
```

2. Run evaluation
Single evaluator
``` bash
# Single evaluator
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir ./runs/openai-gpt-3.5-turbo \
  --judges openai/gpt-3.5-turbo
    -v \
  --checkpoint-every 10 \
  --log-every 5 \
  --max-tokens 256 \
  --temperature 1.0 \
  --resume
```

Double evaluators
``` bash
# Double evaluators (majority/median aggregation)
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir ./runs/openai-gpt-3.5-turbo \
  --judges openai/gpt-3.5-turbo gemini/gemini-2.5-flash-lite
    -v \
  --checkpoint-every 10 \
  --log-every 5 \
  --max-tokens 256 \
  --temperature 1.0 \
  --resume
```

Triple evaluators
``` bash
# Triple evaluators
python eval_committee_v2.py \
  --results ./runs/openai-gpt-3.5-turbo/results.jsonl \
  --outdir ./runs/openai-gpt-3.5-turbo \
  --judges openai/gpt-3.5-turbo gemini/gemini-2.5-flash-lite deepseek-ai/deepseek-coder-6.7b-instruct \
    -v \
  --checkpoint-every 10 \
  --log-every 5 \
  --max-tokens 256 \
  --temperature 1.0 \
  --resume
```

Output:
1. committee_judgments.json — per-item evaluations.
2. committee_summary.json / .csv — aggregate metrics.

## 📊 Evaluation Metrics
* Communication Rate — % of responses with clarifying questions.
* Good Question Rate — % of high-quality clarifying questions (score = 3).
* Acceptable Question Rate — % of questions rated ≥ 2.
* Answer Quality — judge rating of provided recovery answers (1–3).
* False Recovery Rate — % of responses that recovered missing info without explicit questions.

## Results

## References
Wu, Jie & Fard, Amin. HumanEvalComm: Benchmarking the Communication Competence of Code Generation for LLMs and LLM Agents. arXiv preprint, 2024. [paper](https://arxiv.org/pdf/2406.00215)
Original repo: [jie-jw-wu/human-eval-comm](jie-jw-wu/human-eval-comm)
This continuation: [yaman-aljnadi/human-eval-comm-V2](yaman-aljnadi/human-eval-comm-V2)
