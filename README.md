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

## ⚙️ Installation

Requirements:
- Python ≥ 3.9
- Recommended: virtual environment

Install dependencies:
```bash
pip install -r requirements.txt
