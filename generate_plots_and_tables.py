#!/usr/bin/env python3
"""
Generate paper-ready plots and tables for communication competence metrics
using the committee judgments produced from your LLM judges.

Inputs (expected filenames in the input directory):
- committee_judgments_openai-gpt-3.5-turbo.json
- committee_judgments_Meta-Llama-3-13B-Instruct.json
- committee_judgments_gemini-2.5-flash-lite.json
(Optional)
- Judges_Final_Results.csv
- results_openai-gpt-3.5-turbo.jsonl
- resultsMeta-Llama-3-13B-Instruct.jsonl
- resultsgemini-2.5-flash-lite.jsonl

Outputs (written to --outdir, default: paper_assets/):
- metrics_overall.csv
- metrics_by_category.csv
- metrics_overall.tex (LaTeX table)
- PNG figures:
    - fig_comm_rate.png
    - fig_gqr_all.png
    - fig_gqr_conditional.png
    - fig_frr_all.png
    - fig_frr_conditional.png
- Per-category pivot tables (CSV):
    - table_comm_rate_by_category.csv
    - table_gqr_all_by_category.csv
    - table_frr_all_by_category.csv

Usage:
    python generate_plots_and_tables.py --input_dir /path/to/data --outdir /path/to/out

Notes:
- Uses matplotlib (no seaborn), one chart per figure, and no explicit colors.
- Metrics are computed from committee JSON fields:
  final_is_question, final_question_quality, final_false_recovery
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- I/O Helpers -------------------------------

def load_committee_json(p: Path, model_name: str) -> pd.DataFrame:
    """Load a committee judgments JSON into a DataFrame and parse record_id."""
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["model"] = model_name

    # Parse record_id: "<task_id>::<cat>::<model>::seed<seed>"
    def parse_record_id(rid: str):
        parts = rid.split("::") if isinstance(rid, str) else []
        task_id = parts[0] if len(parts) > 0 else None
        cat = parts[1] if len(parts) > 1 else None
        mdl = parts[2] if len(parts) > 2 else None
        seed = parts[3].replace("seed", "") if len(parts) > 3 and isinstance(parts[3], str) else None
        return pd.Series({"task_id": task_id, "category": cat, "id_model": mdl, "seed": seed})

    if "record_id" in df.columns:
        pid = df["record_id"].apply(parse_record_id)
        df = pd.concat([df, pid], axis=1)
    else:
        for col in ["task_id", "category", "id_model", "seed"]:
            if col not in df.columns:
                df[col] = None

    # Ensure expected fields exist (some judge configs may omit them)
    for col in ["final_is_question", "final_question_quality", "final_answer_quality", "final_false_recovery"]:
        if col not in df.columns:
            df[col] = None

    return df


# ------------------------------- Metrics -----------------------------------

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall metrics by model."""
    rows = []
    for model, g in df.groupby("model"):
        n = len(g)
        asked = g["final_is_question"] == True
        not_asked = g["final_is_question"] == False
        good_q = (g["final_question_quality"] == 3) & asked
        f_recovery = (g["final_false_recovery"] == True) & not_asked

        cr = asked.mean() if n else float("nan")
        gqr_all = good_q.sum() / n if n else float("nan")
        gqr_cond = (good_q.sum() / asked.sum()) if asked.sum() else float("nan")
        frr_all = f_recovery.sum() / n if n else float("nan")
        frr_cond = (f_recovery.sum() / not_asked.sum()) if not_asked.sum() else float("nan")

        rows.append({
            "model": model,
            "n_items": n,
            "communication_rate": cr,
            "good_question_rate_all": gqr_all,
            "good_question_rate_conditional": gqr_cond,
            "false_recovery_rate_all": frr_all,
            "false_recovery_rate_conditional": frr_cond,
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def compute_metrics_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the same metrics with a (model, category) breakdown."""
    items = []
    for (model, category), g in df.groupby(["model", "category"]):
        n = len(g)
        asked = g["final_is_question"] == True
        not_asked = g["final_is_question"] == False
        good_q = (g["final_question_quality"] == 3) & asked
        f_recovery = (g["final_false_recovery"] == True) & not_asked

        items.append({
            "model": model,
            "category": category,
            "n_items": n,
            "communication_rate": asked.mean() if n else float("nan"),
            "good_question_rate_all": (good_q.sum() / n) if n else float("nan"),
            "good_question_rate_conditional": (good_q.sum() / asked.sum()) if asked.sum() else float("nan"),
            "false_recovery_rate_all": (f_recovery.sum() / n) if n else float("nan"),
            "false_recovery_rate_conditional": (f_recovery.sum() / not_asked.sum()) if not_asked.sum() else float("nan"),
        })
    return pd.DataFrame(items).sort_values(["model", "category"]).reset_index(drop=True)


# ------------------------------- Plotting ----------------------------------

def save_bar(series: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    """Create a simple single-axis bar chart with matplotlib defaults."""
    plt.figure(figsize=(7, 4.5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def to_latex_pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{100*x:.1f}\\%"


# ------------------------------- Main --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=".", help="Directory containing committee JSON files")
    ap.add_argument("--outdir", type=str, default="paper_assets", help="Output directory")
    # Optional: override default file basenames
    ap.add_argument("--file_gpt", type=str, default="committee_judgments_openai-gpt-3.5-turbo.json")
    ap.add_argument("--file_llama", type=str, default="committee_judgments_Meta-Llama-3-13B-Instruct.json")
    ap.add_argument("--file_gemini", type=str, default="committee_judgments_gemini-2.5-flash-lite.json")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "openai-gpt-3.5-turbo": in_dir / args.file_gpt,
        "Meta-Llama-3-13B-Instruct": in_dir / args.file_llama,
        "gemini-2.5-flash-lite": in_dir / args.file_gemini,
    }

    # Load available committee files
    frames: List[pd.DataFrame] = []
    for model, path in model_files.items():
        if path.exists():
            df = load_committee_json(path, model)
            frames.append(df)
        else:
            print(f"[WARN] Missing committee file for {model}: {path} (skipping)")

    if not frames:
        raise FileNotFoundError("No committee JSON files found. Check --input_dir and filenames.")

    df_committee = pd.concat(frames, ignore_index=True)
    metrics_overall = compute_metrics(df_committee)
    metrics_by_cat = compute_metrics_by_category(df_committee)

    # Save metrics tables
    overall_csv = outdir / "metrics_overall.csv"
    bycat_csv = outdir / "metrics_by_category.csv"
    metrics_overall.to_csv(overall_csv, index=False)
    metrics_by_cat.to_csv(bycat_csv, index=False)
    print(f"[OK] Wrote {overall_csv}")
    print(f"[OK] Wrote {bycat_csv}")

    # LaTeX table (overall)
    latex_rows = []
    for _, r in metrics_overall.iterrows():
        latex_rows.append([
            r["model"],
            int(r["n_items"]),
            to_latex_pct(r["communication_rate"]),
            to_latex_pct(r["good_question_rate_all"]),
            to_latex_pct(r["good_question_rate_conditional"]),
            to_latex_pct(r["false_recovery_rate_all"]),
            to_latex_pct(r["false_recovery_rate_conditional"]),
        ])
    latex_table = (
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "Model & N & CR $\\uparrow$ & GQR-all $\\uparrow$ & GQR-asked $\\uparrow$ & FRR-all $\\downarrow$ & FRR-noQ $\\downarrow$ \\\\\n"
        "\\midrule\n" +
        "\n".join(["{} & {} & {} & {} & {} & {} & {} \\\\".format(*row) for row in latex_rows]) +
        "\n\\bottomrule\n\\end{tabular}\n"
    )
    latex_path = outdir / "metrics_overall.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"[OK] Wrote {latex_path}")

    # Plots
    overall_sorted = metrics_overall.set_index("model")
    save_bar(
        overall_sorted["communication_rate"],
        "Communication Rate by Model",
        "Proportion of items with questions",
        outdir / "fig_comm_rate.png",
    )
    save_bar(
        overall_sorted["good_question_rate_all"],
        "Good Question Rate (All) by Model",
        "Proportion of all items with good questions",
        outdir / "fig_gqr_all.png",
    )
    save_bar(
        overall_sorted["good_question_rate_conditional"],
        "Good Question Rate (Among Asked)",
        "Proportion of asked questions rated 'Good'",
        outdir / "fig_gqr_conditional.png",
    )
    save_bar(
        overall_sorted["false_recovery_rate_all"],
        "False Recovery Rate (All) by Model",
        "Proportion of all items with false recovery",
        outdir / "fig_frr_all.png",
    )
    save_bar(
        overall_sorted["false_recovery_rate_conditional"],
        "False Recovery Rate (Among No-Question)",
        "Proportion of no-question items with false recovery",
        outdir / "fig_frr_conditional.png",
    )
    print(f"[OK] Wrote figures to {outdir}")

    # Per-category pivots
    pivot_cr = metrics_by_cat.pivot(index="category", columns="model", values="communication_rate")
    pivot_gqr_all = metrics_by_cat.pivot(index="category", columns="model", values="good_question_rate_all")
    pivot_frr_all = metrics_by_cat.pivot(index="category", columns="model", values="false_recovery_rate_all")

    pivot_cr.to_csv(outdir / "table_comm_rate_by_category.csv")
    pivot_gqr_all.to_csv(outdir / "table_gqr_all_by_category.csv")
    pivot_frr_all.to_csv(outdir / "table_frr_all_by_category.csv")
    print(f"[OK] Wrote per-category pivot CSVs to {outdir}")


if __name__ == "__main__":
    main()
