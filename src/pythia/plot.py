"""Performance plotting for Pythia model evaluation.

Generates publication-quality figures using seaborn styling.
All axis labels and titles are bold per project conventions.

Usage:
    python plot.py \\
        --task rbp \\
        --predictions /results/ELAVL1/predictions.tsv \\
        --output-dir /results/ELAVL1/figures

    python plot.py \\
        --task ssp \\
        --metrics-json /results/ssp/metrics_best.json \\
        --output-dir /results/ssp/figures

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")

sns.set_theme(style="whitegrid", context="paper")

_FONT_KW = {"fontweight": "bold"}


# ---------------------------------------------------------------------------
# RBP plots
# ---------------------------------------------------------------------------


def plot_rbp_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
    label: str = "PythiaRBP",
) -> Path:
    """Plot ROC curve for RBP binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", **_FONT_KW)
    ax.set_ylabel("True Positive Rate", **_FONT_KW)
    ax.set_title("ROC Curve", **_FONT_KW)
    ax.legend(frameon=True)
    sns.despine()
    fpath = output_dir / "rbp_roc_curve.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fpath


def plot_rbp_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
    label: str = "PythiaRBP",
) -> Path:
    """Plot Precision-Recall curve for RBP binary classification."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, lw=2, label=f"{label} (AUPR={aupr:.3f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", ls="--", lw=1, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", **_FONT_KW)
    ax.set_ylabel("Precision", **_FONT_KW)
    ax.set_title("Precision-Recall Curve", **_FONT_KW)
    ax.legend(frameon=True)
    sns.despine()
    fpath = output_dir / "rbp_pr_curve.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fpath


def plot_rbp_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot binding probability distributions for bound/unbound classes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    df = pd.DataFrame({"score": y_score, "label": y_true})
    for cls, color, lbl in [(0, "steelblue", "Unbound"), (1, "tomato", "Bound")]:
        subset = df[df["label"] == cls]["score"]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_xlabel("Predicted Binding Probability", **_FONT_KW)
    ax.set_ylabel("Density", **_FONT_KW)
    ax.set_title("Score Distribution by Class", **_FONT_KW)
    ax.legend()
    sns.despine()
    fpath = output_dir / "rbp_score_distribution.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fpath


def plot_rbp_from_predictions(pred_csv: Path, output_dir: Path) -> None:
    """Generate all RBP plots from a predictions TSV."""
    df = pd.read_csv(pred_csv, sep="\t")
    if "Response" not in df.columns or "pred_prob_bound" not in df.columns:
        raise ValueError("Predictions CSV must have 'Response' and 'pred_prob_bound' columns.")

    label_map = {"Bound": 1, "Unbound": 0}
    y_true = df["Response"].map(label_map).values
    y_score = df["pred_prob_bound"].values

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rbp_roc(y_true, y_score, output_dir)
    plot_rbp_pr(y_true, y_score, output_dir)
    plot_rbp_score_distribution(y_true, y_score, output_dir)


# ---------------------------------------------------------------------------
# SSP / CMP plots
# ---------------------------------------------------------------------------


def plot_ssp_metrics_from_json(metrics_json: Path, output_dir: Path) -> None:
    """Bar chart of precision, recall, F1 from metrics_best.json."""
    with open(metrics_json) as fh:
        data = json.load(fh)

    splits = {}
    for split in ("val", "test"):
        if split in data and data[split]:
            splits[split] = data[split]

    if not splits:
        print("No val/test metrics found in JSON.")
        return

    split_names, prec_vals, rec_vals, f1_vals = [], [], [], []
    for split, m in splits.items():
        split_names.append(split)
        prec_vals.append(m.get("precision", 0.0))
        rec_vals.append(m.get("recall", 0.0))
        f1_vals.append(m.get("f1", 0.0))

    x = np.arange(len(split_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width, prec_vals, width, label="Precision", color="steelblue")
    ax.bar(x, rec_vals, width, label="Recall", color="seagreen")
    ax.bar(x + width, f1_vals, width, label="F1", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_xlabel("Split", **_FONT_KW)
    ax.set_ylabel("Score", **_FONT_KW)
    ax.set_title("SSP Metrics", **_FONT_KW)
    ax.set_ylim(0, 1)
    ax.legend()
    sns.despine()
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "ssp_metrics.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


def plot_cmp_metrics_from_json(metrics_json: Path, output_dir: Path) -> None:
    """Bar chart of Top-L precision tiers from metrics_best.json."""
    with open(metrics_json) as fh:
        data = json.load(fh)

    splits = {}
    for split in ("val", "test"):
        if split in data and data[split]:
            splits[split] = data[split]

    if not splits:
        print("No val/test metrics found in JSON.")
        return

    keys = ["top_l_precision", "top_l/2_precision", "top_l/5_precision", "top_l/10_precision"]
    labels = ["Top-L", "Top-L/2", "Top-L/5", "Top-L/10"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    width = 0.35

    for idx, (split, m) in enumerate(splits.items()):
        vals = [m.get(k, 0.0) for k in keys]
        offset = (idx - len(splits) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=split.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Metric", **_FONT_KW)
    ax.set_ylabel("Precision", **_FONT_KW)
    ax.set_title("CMP Top-L Precision", **_FONT_KW)
    ax.set_ylim(0, 1)
    ax.legend()
    sns.despine()
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "cmp_topL_precision.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


# ---------------------------------------------------------------------------
# DMP / SSI plots
# ---------------------------------------------------------------------------


def plot_regression_metrics_from_json(
    metrics_json: Path, output_dir: Path, title: str = "Regression Metrics"
) -> None:
    """Bar chart for R² and MSE from metrics_best.json."""
    with open(metrics_json) as fh:
        data = json.load(fh)

    splits = {}
    for split in ("val", "test"):
        if split in data and data[split]:
            splits[split] = data[split]

    if not splits:
        print("No val/test metrics in JSON.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    split_names = list(splits.keys())
    r2_vals = [splits[s].get("r2", 0.0) for s in split_names]
    mse_vals = [splits[s].get("mse", 0.0) for s in split_names]

    axes[0].bar(split_names, r2_vals, color=["steelblue", "tomato"][: len(split_names)])
    axes[0].set_ylabel("R²", **_FONT_KW)
    axes[0].set_xlabel("Split", **_FONT_KW)
    axes[0].set_title(f"{title} — R²", **_FONT_KW)
    axes[0].set_ylim(0, 1)

    axes[1].bar(split_names, mse_vals, color=["steelblue", "tomato"][: len(split_names)])
    axes[1].set_ylabel("MSE", **_FONT_KW)
    axes[1].set_xlabel("Split", **_FONT_KW)
    axes[1].set_title(f"{title} — MSE", **_FONT_KW)

    for ax in axes:
        sns.despine(ax=ax)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f"{title.lower().replace(' ', '_')}_metrics.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


def plot_training_curve(
    log_csv: Path, output_dir: Path, metric: str = "val/loss"
) -> None:
    """Plot a Lightning CSV log training curve."""
    df = pd.read_csv(log_csv)
    if metric not in df.columns:
        available = [c for c in df.columns if "val" in c or "train" in c]
        print(f"Metric '{metric}' not found. Available: {available}")
        if not available:
            return
        metric = available[0]

    df_metric = df[["epoch", metric]].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_metric["epoch"], df_metric[metric], lw=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch", **_FONT_KW)
    ax.set_ylabel(metric.replace("/", " / "), **_FONT_KW)
    ax.set_title(f"Training Curve — {metric}", **_FONT_KW)
    sns.despine()
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = metric.replace("/", "_").replace(" ", "_")
    fpath = output_dir / f"training_curve_{fname}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pythia performance plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["rbp", "ssp", "cmp", "dmp", "ssi"],
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions TSV (RBP task).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Path to metrics_best.json (BEACON tasks).",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional Lightning CSV log for training curve.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "rbp":
        assert args.predictions is not None, "--predictions required for rbp."
        plot_rbp_from_predictions(args.predictions, args.output_dir)
        if args.log_csv is not None:
            plot_training_curve(args.log_csv, args.output_dir)

    elif args.task == "ssp":
        assert args.metrics_json is not None, "--metrics-json required for ssp."
        plot_ssp_metrics_from_json(args.metrics_json, args.output_dir)

    elif args.task == "cmp":
        assert args.metrics_json is not None, "--metrics-json required for cmp."
        plot_cmp_metrics_from_json(args.metrics_json, args.output_dir)

    elif args.task == "dmp":
        assert args.metrics_json is not None, "--metrics-json required for dmp."
        plot_regression_metrics_from_json(
            args.metrics_json, args.output_dir, title="DMP"
        )

    elif args.task == "ssi":
        assert args.metrics_json is not None, "--metrics-json required for ssi."
        plot_regression_metrics_from_json(
            args.metrics_json, args.output_dir, title="SSI"
        )


if __name__ == "__main__":
    main()
