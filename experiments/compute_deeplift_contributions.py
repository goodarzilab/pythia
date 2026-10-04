"""Quantify sequence vs. structure contribution to RBP binding using DeepLIFT.

For each trained PythiaRBP model, this script:
  1. Loads the validation set (bound peaks only, up to 10k).
  2. Applies captum DeepLIFT on the merged embedding layer of PythiaRBP,
     using three conditions:
       - Full model              -> "dl" attribution
       - Sequence arm randomised -> "seq" attribution (structure contribution)
       - Structure arm randomised-> "structure" attribution (sequence contribution)
     The "randomised" arm is replaced with an all-zero (no-nucleotide) input,
     the null baseline used for the published results.
  3. Computes the ratio log2(seq / structure) per peak, indicating whether
     the structure (FD conv) arm or the sequence (1D conv) arm drives binding.
  4. Writes per-RBP peak-ratio TSVs to
       {data_dir}/{rbp}/{model_version}/{rbp}_deeplift_peak_ratio.tsv.gz
  5. After all RBPs are processed, aggregates and plots summary figures
     to --outdir.

Checkpointing: RBPs with an existing peak-ratio TSV are skipped.
Memory: intermediate arrays are deleted immediately after each mini-batch.

Usage
-----
    python compute_deeplift_contributions.py \\
        --data-dir output/rbp_models \\
        --splits-dir output/dataSplits \\
        --outdir output/deeplift_contributions

See --help for full argument list.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import DeepLift

from pythia.configs import RBPModelConfig
from pythia.models.rbp import PythiaRBP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INPUTSIZE: int = 256
_NUC_MAP: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(ckpt_path: Path) -> PythiaRBP:
    """Load PythiaRBP from a PyTorch-Lightning checkpoint.

    Parameters
    ----------
    ckpt_path:
        Path to rbp_best.ckpt.

    Returns
    -------
    PythiaRBP
        Loaded model in eval mode on CPU.
    """
    cfg_path = ckpt_path.parent.parent / "configs.json"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            saved = json.load(fh)
        model_cfg = RBPModelConfig(**saved["model"])
    else:
        model_cfg = RBPModelConfig()

    model = PythiaRBP(model_cfg)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    sd = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in sd.items()
    }
    model_keys = set(model.state_dict().keys())
    unexpected = [k for k in sd if k not in model_keys]
    if unexpected:
        log.warning("Ignoring unexpected checkpoint keys: %s", unexpected)
        sd = {k: v for k, v in sd.items() if k in model_keys}
    model.load_state_dict(sd)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------


def one_hot_encode(seq: str, length: int = INPUTSIZE) -> np.ndarray:
    """One-hot encode an RNA/DNA sequence to shape (4, length).

    Parameters
    ----------
    seq:
        RNA/DNA sequence string (T treated as U).
    length:
        Output length; truncated or zero-padded as needed.

    Returns
    -------
    np.ndarray
        Float32 array of shape (4, length).
    """
    arr = np.zeros((4, length), dtype=np.float32)
    for i, ch in enumerate(seq.upper()[:length]):
        idx = _NUC_MAP.get(ch)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def sequences_to_tensor(
    sequences: np.ndarray,
    device: torch.device,
    length: int = INPUTSIZE,
) -> torch.Tensor:
    """Convert an array of sequence strings to a one-hot tensor.

    Parameters
    ----------
    sequences:
        1-D numpy array of RNA sequence strings.
    device:
        Target device.
    length:
        Sequence length for one-hot encoding.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape (N, 4, length).
    """
    tensors = [torch.from_numpy(one_hot_encode(s, length)) for s in sequences]
    return torch.stack(tensors).to(device)


# ---------------------------------------------------------------------------
# Architecture helpers for PythiaRBP
# ---------------------------------------------------------------------------


class _InnerModel(nn.Module):
    """Classification head of PythiaRBP, operating on the merged feature vector.

    Accepts the pre-computed flattened concatenation of arm1 and arm2
    outputs and produces class logits.

    Parameters
    ----------
    dense:
        The nn.Sequential dense block from a loaded PythiaRBP instance.
    """

    def __init__(self, dense: nn.Sequential) -> None:
        super().__init__()
        self.dense = dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classification head only.

        Parameters
        ----------
        x:
            Flattened merged arm features, shape (B, feature_dim).

        Returns
        -------
        torch.Tensor
            Logits, shape (B, 2).
        """
        return self.dense(x)


def _structure_arm_output(model: PythiaRBP, x: torch.Tensor) -> torch.Tensor:
    """Compute the output of the fixed-dilated (structure) arm.

    Replicates the FD-conv branch of PythiaRBP._forward_features.

    Parameters
    ----------
    model:
        PythiaRBP instance.
    x:
        One-hot input, shape (B, 4, L).

    Returns
    -------
    torch.Tensor
        Arm1 output, shape (B, C, L').
    """
    pad = x.shape[2]
    x_pad = F.pad(x, (pad, pad))
    fd_out = model.conv_fd(x_pad)
    fd_pooled = model.pool_fd(fd_out)
    fd_pooled = fd_pooled.transpose(1, 2)
    fd_processed = model.bn_fd(fd_pooled)
    return model.arm1(fd_processed)


def _sequence_arm_output(model: PythiaRBP, x: torch.Tensor) -> torch.Tensor:
    """Compute the output of the 1-D sequence (arm2) arm.

    Parameters
    ----------
    model:
        PythiaRBP instance.
    x:
        One-hot input, shape (B, 4, L).

    Returns
    -------
    torch.Tensor
        Arm2 output, shape (B, C, L'').
    """
    return model.arm2(x)


def _merge_and_flatten(arm1_out: torch.Tensor, arm2_out: torch.Tensor) -> torch.Tensor:
    """Concatenate arm outputs along the last dim and flatten.

    Parameters
    ----------
    arm1_out:
        Structure arm output, shape (B, C1, L1).
    arm2_out:
        Sequence arm output, shape (B, C2, L2).

    Returns
    -------
    torch.Tensor
        Flattened tensor, shape (B, C1*L1 + C2*L2).
    """
    merged = torch.cat([arm1_out, arm2_out], dim=2)
    return merged.view(merged.shape[0], -1)


def _zero_sequence(x: torch.Tensor) -> torch.Tensor:
    """Return an all-zero tensor of the same shape as x (null/no-nucleotide baseline).

    Parameters
    ----------
    x:
        One-hot tensor, shape (B, 4, L).

    Returns
    -------
    torch.Tensor
        Zero tensor with the same shape and device as x.
    """
    return torch.zeros_like(x)


def _write_yaml(path: Path, data: Dict) -> None:
    """Write a flat dict as a YAML file."""
    with open(path, "w") as fh:
        for k, v in data.items():
            if isinstance(v, str):
                fh.write(f"{k}: \"{v}\"\n")
            else:
                fh.write(f"{k}: {v}\n")


# ---------------------------------------------------------------------------
# DeepLIFT attribution computation
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: PythiaRBP,
    x: torch.Tensor,
    inner_model: _InnerModel,
) -> pd.DataFrame:
    """Compute DeepLIFT attribution scores separating structure vs. sequence.

    Each arm is in turn replaced with the all-zero (no-nucleotide) baseline:
      - "dl"       : DeepLIFT on the full merged embedding.
      - "seq"      : DeepLIFT with the structure (FD conv) arm zeroed out
                     (isolates the structure-driven signal).
      - "structure": DeepLIFT with the sequence arm zeroed out
                     (isolates the sequence-driven signal).

    Parameters
    ----------
    model:
        PythiaRBP in eval mode.
    x:
        One-hot tensor for bound peaks, shape (B, 4, L).
    inner_model:
        _InnerModel wrapping the dense classification head.

    Returns
    -------
    pd.DataFrame
        Per-peak attribution DataFrame with columns:
        dl, seq, structure, ratio_seq_structure, log_ratio.
    """
    dl_inner = DeepLift(inner_model)
    baseline = torch.zeros(
        (x.shape[0], inner_model.dense[0].in_features), device=x.device
    )

    with torch.no_grad():
        arm1_out = _structure_arm_output(model, x)
        arm2_out = _sequence_arm_output(model, x)
        out_full = _merge_and_flatten(arm1_out, arm2_out)

        x_zero = _zero_sequence(x)
        arm1_zero = _structure_arm_output(model, x_zero)
        arm2_zero = _sequence_arm_output(model, x_zero)

    out_seq_rand = _merge_and_flatten(arm1_out, arm2_zero)
    out_str_rand = _merge_and_flatten(arm1_zero, arm2_out)

    result: Dict[str, np.ndarray] = {}
    for key, out in [
        ("dl", out_full),
        ("seq", out_seq_rand),
        ("structure", out_str_rand),
    ]:
        attr = dl_inner.attribute(out, baselines=baseline, target=1)
        result[key] = torch.sum(torch.abs(attr), dim=1).detach().cpu().numpy()

    df = pd.DataFrame(result)
    df["ratio_seq_structure"] = df["seq"] / (df["structure"] + 1e-9)
    df["log_ratio"] = np.log2(df["ratio_seq_structure"] + 1e-3)
    return df


# ---------------------------------------------------------------------------
# Per-RBP processing
# ---------------------------------------------------------------------------


def process_rbp(
    rbp: str,
    val_tsv: Path,
    ckpt_path: Path,
    outdir: Path,
    device: torch.device,
    minibatch: int = 32,
    max_peaks: int = 10_000,
    model_version: str = "Pythia",
) -> Optional[pd.DataFrame]:
    """Run DeepLIFT structure/sequence attribution for one RBP.

    Skips if the output file already exists (checkpointing).

    Parameters
    ----------
    rbp:
        RBP name.
    val_tsv:
        Path to the validation TSV.gz file with columns Input, Response, SeqNames.
    ckpt_path:
        Checkpoint path (.ckpt).
    outdir:
        Output directory for this RBP's model version folder.
    device:
        Torch device.
    minibatch:
        Mini-batch size for DeepLIFT (keep small to avoid OOM).
    max_peaks:
        Maximum number of bound peaks to analyse (random subsample).
    model_version:
        Model version string used in YAML metadata, e.g. "Pythia".

    Returns
    -------
    Optional[pd.DataFrame]
        Peak-ratio DataFrame, or None on failure.
    """
    outpath = outdir / f"{rbp}_deeplift_peak_ratio.tsv.gz"
    yaml_path = outdir / f"{rbp}_deeplift_peak_ratio.yaml"
    if outpath.exists():
        log.info("[%s] Output already exists — skipping: %s", rbp, outpath)
        return pd.read_csv(outpath, sep="\t", compression="gzip")

    if not val_tsv.exists():
        log.warning("[%s] Validation TSV not found: %s", rbp, val_tsv)
        return None

    try:
        model = load_model(ckpt_path)
    except Exception as exc:
        log.error("[%s] Failed to load model: %s", rbp, exc)
        return None

    model.eval().to(device)
    inner_model = _InnerModel(model.dense).to(device)
    inner_model.eval()

    # Load validation set and filter to bound peaks
    val_df = pd.read_csv(val_tsv, sep="\t")
    bound_mask = val_df["Response"].astype(str).isin(["Bound", "1"])
    bound_df = val_df[bound_mask].reset_index(drop=True)

    if bound_df.empty:
        log.warning("[%s] No bound peaks in validation set.", rbp)
        return None

    if len(bound_df) > max_peaks:
        np.random.seed(42)
        bound_df = bound_df.sample(n=max_peaks, random_state=42).reset_index(drop=True)
        log.info("[%s] Subsampled to %d bound peaks.", rbp, max_peaks)

    sequences = bound_df["Input"].to_numpy(dtype=str)
    seq_names = (
        bound_df["SeqNames"].to_numpy(dtype=str)
        if "SeqNames" in bound_df.columns
        else np.arange(len(bound_df)).astype(str)
    )

    list_peak_dfs: List[pd.DataFrame] = []
    n_total = len(sequences)
    for start in range(0, n_total, minibatch):
        end = min(start + minibatch, n_total)
        log.info("[%s] DeepLIFT mini-batch %d-%d / %d", rbp, start, end, n_total)

        x = sequences_to_tensor(sequences[start:end], device)
        with torch.set_grad_enabled(True):
            batch_df = compute_shap_values(model, x, inner_model)

        batch_df["Sequence.Name"] = seq_names[start:end]
        list_peak_dfs.append(batch_df)

        del x, batch_df
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not list_peak_dfs:
        log.warning("[%s] No DeepLIFT results produced.", rbp)
        return None

    peak_df = pd.concat(list_peak_dfs, ignore_index=True)
    peak_df["RBP"] = rbp
    peak_df.to_csv(outpath, sep="\t", index=False, compression="gzip")
    log.info("[%s] Saved %d rows -> %s", rbp, len(peak_df), outpath)

    _write_yaml(yaml_path, {
        "rbp": rbp,
        "baseline": "zero (all-zero, no-nucleotide arm replacement)",
        "minibatch": minibatch,
        "max_peaks": max_peaks,
        "n_peaks_analysed": len(peak_df),
        "model_version": model_version,
        "inputsize": INPUTSIZE,
        "ckpt_path": str(ckpt_path),
    })
    log.info("[%s] Saved YAML -> %s", rbp, yaml_path)
    return peak_df


# ---------------------------------------------------------------------------
# Plotting (equivalent to 01_plot_contribution_structure.R)
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, outdir: Path, stem: str) -> None:
    """Save a figure as both PNG (200 dpi) and PDF."""
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fpath = outdir / f"{stem}.{ext}"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        log.info("Saved: %s", fpath)
    plt.close(fig)


def plot_contributions(
    stat_df: pd.DataFrame, outdir: Path, model_version: str = "Pythia"
) -> None:
    """Produce summary boxplots of structure vs. sequence contributions.

    Replicates the three plots from 01_plot_contribution_structure.R:
      1. Boxplot of log2(seq/structure) per RBP (median-ordered, magma fill).
      2. Boxplot of seq.minus.dl (sequence-specific attribution minus full).
      3. TSV of per-RBP median log-ratio with Wilcoxon FDR.

    Parameters
    ----------
    stat_df:
        Combined per-peak DataFrame with log_ratio, seq, structure, dl, RBP.
    outdir:
        Directory for PNG and PDF outputs.
    model_version:
        Model version string used in output file names.
    """
    # Compute per-RBP median log_ratio for ordering and colouring
    mediandf = (
        stat_df.groupby("RBP")
        .agg(
            Median_LR=("log_ratio", "median"),
            p_value=("log_ratio", lambda v: _wilcox_pvalue(v)),
        )
        .reset_index()
    )
    mediandf["FDR"] = _fdr_bh(mediandf["p_value"].values)
    mediandf = mediandf.sort_values("Median_LR", ascending=False)
    order_rbps = mediandf["RBP"].tolist()

    # Save summary TSV
    mediandf.to_csv(outdir / f"{model_version}_structure_seq_medians.tsv", sep="\t", index=False)

    stat_df = stat_df.copy()
    stat_df["RBP"] = pd.Categorical(stat_df["RBP"], categories=order_rbps, ordered=True)
    stat_df = stat_df.merge(mediandf[["RBP", "Median_LR"]], on="RBP", how="left")

    # Colour per RBP by Median_LR (viridis magma palette)
    try:
        import matplotlib.colors as mcolors
        import matplotlib.colormaps
        norm = mcolors.Normalize(
            vmin=mediandf["Median_LR"].min(),
            vmax=mediandf["Median_LR"].max(),
        )
        cmap = matplotlib.colormaps["magma"]
        fill_colors = {
            rbp: cmap(norm(stat_df.loc[stat_df["RBP"] == rbp, "Median_LR"].iloc[0]))
            for rbp in order_rbps
            if rbp in stat_df["RBP"].values
        }
    except Exception:
        fill_colors = None

    # --- Boxplot: log_ratio per RBP (magma coloured by median) ---
    fig, ax = plt.subplots(figsize=(22, 8))
    if fill_colors:
        for i, rbp in enumerate(order_rbps):
            grp = stat_df[stat_df["RBP"] == rbp]["log_ratio"]
            if grp.empty:
                continue
            bp = ax.boxplot(
                grp.dropna().values,
                positions=[i],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(fill_colors.get(rbp, (0.5, 0.5, 0.5, 1)))
        ax.set_xticks(range(len(order_rbps)))
        ax.set_xticklabels(order_rbps, rotation=45, ha="right")
    else:
        sns.boxplot(
            data=stat_df, x="RBP", y="log_ratio",
            order=order_rbps, ax=ax, showfliers=False,
        )
        ax.tick_params(axis="x", rotation=45)

    ax.set_xlabel("", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "log\u2082 (DeepLIFT randomised sequence / DeepLIFT randomised structure)",
        fontsize=12, fontweight="bold",
    )
    ax.set_title(
        "Structure vs. sequence contribution to RBP binding",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    _save_fig(fig, outdir, f"{model_version}_structure_contribution_boxplot_medianOrdered_magma")

    # --- Compute seq.minus.dl ---
    stat_df["seq.minus.dl"] = stat_df["seq"] - stat_df["dl"]
    mediandf_seq = (
        stat_df.groupby("RBP")["seq.minus.dl"]
        .median()
        .reset_index()
        .rename(columns={"seq.minus.dl": "Median_seq_minus_dl"})
        .sort_values("Median_seq_minus_dl", ascending=False)
    )
    seq_order = mediandf_seq["RBP"].tolist()
    stat_df["RBP"] = pd.Categorical(stat_df["RBP"], categories=seq_order, ordered=True)

    fig, ax = plt.subplots(figsize=(22, 8))
    sns.boxplot(
        data=stat_df, x="RBP", y="seq.minus.dl",
        hue="RBP", order=seq_order, ax=ax, showfliers=False,
        palette="magma", legend=False,
    )
    ax.set_xlabel("", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "DeepLIFT (randomised sequence) \u2212 full network",
        fontsize=12, fontweight="bold",
    )
    ax.set_title("Sequence arm contribution", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    _save_fig(fig, outdir, f"{model_version}_structure_contribution_boxplot_seq_minus_dl")


def _wilcox_pvalue(values: pd.Series) -> float:
    """One-sample Wilcoxon signed-rank test against zero.

    Parameters
    ----------
    values:
        Series of log-ratio values.

    Returns
    -------
    float
        Two-sided p-value.
    """
    from scipy.stats import wilcoxon

    v = values.dropna().values
    if len(v) < 3:
        return 1.0
    try:
        _, p = wilcoxon(v)
    except Exception:
        return 1.0
    return float(p)


def _fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues:
        Array of p-values.

    Returns
    -------
    np.ndarray
        FDR-adjusted p-values.
    """
    n = len(pvalues)
    if n == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = np.argsort(order) + 1
    fdr = np.minimum(1.0, pvalues * n / ranked)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        fdr[order[i]] = min(fdr[order[i]], fdr[order[i + 1]])
    return fdr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify sequence vs. structure contribution to RBP binding "
            "using DeepLIFT on PythiaRBP checkpoints, with an all-zero "
            "(no-nucleotide) baseline for the randomised arm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing per-RBP sub-directories with "
            "{model-version}/checkpoints/rbp_best.ckpt (see train_rbp.py)."
        ),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Root directory of per-RBP data splits (for validation TSV files).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for aggregated TSVs and summary figures.",
    )
    parser.add_argument(
        "--rbp",
        type=str,
        default=None,
        help="Process only a single named RBP (default: all with checkpoints).",
    )
    parser.add_argument(
        "--minibatch",
        type=int,
        default=32,
        help="Mini-batch size for DeepLIFT (lower = less memory).",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=10_000,
        help="Max number of bound peaks per RBP (random subsample if exceeded).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="Pythia",
        help="Model version sub-directory name under --data-dir/{rbp}/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    model_version = args.model_version

    if args.rbp:
        rbps = [args.rbp]
    else:
        rbps = sorted(
            d.name
            for d in args.data_dir.iterdir()
            if d.is_dir()
            and (d / model_version / "checkpoints" / "rbp_best.ckpt").exists()
        )
    log.info("RBPs to process: %d", len(rbps))

    all_peak_dfs: List[pd.DataFrame] = []
    for rbp in rbps:
        ckpt_path = args.data_dir / rbp / model_version / "checkpoints" / "rbp_best.ckpt"
        val_tsv = args.splits_dir / rbp / f"{rbp}_validationSet.tsv.gz"
        model_dir = args.data_dir / rbp / model_version

        if not ckpt_path.exists():
            log.warning("[%s] Checkpoint not found — skipping.", rbp)
            continue

        peak_df = process_rbp(
            rbp=rbp,
            val_tsv=val_tsv,
            ckpt_path=ckpt_path,
            outdir=model_dir,
            device=device,
            minibatch=args.minibatch,
            max_peaks=args.max_peaks,
            model_version=model_version,
        )
        if peak_df is not None and not peak_df.empty:
            all_peak_dfs.append(peak_df)

    if not all_peak_dfs:
        log.error("No DeepLIFT results produced. Check checkpoint and validation paths.")
        return

    combined_df = pd.concat(all_peak_dfs, ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    combined_tsv = args.outdir / f"allRbps_deeplift_peak_ratio_{model_version}.tsv.gz"
    combined_df.to_csv(combined_tsv, sep="\t", index=False, compression="gzip")
    log.info(
        "Combined DeepLIFT results: %d rows -> %s", len(combined_df), combined_tsv
    )

    plot_contributions(combined_df, args.outdir, model_version=model_version)
    log.info("Done. Figures written to %s", args.outdir)


if __name__ == "__main__":
    main()
