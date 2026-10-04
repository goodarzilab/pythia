"""Inference script for any Pythia task.

Loads a saved checkpoint and runs predictions on an input CSV.

Usage:
    python infer.py \\
        --task rbp \\
        --checkpoint /results/ELAVL1/checkpoints/rbp_best.ckpt \\
        --input-csv /data/ELAVL1_validationSet.tsv.gz \\
        --output-csv /results/ELAVL1/predictions.tsv \\
        --inputsize 256

    python infer.py \\
        --task ssp \\
        --checkpoint /results/ssp/ssp_best.pt \\
        --input-csv /data/bpRNA.csv \\
        --output-csv /results/ssp/predictions.tsv

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pythia.configs import (
    CMPModelConfig,
    DMPModelConfig,
    RBPModelConfig,
    SSIModelConfig,
    SSPModelConfig,
)
from pythia.data.beacon_dataset import (
    BeaconCollator,
    CMPDataset,
    DMPDataset,
    SSICollator,
    SSIDataset,
    SSPDataset,
)
from pythia.data.rbp_dataset import RBPDataset
from pythia.models.cmp import PythiaCMP
from pythia.models.dmp import PythiaDMP
from pythia.models.rbp import PythiaRBP
from pythia.models.ssi import PythiaSSI
from pythia.models.ssp import PythiaSSP


def _load_model_cfg(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    """Attempt to load config from a configs.json sibling."""
    cfg_path = ckpt_path.parent.parent / "configs.json"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            return json.load(fh)
    return None


def load_rbp_model(ckpt_path: Path, args: argparse.Namespace) -> PythiaRBP:
    """Load PythiaRBP weights from checkpoint."""
    saved_cfg = _load_model_cfg(ckpt_path)
    if saved_cfg and "model" in saved_cfg:
        model_cfg = RBPModelConfig(**saved_cfg["model"])
    else:
        model_cfg = RBPModelConfig(
            inputsize=args.inputsize,
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            binarize_fd=args.binarize_fd,
            dp=args.dp,
        )

    model = PythiaRBP(model_cfg)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Lightning .ckpt files store weights under 'state_dict' with a 'model.' prefix
    sd = state.get("state_dict", state)
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("criterion.")}
    model.load_state_dict(sd)
    return model


def load_structural_model(
    task: str, ckpt_path: Path, args: argparse.Namespace
) -> torch.nn.Module:
    """Load a BEACON structural model from a .pt checkpoint."""
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]

    arm1 = tuple(args.arm1_widths)
    arm2 = tuple(args.arm2_widths)

    if task == "ssp":
        cfg = SSPModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
        )
        model: torch.nn.Module = PythiaSSP(cfg)
    elif task == "cmp":
        cfg_cmp = CMPModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
        )
        model = PythiaCMP(cfg_cmp)
    elif task == "dmp":
        cfg_dmp = DMPModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
        )
        model = PythiaDMP(cfg_dmp)
    else:  # ssi
        cfg_ssi = SSIModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
        )
        model = PythiaSSI(cfg_ssi)

    model.load_state_dict(state, strict=False)
    return model


# ---------------------------------------------------------------------------
# Per-task inference runners
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_rbp_inference(
    model: PythiaRBP,
    input_csv: Path,
    output_csv: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    ds = RBPDataset(input_csv, max_len=args.inputsize)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model.eval().to(device)

    all_probs: List[float] = []
    all_preds: List[int] = []

    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

    df = pd.read_csv(input_csv, sep="\t") if str(input_csv).endswith((".tsv", ".tsv.gz")) else pd.read_csv(input_csv)
    df["pred_prob_bound"] = all_probs
    df["pred_label"] = ["Bound" if p == 1 else "Unbound" for p in all_preds]
    df.to_csv(output_csv, sep="\t", index=False)
    print(f"RBP predictions written to {output_csv}")


@torch.no_grad()
def run_ssp_inference(
    model: PythiaSSP,
    input_csv: Path,
    output_csv: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    ds = SSPDataset(input_csv, max_len=args.max_len)
    coll = BeaconCollator(max_len=args.max_len, task="ssp")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=coll)
    model.eval().to(device)

    rows: List[Dict[str, Any]] = []
    for i, batch in enumerate(loader):
        onehot = batch["onehot"].to(device)
        logits = model(onehot)
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        pred_contact = (probs > args.threshold).astype(int)
        rows.append({
            "sample_idx": i,
            "pred_contacts_sum": int(pred_contact.sum()),
            "pred_matrix_shape": str(pred_contact.shape),
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"SSP predictions (summary) written to {output_csv}")


@torch.no_grad()
def run_ssi_inference(
    model: PythiaSSI,
    input_csv: Path,
    output_csv: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    ds = SSIDataset(input_csv, max_len=args.max_len)
    coll = SSICollator(max_len=args.max_len)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=coll
    )
    model.eval().to(device)

    all_preds: List[np.ndarray] = []
    all_lens: List[int] = []

    for batch in loader:
        onehot = batch["onehot"].to(device)
        obs_normed = batch["obs_normed"].to(device)
        obs_known = batch["obs_known"].to(device)
        lengths = batch["lengths"]
        preds = model(onehot, obs_normed, obs_known).cpu().numpy()
        for i, L in enumerate(lengths.tolist()):
            all_preds.append(preds[i, :L])
            all_lens.append(L)

    df_in = pd.read_csv(input_csv)
    df_in["pred_struct"] = [" ".join(f"{v:.4f}" for v in arr) for arr in all_preds]
    df_in.to_csv(output_csv, index=False)
    print(f"SSI predictions written to {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with any Pythia model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True, choices=["rbp", "ssp", "cmp", "dmp", "ssi"]
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5, help="SSP sigmoid threshold.")
    # Model arch args (used if no configs.json found)
    parser.add_argument("--inputsize", type=int, default=256, help="RBP sequence length.")
    parser.add_argument("--dil-start", type=int, default=5)
    parser.add_argument("--dil-end", type=int, default=24)
    parser.add_argument("--bulge-size", type=int, default=2)
    parser.add_argument("--binarize-fd", action="store_true", default=False)
    parser.add_argument("--dp", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-res-layers", type=int, default=8)
    parser.add_argument("--arm1-widths", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--arm2-widths", type=int, nargs=3, default=[256, 128, 128])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.task == "rbp":
        model = load_rbp_model(args.checkpoint, args)
        run_rbp_inference(model, args.input_csv, args.output_csv, args, device)
    elif args.task == "ssp":
        model = load_structural_model("ssp", args.checkpoint, args)
        run_ssp_inference(model, args.input_csv, args.output_csv, args, device)
    elif args.task == "ssi":
        model = load_structural_model("ssi", args.checkpoint, args)
        run_ssi_inference(model, args.input_csv, args.output_csv, args, device)
    else:
        print(f"Inference for task '{args.task}' produces L×L matrices.")
        print("Use task-specific analysis scripts or the plot.py script.")


if __name__ == "__main__":
    main()
