"""Train Pythia BEACON structural tasks: SSP, CMP, DMP, SSI.

Uses PyTorch native training (no Lightning) with cosine + warmup LR schedule
and early stopping.

Example usage:
    python train_beacon.py \\
        --task ssp \\
        --csv /data/bpRNA.csv \\
        --output-dir /results/ssp \\
        --max-epochs 80 --batch-size 1

    python train_beacon.py \\
        --task dmp \\
        --csv /data/dmp_train.csv \\
        --distance-dir /data/distance_maps \\
        --output-dir /results/dmp

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from pythia.configs import (
    CMPModelConfig,
    CMPTrainConfig,
    DMPModelConfig,
    DMPTrainConfig,
    SSIModelConfig,
    SSITrainConfig,
    SSPModelConfig,
    SSPTrainConfig,
)
from pythia.data.beacon_dataset import (
    BeaconCollator,
    CMPDataset,
    CMPNpyDataset,
    DMPDataset,
    DMPNpyDataset,
    SSICollator,
    SSIDataset,
    SSPDataset,
)
from pythia.models.cmp import PythiaCMP, topL_precision
from pythia.models.dmp import PythiaDMP, compute_dmp_metrics
from pythia.models.ssi import PythiaSSI
from pythia.models.ssp import PythiaSSP, compute_ssp_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    warmup_steps = steps_per_epoch * warmup_epochs
    total_steps = steps_per_epoch * total_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# SSP training
# ---------------------------------------------------------------------------


def train_ssp(
    cfg_model: SSPModelConfig,
    cfg_train: SSPTrainConfig,
    csv_path: Path,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train PythiaSSP and return best metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config("ssp", cfg_model, output_dir)

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = SSPDataset(csv_path, split=split, max_len=cfg_train.max_len)
        coll = BeaconCollator(max_len=cfg_train.max_len, task="ssp")
        return DataLoader(
            ds,
            batch_size=cfg_train.batch_size if shuffle else cfg_train.batch_eval,
            shuffle=shuffle,
            num_workers=cfg_train.num_workers,
            collate_fn=coll,
            pin_memory=True,
        )

    tr_loader = make_loader("train", True)
    vl_loader = make_loader("val", False)
    ts_loader = make_loader("test", False)

    model = PythiaSSP(cfg_model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay
    )

    steps_per_epoch = math.ceil(
        len(tr_loader.dataset) / cfg_train.batch_size / cfg_train.grad_accum
    )
    scheduler = build_cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg_train.max_epochs, cfg_train.warmup_epochs
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best: Dict[str, Any] = {"f1": -1.0, "epoch": 0, "val": {}, "test": {}}
    no_improve = 0

    for epoch in range(1, cfg_train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_batches = 0

        for it, batch in enumerate(tr_loader):
            onehot = batch["onehot"].to(device)
            labels = batch["labels"].to(device)
            mask = labels != -1.0

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(onehot)
                # Trim logits to match label dims
                bl = labels.shape[1]
                if logits.shape[1] > bl:
                    logits = logits[:, :bl, :bl]
                elif logits.shape[1] < bl:
                    logits = F.pad(logits, (0, bl - logits.shape[1], 0, bl - logits.shape[1]))
                loss = (
                    criterion(logits[mask], labels[mask])
                    if mask.any()
                    else logits.sum() * 0.0
                )

            scaler.scale(loss / cfg_train.grad_accum).backward()
            n_batches += 1
            epoch_loss += float(loss.detach())

            if (it + 1) % cfg_train.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_m = _eval_ssp(vl_loader, model, device, cfg_train.threshold, use_amp)
        avg_loss = epoch_loss / max(1, n_batches)

        print(
            f"[SSP] epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"| val_f1={val_m['f1']:.4f}"
        )

        if val_m["f1"] > best["f1"]:
            best = {
                "f1": val_m["f1"],
                "epoch": epoch,
                "val": val_m,
                "test": _eval_ssp(ts_loader, model, device, cfg_train.threshold, use_amp),
            }
            torch.save(model.state_dict(), output_dir / "ssp_best.pt")
            _save_json(best, output_dir / "metrics_best.json")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg_train.patience:
                print("Early stopping triggered.")
                break

    return best


@torch.no_grad()
def _eval_ssp(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> Dict[str, float]:
    """Evaluate SSP with global (micro-averaged) precision, recall, and F1."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        onehot = batch["onehot"].to(device)
        labels = batch["labels"].to(device)
        mask = labels != -1.0

        if not mask.any():
            continue

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(onehot)
            bl = labels.shape[1]
            if logits.shape[1] > bl:
                logits = logits[:, :bl, :bl]
            elif logits.shape[1] < bl:
                logits = F.pad(logits, (0, bl - logits.shape[1], 0, bl - logits.shape[1]))

        preds = (torch.sigmoid(logits[mask]) > threshold).cpu().numpy().astype(np.int32)
        lbls = labels[mask].cpu().numpy().astype(np.int32)
        all_preds.append(preds)
        all_labels.append(lbls)

    if not all_preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    all_preds_arr = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels)

    tp = float((all_preds_arr * all_labels_arr).sum())
    fp = float((all_preds_arr * (1 - all_labels_arr)).sum())
    fn = float(((1 - all_preds_arr) * all_labels_arr).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ---------------------------------------------------------------------------
# CMP training
# ---------------------------------------------------------------------------


def train_cmp(
    cfg_model: CMPModelConfig,
    cfg_train: CMPTrainConfig,
    csv_path: Path,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train PythiaCMP and return best metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config("cmp", cfg_model, output_dir)

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = CMPDataset(
            csv_path,
            split=split,
            max_len=cfg_train.max_len,
            min_separation=cfg_model.min_separation,
        )
        coll = BeaconCollator(max_len=cfg_train.max_len, task="cmp")
        return DataLoader(
            ds,
            batch_size=cfg_train.batch_size if shuffle else cfg_train.batch_eval,
            shuffle=shuffle,
            num_workers=cfg_train.num_workers,
            collate_fn=coll,
            pin_memory=True,
        )

    tr_loader = make_loader("train", True)
    vl_loader = make_loader("val", False)
    ts_loader = make_loader("test", False)

    model = PythiaCMP(cfg_model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay
    )

    steps_per_epoch = math.ceil(
        len(tr_loader.dataset) / cfg_train.batch_size / cfg_train.grad_accum
    )
    scheduler = build_cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg_train.max_epochs, cfg_train.warmup_epochs
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best: Dict[str, Any] = {"top_l": -1.0, "epoch": 0, "val": {}, "test": {}}
    no_improve = 0

    for epoch in range(1, cfg_train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for it, batch in enumerate(tr_loader):
            onehot = batch["onehot"].to(device)
            labels = batch["labels"].to(device).float()
            mask = labels != -1.0

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(onehot)
                bl = labels.shape[1]
                if logits.shape[1] > bl:
                    logits = logits[:, :bl, :bl]
                elif logits.shape[1] < bl:
                    logits = F.pad(logits, (0, bl - logits.shape[1], 0, bl - logits.shape[1]))
                loss = (
                    criterion(logits[mask], labels[mask])
                    if mask.any()
                    else logits.sum() * 0.0
                )

            scaler.scale(loss / cfg_train.grad_accum).backward()
            epoch_loss += float(loss.detach())

            if (it + 1) % cfg_train.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_m = _eval_cmp(vl_loader, model, device, cfg_model.min_separation, use_amp)
        print(
            f"[CMP] epoch {epoch:03d} | val_topL={val_m.get('top_l_precision', 0.0):.4f}"
        )

        top_l_val = float(val_m.get("top_l_precision", 0.0))
        if top_l_val > best["top_l"]:
            best = {
                "top_l": top_l_val,
                "epoch": epoch,
                "val": val_m,
                "test": _eval_cmp(ts_loader, model, device, cfg_model.min_separation, use_amp),
            }
            torch.save(model.state_dict(), output_dir / "cmp_best.pt")
            _save_json(best, output_dir / "metrics_best.json")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg_train.patience:
                print("Early stopping triggered.")
                break

    return best


@torch.no_grad()
def _eval_cmp(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    min_separation: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    logits_list, labels_list = [], []

    for batch in loader:
        onehot = batch["onehot"].to(device)
        labels = batch["labels"]
        lengths = batch["lengths"]

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(onehot)

        for i in range(logits.shape[0]):
            L = int(lengths[i].item())
            lo = logits[i, :L, :L].detach().cpu().numpy()
            la = labels[i, :L, :L].numpy()
            valid = la != -1.0
            la_clean = np.where(valid, la, 0.0)
            logits_list.append(lo)
            labels_list.append(la_clean)

    return topL_precision(logits_list, labels_list, min_separation=min_separation)


# ---------------------------------------------------------------------------
# CMP training — BEACON ContactMap format (separate CSVs + .npy files)
# ---------------------------------------------------------------------------


def train_cmp_npy(
    cfg_model: CMPModelConfig,
    cfg_train: CMPTrainConfig,
    train_csv: Path,
    val_csv: Path,
    contact_map_dir: Path,
    output_dir: Path,
    device: torch.device,
    test_csvs: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Train PythiaCMP on BEACON ContactMap format data.

    - Loads pre-computed .npy contact maps (no dot-bracket conversion).
    - Early stopping tracks improvement over the *previous* epoch's val score
      (not the global best).
    - Evaluates and saves metrics for all named test splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config("cmp", cfg_model, output_dir)

    def make_loader(csv: Path, shuffle: bool) -> DataLoader:
        ds = CMPNpyDataset(csv, contact_map_dir, max_len=cfg_train.max_len)
        coll = BeaconCollator(max_len=cfg_train.max_len, task="cmp")
        return DataLoader(
            ds,
            batch_size=cfg_train.batch_size if shuffle else cfg_train.batch_eval,
            shuffle=shuffle,
            num_workers=cfg_train.num_workers,
            collate_fn=coll,
            pin_memory=True,
        )

    tr_loader = make_loader(train_csv, True)
    vl_loader = make_loader(val_csv, False)
    test_loaders: Dict[str, DataLoader] = {}
    if test_csvs:
        for name, csv in test_csvs.items():
            if csv.exists():
                test_loaders[name] = make_loader(csv, False)
            else:
                print(f"[CMP] Warning: test CSV not found, skipping split '{name}': {csv}")

    model = PythiaCMP(cfg_model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay
    )

    steps_per_epoch = math.ceil(
        len(tr_loader.dataset) / cfg_train.batch_size / cfg_train.grad_accum
    )
    scheduler = build_cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg_train.max_epochs, cfg_train.warmup_epochs
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best: Dict[str, Any] = {"top_l": -1.0, "epoch": 0, "val": {}, "test": {}}
    last_val = -1.0
    no_improve = 0

    for epoch in range(1, cfg_train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for it, batch in enumerate(tr_loader):
            onehot = batch["onehot"].to(device)
            labels = batch["labels"].to(device).float()
            mask = labels != -1.0

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(onehot)
                bl = labels.shape[1]
                if logits.shape[1] > bl:
                    logits = logits[:, :bl, :bl]
                elif logits.shape[1] < bl:
                    logits = F.pad(logits, (0, bl - logits.shape[1], 0, bl - logits.shape[1]))
                loss = (
                    criterion(logits[mask], labels[mask])
                    if mask.any()
                    else logits.sum() * 0.0
                )

            scaler.scale(loss / cfg_train.grad_accum).backward()
            epoch_loss += float(loss.detach())

            if (it + 1) % cfg_train.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_m = _eval_cmp(vl_loader, model, device, cfg_model.min_separation, use_amp)
        top_l_val = float(val_m.get("top_l_precision", 0.0))

        print(
            f"[CMP] epoch {epoch:03d} | loss={epoch_loss:.4f} "
            f"| val_topL={top_l_val:.4f} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model (based on all-time best val)
        if top_l_val > best["top_l"]:
            test_m = {
                name: _eval_cmp(loader, model, device, cfg_model.min_separation, use_amp)
                for name, loader in test_loaders.items()
            }
            best = {"top_l": top_l_val, "epoch": epoch, "val": val_m, "test": test_m}
            torch.save(model.state_dict(), output_dir / "cmp_best.pt")
            _save_json(best, output_dir / "metrics_best.json")
            for split_name, split_m in test_m.items():
                print(
                    f"  [{split_name}] Top-L={split_m.get('top_l_precision', 0):.4f} "
                    f"Top-L/2={split_m.get('top_l/2_precision', 0):.4f}"
                )

        # Early stopping: track improvement over *previous* epoch (notebook behaviour)
        if top_l_val > last_val:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg_train.patience:
            print("Early stopping triggered.")
            break

        last_val = top_l_val

    return best


# ---------------------------------------------------------------------------
# DMP training
# ---------------------------------------------------------------------------


def train_dmp(
    cfg_model: DMPModelConfig,
    cfg_train: DMPTrainConfig,
    csv_path: Path,
    distance_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train PythiaDMP and return best metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config("dmp", cfg_model, output_dir)

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = DMPDataset(
            csv_path,
            distance_dir=distance_dir,
            split=split,
            max_len=cfg_train.max_len,
            max_distance=cfg_model.max_distance,
        )
        coll = BeaconCollator(max_len=cfg_train.max_len, task="dmp")
        return DataLoader(
            ds,
            batch_size=cfg_train.batch_size if shuffle else cfg_train.batch_eval,
            shuffle=shuffle,
            num_workers=cfg_train.num_workers,
            collate_fn=coll,
            pin_memory=True,
        )

    tr_loader = make_loader("train", True)
    vl_loader = make_loader("val", False)
    ts_loader = make_loader("test", False)

    model = PythiaDMP(cfg_model).to(device)
    criterion = nn.HuberLoss(delta=cfg_train.huber_delta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay
    )

    steps_per_epoch = math.ceil(
        len(tr_loader.dataset) / cfg_train.batch_size / cfg_train.grad_accum
    )
    scheduler = build_cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg_train.max_epochs, cfg_train.warmup_epochs
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best: Dict[str, Any] = {"r2": -1.0, "epoch": 0, "val": {}, "test": {}}
    no_improve = 0

    for epoch in range(1, cfg_train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for it, batch in enumerate(tr_loader):
            onehot = batch["onehot"].to(device)
            labels = batch["labels"].to(device)
            mask = (labels >= 0).float()
            diag = torch.eye(labels.shape[1], device=device, dtype=torch.bool).unsqueeze(0)
            mask = mask.bool() & (~diag)

            with torch.amp.autocast(device.type, enabled=use_amp):
                preds = model(onehot)
                bl = labels.shape[1]
                if preds.shape[1] > bl:
                    preds = preds[:, :bl, :bl]
                elif preds.shape[1] < bl:
                    preds = F.pad(preds, (0, bl - preds.shape[1], 0, bl - preds.shape[1]))
                preds = preds.clamp(0.0, 1.0)
                loss = (
                    criterion(preds[mask], labels[mask])
                    if mask.any()
                    else preds.sum() * 0.0
                )

            scaler.scale(loss / cfg_train.grad_accum).backward()
            epoch_loss += float(loss.detach())

            if (it + 1) % cfg_train.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_m = _eval_dmp(vl_loader, model, device, use_amp)
        print(f"[DMP] epoch {epoch:03d} | val_r2={val_m['r2']:.4f}")

        if val_m["r2"] > best["r2"]:
            best = {
                "r2": val_m["r2"],
                "epoch": epoch,
                "val": val_m,
                "test": _eval_dmp(ts_loader, model, device, use_amp),
            }
            torch.save(model.state_dict(), output_dir / "dmp_best.pt")
            _save_json(best, output_dir / "metrics_best.json")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg_train.patience and epoch >= cfg_train.min_epochs:
                print("Early stopping triggered.")
                break

    return best


@torch.no_grad()
def _eval_dmp(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Evaluate DMP using the BEACON official metric (global Pearson R²).

    All non-padded positions across all samples are concatenated, then a
    single global Pearson r² is computed.
    """
    model.eval()
    preds_list, labels_list = [], []

    for batch in loader:
        onehot = batch["onehot"].to(device)
        labels = batch["labels"]
        lengths = batch["lengths"]

        with torch.amp.autocast(device.type, enabled=use_amp):
            preds = model(onehot)  # sigmoid output already in [0, 1]

        for i in range(preds.shape[0]):
            L = int(lengths[i].item())
            preds_list.append(preds[i, :L, :L].detach().cpu().numpy())
            labels_list.append(labels[i, :L, :L].numpy())

    return compute_dmp_metrics(preds_list, labels_list)


# ---------------------------------------------------------------------------
# DMP training — BEACON DistanceMap format (separate CSVs + .npy files)
# ---------------------------------------------------------------------------


def train_dmp_npy(
    cfg_model: DMPModelConfig,
    cfg_train: DMPTrainConfig,
    train_csv: Path,
    val_csv: Path,
    distance_map_dir: Path,
    output_dir: Path,
    device: torch.device,
    test_csvs: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Train PythiaDMP on BEACON DistanceMap format data.

    - Loads pre-computed .npy distance maps without Angstrom normalization
      (BEACON .npy files are already in [0, 1]).
    - Uses global Pearson R² for model selection and early stopping.
    - Early stopping tracks improvement over the *previous* epoch's val score
      (not the global best).
    - Evaluates and saves metrics for all named test splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config("dmp", cfg_model, output_dir)

    def make_loader(csv: Path, shuffle: bool) -> DataLoader:
        ds = DMPNpyDataset(csv, distance_map_dir, max_len=cfg_train.max_len)
        coll = BeaconCollator(max_len=cfg_train.max_len, task="dmp")
        return DataLoader(
            ds,
            batch_size=cfg_train.batch_size if shuffle else cfg_train.batch_eval,
            shuffle=shuffle,
            num_workers=cfg_train.num_workers,
            collate_fn=coll,
            pin_memory=True,
        )

    tr_loader = make_loader(train_csv, True)
    vl_loader = make_loader(val_csv, False)
    test_loaders: Dict[str, DataLoader] = {}
    if test_csvs:
        for name, csv in test_csvs.items():
            if csv.exists():
                test_loaders[name] = make_loader(csv, False)
            else:
                print(f"[DMP] Warning: test CSV not found, skipping split '{name}': {csv}")

    model = PythiaDMP(cfg_model).to(device)
    criterion = nn.HuberLoss(delta=cfg_train.huber_delta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay
    )

    steps_per_epoch = math.ceil(
        len(tr_loader.dataset) / cfg_train.batch_size / cfg_train.grad_accum
    )
    scheduler = build_cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg_train.max_epochs, cfg_train.warmup_epochs
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best: Dict[str, Any] = {"r2": -1.0, "epoch": 0, "val": {}, "test": {}}
    last_val = -1.0
    no_improve = 0

    for epoch in range(1, cfg_train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for it, batch in enumerate(tr_loader):
            onehot = batch["onehot"].to(device)
            labels = batch["labels"].to(device).float()
            mask = labels != -1.0

            with torch.amp.autocast(device.type, enabled=use_amp):
                preds = model(onehot)
                bl = labels.shape[1]
                if preds.shape[1] > bl:
                    preds = preds[:, :bl, :bl]
                elif preds.shape[1] < bl:
                    preds = F.pad(preds, (0, bl - preds.shape[1], 0, bl - preds.shape[1]))
                loss = (
                    criterion(preds[mask], labels[mask])
                    if mask.any()
                    else preds.sum() * 0.0
                )

            scaler.scale(loss / cfg_train.grad_accum).backward()
            epoch_loss += float(loss.detach())

            if (it + 1) % cfg_train.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_m = _eval_dmp(vl_loader, model, device, use_amp)
        r2_val = float(val_m.get("r2", 0.0))

        print(
            f"[DMP] epoch {epoch:03d} | loss={epoch_loss:.4f} "
            f"| val_R²={r2_val:.4f} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model (based on all-time best val R²)
        if r2_val > best["r2"]:
            test_m = {
                name: _eval_dmp(loader, model, device, use_amp)
                for name, loader in test_loaders.items()
            }
            best = {"r2": r2_val, "epoch": epoch, "val": val_m, "test": test_m}
            torch.save(model.state_dict(), output_dir / "dmp_best.pt")
            _save_json(best, output_dir / "metrics_best.json")
            for split_name, split_m in test_m.items():
                print(
                    f"  [{split_name}] R²={split_m.get('r2', 0):.4f} "
                    f"MSE={split_m.get('mse', 0):.4f}"
                )

        # Early stopping: track improvement over *previous* epoch (notebook behaviour)
        if r2_val > last_val:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg_train.patience and epoch >= cfg_train.min_epochs:
            print("Early stopping triggered.")
            break

        last_val = r2_val

    return best


# ---------------------------------------------------------------------------
# SSI training
# ---------------------------------------------------------------------------


def train_ssi(
    cfg_model: SSIModelConfig,
    cfg_train: SSITrainConfig,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train PythiaSSI and return best metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config("ssi", cfg_model, output_dir)

    coll = SSICollator(max_len=cfg_train.max_len)

    def make_loader(csv: Path, batch_size: int, shuffle: bool) -> DataLoader:
        ds = SSIDataset(csv, max_len=cfg_train.max_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg_train.num_workers,
            collate_fn=coll,
            pin_memory=True,
        )

    tr_loader = make_loader(train_csv, cfg_train.batch_size, True)
    vl_loader = make_loader(val_csv, cfg_train.batch_eval, False)
    ts_loader = make_loader(test_csv, cfg_train.batch_eval, False)

    model = PythiaSSI(cfg_model).to(device)

    if cfg_train.loss_fn == "huber":
        criterion: nn.Module = nn.HuberLoss(delta=cfg_train.huber_delta)
    elif cfg_train.loss_fn == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # Two-group optimizer: everything except the head at lr_feat, head at lr_head
    head_ids = {id(p) for p in model.head.parameters()}
    feat_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": feat_params, "lr": cfg_train.lr_feat},
            {"params": list(model.head.parameters()), "lr": cfg_train.lr_head},
        ],
        weight_decay=cfg_train.weight_decay,
    )

    steps_per_epoch = math.ceil(
        len(tr_loader.dataset) / cfg_train.batch_size / cfg_train.grad_accum
    )
    scheduler = build_cosine_warmup_scheduler(
        optimizer, steps_per_epoch, cfg_train.max_epochs, cfg_train.warmup_epochs
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best: Dict[str, Any] = {"r2": -1.0, "epoch": 0, "val": {}, "test": {}}
    no_improve = 0

    for epoch in range(1, cfg_train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for it, batch in enumerate(tr_loader):
            onehot = batch["onehot"].to(device)
            lengths = batch["lengths"].to(device)
            y_true = batch["y_true"].to(device)
            m_imp = batch["mask_impute"].to(device)
            obs_normed = batch["obs_normed"].to(device)
            obs_known = batch["obs_known"].to(device)

            L_max = onehot.shape[2]
            rng = torch.arange(L_max, device=device).unsqueeze(0)
            len_mask = rng < lengths.unsqueeze(1)
            mask = m_imp & len_mask

            if not mask.any():
                continue

            with torch.amp.autocast(device.type, enabled=use_amp):
                preds = model(onehot, obs_normed, obs_known)
                loss = criterion(preds[mask], y_true[mask])

            scaler.scale(loss / cfg_train.grad_accum).backward()
            epoch_loss += float(loss.detach())

            if (it + 1) % cfg_train.grad_accum == 0:
                scaler.unscale_(optimizer)
                if cfg_train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_m = _eval_ssi(vl_loader, model, device, use_amp)
        print(f"[SSI] epoch {epoch:03d} | val_r2={val_m['r2']:.4f}")

        if val_m["r2"] > best["r2"]:
            best = {
                "r2": val_m["r2"],
                "epoch": epoch,
                "val": val_m,
                "test": _eval_ssi(ts_loader, model, device, use_amp),
            }
            torch.save(model.state_dict(), output_dir / "ssi_best.pt")
            _save_json(best, output_dir / "metrics_best.json")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg_train.patience:
                print("Early stopping triggered.")
                break

    return best


@torch.no_grad()
def _eval_ssi(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_trues = [], []

    for batch in loader:
        onehot = batch["onehot"].to(device)
        lengths = batch["lengths"].to(device)
        y_true = batch["y_true"].to(device)
        m_imp = batch["mask_impute"].to(device)
        obs_normed = batch["obs_normed"].to(device)
        obs_known = batch["obs_known"].to(device)

        L_max = onehot.shape[2]
        rng = torch.arange(L_max, device=device).unsqueeze(0)
        len_mask = rng < lengths.unsqueeze(1)
        mask = m_imp & len_mask

        if not mask.any():
            continue

        with torch.amp.autocast(device.type, enabled=use_amp):
            preds = model(onehot, obs_normed, obs_known)

        all_preds.append(preds[mask].cpu().numpy())
        all_trues.append(y_true[mask].cpu().numpy())

    if not all_preds:
        return {"r2": 0.0, "mse": 0.0}

    p = np.concatenate(all_preds).astype(np.float64)
    y = np.concatenate(all_trues).astype(np.float64)
    corr = pearsonr(y, p)[0] if len(y) > 1 else 0.0
    r2 = float(corr**2) if np.isfinite(corr) else 0.0
    mse = float(np.mean((y - p) ** 2))
    return {"r2": r2, "mse": mse}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _save_config(task: str, cfg_model: Any, output_dir: Path) -> None:
    """Write config.json with the model architecture hyperparameters.

    Saved once at the start of training so the checkpoint can be reloaded
    without re-specifying every flag::

        with open(output_dir / "config.json") as f:
            cfg = json.load(f)
        model = PythiaCMP(CMPModelConfig(**cfg["model"]))
        model.load_state_dict(torch.load(output_dir / "cmp_best.pt"))
    """
    _save_json({"task": task, "model": cfg_model.model_dump()}, output_dir / "config.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pythia BEACON structural tasks (SSP / CMP / DMP / SSI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["ssp", "cmp", "dmp", "ssi"],
        help="Structural task to train.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to main CSV (SSP/CMP/DMP single-file with split column).",
    )
    # SSI uses separate split files
    parser.add_argument("--train-csv", type=Path, default=None, help="Train CSV (SSI).")
    parser.add_argument("--val-csv", type=Path, default=None, help="Val CSV (SSI).")
    parser.add_argument("--test-csv", type=Path, default=None, help="Test CSV (SSI).")
    # DMP distance maps
    parser.add_argument(
        "--distance-dir",
        type=Path,
        default=None,
        help="Directory with {id}.npy distance maps (DMP only).",
    )
    # CMP npy-format (BEACON ContactMap)
    parser.add_argument(
        "--contact-map-dir",
        type=Path,
        default=None,
        help=(
            "Directory with {id}.npy contact maps (CMP npy mode). "
            "When provided, --train-csv and --val-csv are used instead of --csv."
        ),
    )
    parser.add_argument(
        "--test-splits",
        type=str,
        nargs="+",
        default=["RFAM19", "DIRECT", "test"],
        metavar="SPLIT",
        help=(
            "Names of test splits (CMP npy mode). "
            "Each name resolves to {val-csv-dir}/{name}.csv."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Shared model hyperparams
    parser.add_argument("--dil-start", type=int, default=5)
    parser.add_argument("--dil-end", type=int, default=24)
    parser.add_argument("--bulge-size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-res-layers", type=int, default=8)
    parser.add_argument(
        "--arm1-widths",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        metavar=("W0", "W1", "W2"),
    )
    parser.add_argument(
        "--arm2-widths",
        type=int,
        nargs=3,
        default=[256, 128, 128],
        metavar=("W0", "W1", "W2"),
    )
    parser.add_argument(
        "--binarize-fd", action="store_true", default=False
    )
    parser.add_argument(
        "--dist-enc-dim",
        type=int,
        default=None,
        help="Distance encoding dimension (CMP/SSP/DMP). Defaults to task-specific config value.",
    )

    # Shared training hyperparams
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-eval", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=512)
    # SSI-specific
    parser.add_argument("--lr-feat", type=float, default=3e-4)
    parser.add_argument("--lr-head", type=float, default=1.5e-3)
    # CMP-specific
    parser.add_argument("--min-separation", type=int, default=23)
    # DMP-specific
    parser.add_argument("--max-distance", type=float, default=20.0)
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Early stopping cannot trigger before this many epochs (DMP only).",
    )
    # SSP-specific
    parser.add_argument("--threshold", type=float, default=0.5)
    # SSI-specific loss & clipping
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="mse",
        choices=["mse", "huber", "l1"],
        help="Loss function for SSI (mse / huber / l1).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping in SSI (0 disables clipping).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    arm1 = tuple(args.arm1_widths)
    arm2 = tuple(args.arm2_widths)

    if args.task == "ssp":
        assert args.csv is not None, "--csv required for ssp."
        # dist_enc_dim falls back to SSPModelConfig default (64) if not specified
        dist_enc_dim_ssp = (
            args.dist_enc_dim if args.dist_enc_dim is not None else SSPModelConfig().dist_enc_dim
        )
        cfg_model = SSPModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
            binarize_fd=args.binarize_fd,
            dist_enc_dim=dist_enc_dim_ssp,
        )
        cfg_train = SSPTrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            batch_eval=args.batch_eval,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            max_len=args.max_len,
            threshold=args.threshold,
        )
        result = train_ssp(cfg_model, cfg_train, args.csv, args.output_dir, device)

    elif args.task == "cmp":
        # dist_enc_dim falls back to CMPModelConfig default (64) if not specified
        dist_enc_dim_cmp = args.dist_enc_dim if args.dist_enc_dim is not None else CMPModelConfig().dist_enc_dim
        cfg_model = CMPModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
            binarize_fd=args.binarize_fd,
            min_separation=args.min_separation,
            dist_enc_dim=dist_enc_dim_cmp,
        )
        cfg_train = CMPTrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            batch_eval=args.batch_eval,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            max_len=args.max_len,
        )

        if args.contact_map_dir is not None:
            # BEACON ContactMap format: separate CSVs + .npy files
            assert args.train_csv is not None, "--train-csv required with --contact-map-dir."
            assert args.val_csv is not None, "--val-csv required with --contact-map-dir."
            val_dir = args.val_csv.parent
            test_csvs = {name: val_dir / f"{name}.csv" for name in args.test_splits}
            result = train_cmp_npy(
                cfg_model,
                cfg_train,
                args.train_csv,
                args.val_csv,
                args.contact_map_dir,
                args.output_dir,
                device,
                test_csvs=test_csvs,
            )
        else:
            # Legacy bpRNA format: single CSV with split column
            assert args.csv is not None, "--csv required for cmp (or use --contact-map-dir for npy mode)."
            result = train_cmp(cfg_model, cfg_train, args.csv, args.output_dir, device)

    elif args.task == "dmp":
        assert args.distance_dir is not None, "--distance-dir required for dmp."
        # dist_enc_dim falls back to DMPModelConfig default (64) if not specified
        dist_enc_dim_dmp = args.dist_enc_dim if args.dist_enc_dim is not None else DMPModelConfig().dist_enc_dim
        cfg_model = DMPModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
            binarize_fd=args.binarize_fd,
            max_distance=args.max_distance,
            dist_enc_dim=dist_enc_dim_dmp,
        )
        cfg_train = DMPTrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            batch_eval=args.batch_eval,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            max_len=args.max_len,
            huber_delta=args.huber_delta,
            min_epochs=args.min_epochs,
        )

        if args.train_csv is not None:
            # BEACON DistanceMap format: separate CSVs + .npy files
            assert args.val_csv is not None, "--val-csv required with --train-csv for dmp."
            val_dir = args.val_csv.parent
            test_csvs = {name: val_dir / f"{name}.csv" for name in args.test_splits}
            result = train_dmp_npy(
                cfg_model,
                cfg_train,
                args.train_csv,
                args.val_csv,
                args.distance_dir,
                args.output_dir,
                device,
                test_csvs=test_csvs,
            )
        else:
            # Legacy bpRNA format: single CSV with split column
            assert args.csv is not None, "--csv required for dmp (or use --train-csv/--val-csv for npy mode)."
            result = train_dmp(
                cfg_model,
                cfg_train,
                args.csv,
                args.distance_dir,
                args.output_dir,
                device,
            )

    else:  # ssi
        assert args.train_csv is not None, "--train-csv required for ssi."
        assert args.val_csv is not None, "--val-csv required for ssi."
        assert args.test_csv is not None, "--test-csv required for ssi."
        cfg_model = SSIModelConfig(
            dil_start=args.dil_start,
            dil_end=args.dil_end,
            bulge_size=args.bulge_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            num_res_layers=args.num_res_layers,
            arm1_widths=arm1,
            arm2_widths=arm2,
            binarize_fd=args.binarize_fd,
        )
        cfg_train = SSITrainConfig(
            lr_feat=args.lr_feat,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            batch_eval=args.batch_eval,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            max_len=min(args.max_len, 440),
            loss_fn=args.loss_fn,
            huber_delta=args.huber_delta,
            grad_clip=args.grad_clip,
        )
        result = train_ssi(
            cfg_model,
            cfg_train,
            args.train_csv,
            args.val_csv,
            args.test_csv,
            args.output_dir,
            device,
        )

    print("\nTraining complete. Summary:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
