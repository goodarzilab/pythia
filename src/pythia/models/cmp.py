"""PythiaCMP: long-range contact map prediction model.

Architecture (faithful to the BEACON contact-map benchmark reference model):
    One-hot → FD (frozen) → arm1 [K16, K10, K5] (tapered widths) ──┐
    One-hot → arm2 [K12, K7,  K5] (tapered widths) ─────────────────┼→ concat
                                                                     ↓
        Pairwise: outer_prod + outer_sum + outer_diff + dist_enc (learned bins)
                                                                     ↓
                    pair_proj (Conv2d + BN + ReLU) → symmetrize
                                                                     ↓
                2D ResNet (BatchNorm + ReLU, multi-scale dilations)
                                                                     ↓
                            2-layer output head → symmetrize → (B, L, L)

    Only positions with |i-j| >= min_separation (default 23) are evaluated
    by the metric; training uses all non-padded positions.

    Loss: BCEWithLogitsLoss
    Metric: BEACON official Top-L precision (micro-averaged, 0.5 threshold)

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.configs import CMPModelConfig
from pythia.models.common import (
    PositionalEncoding2DLearned,
    ResBlock2DBN,
    symmetrize,
)
from pythia.models.fixed_dilated_conv import FixedDilatedConv

# Multi-scale dilation pattern (cycles through [1,2,4,8], truncated to num_res_layers)
_DILATION_PATTERN: Tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8, 1, 1)


class PythiaCMP(nn.Module):
    """Contact map prediction model.

    The 1-D backbone arms feed directly into pairwise features without an
    intermediate hidden-dim projection.

    Parameters
    ----------
    cfg:
        CMPModelConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: Optional[CMPModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = CMPModelConfig()
        self.cfg = cfg

        # ------------------------------------------------------------------ #
        # Fixed dilated conv (Pythia core — always frozen)
        # ------------------------------------------------------------------ #
        self.fd_conv = FixedDilatedConv(
            in_channel=4,
            dil_start=cfg.dil_start,
            dil_end=cfg.dil_end,
            bulge_size=cfg.bulge_size,
            trainable=False,
            binarize_fd=cfg.binarize_fd,
        )

        # ------------------------------------------------------------------ #
        # Arm 1: FD output → [K16, K10, K5]
        # ------------------------------------------------------------------ #
        self.arm1 = nn.Sequential(
            nn.Conv1d(self.fd_conv.out_channel, cfg.arm1_widths[0], kernel_size=16, padding=8),
            nn.BatchNorm1d(cfg.arm1_widths[0]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.arm1_widths[0], cfg.arm1_widths[1], kernel_size=10, padding=5),
            nn.BatchNorm1d(cfg.arm1_widths[1]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.arm1_widths[1], cfg.arm1_widths[2], kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.arm1_widths[2]),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------ #
        # Arm 2: one-hot → [K12, K7, K5]
        # ------------------------------------------------------------------ #
        self.arm2 = nn.Sequential(
            nn.Conv1d(4, cfg.arm2_widths[0], kernel_size=12, padding=6),
            nn.BatchNorm1d(cfg.arm2_widths[0]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.arm2_widths[0], cfg.arm2_widths[1], kernel_size=7, padding=3),
            nn.BatchNorm1d(cfg.arm2_widths[1]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.arm2_widths[1], cfg.arm2_widths[2], kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.arm2_widths[2]),
            nn.ReLU(),
        )

        total_1d = cfg.arm1_widths[2] + cfg.arm2_widths[2]

        # Optional BatchNorm on concatenated arm features before pairwise building
        self.transition_norm: nn.Module = (
            nn.BatchNorm1d(total_1d) if cfg.use_transition_norm else nn.Identity()
        )

        # ------------------------------------------------------------------ #
        # Distance encoding (learned bins, not sinusoidal)
        # ------------------------------------------------------------------ #
        self.dist_enc = PositionalEncoding2DLearned(cfg.dist_enc_dim)

        # ------------------------------------------------------------------ #
        # Pairwise projection: (total_1d*3 + dist_enc_dim) → hidden_dim
        # ------------------------------------------------------------------ #
        pairwise_in = total_1d * 3 + cfg.dist_enc_dim
        self.pair_proj = nn.Sequential(
            nn.Conv2d(pairwise_in, cfg.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------ #
        # 2-D ResNet with multi-scale dilations (BN + ReLU blocks)
        # ------------------------------------------------------------------ #
        self.res_blocks = nn.ModuleList([
            ResBlock2DBN(
                cfg.hidden_dim,
                dropout=cfg.dropout,
                dilation=_DILATION_PATTERN[i] if i < len(_DILATION_PATTERN) else 1,
            )
            for i in range(cfg.num_res_layers)
        ])

        # ------------------------------------------------------------------ #
        # Two-layer output head
        # ------------------------------------------------------------------ #
        self.head = nn.Sequential(
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(cfg.hidden_dim // 2, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _align(x: torch.Tensor, target: int) -> torch.Tensor:
        """Center-crop or center-pad the last dimension to `target` length."""
        curr = x.shape[-1]
        if curr > target:
            start = (curr - target) // 2
            return x[..., start: start + target]
        if curr < target:
            diff = target - curr
            return F.pad(x, (diff // 2, diff - diff // 2))
        return x

    def _run_fd(self, onehot: torch.Tensor) -> torch.Tensor:
        L = onehot.shape[2]
        pad = self.fd_conv.kernel_size // 2
        fd = self.fd_conv(F.pad(onehot, (pad, pad)))
        return self._align(fd, L)

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        onehot : (B, 4, L)

        Returns
        -------
        logits : (B, L, L)
        """
        B, _, L = onehot.shape

        # 1-D features from both arms
        h2 = self._align(self.arm2(onehot), L)           # (B, arm2[-1], L)
        h1 = self._align(self.arm1(self._run_fd(onehot)), L)  # (B, arm1[-1], L)
        h = self.transition_norm(torch.cat([h1, h2], dim=1))  # (B, total_1d, L)

        # Rich pairwise features
        outer_prod = torch.einsum("bci,bcj->bcij", h, h)
        outer_sum = h.unsqueeze(-1) + h.unsqueeze(-2)
        outer_diff = (h.unsqueeze(-1) - h.unsqueeze(-2)).abs()
        dist = self.dist_enc(L, onehot.device).expand(B, -1, -1, -1)

        pair_feat = torch.cat([outer_prod, outer_sum, outer_diff, dist], dim=1)

        # Project + symmetrize before 2-D ResNet
        pair_feat = self.pair_proj(pair_feat)
        pair_feat = (pair_feat + pair_feat.transpose(-1, -2)) / 2

        for block in self.res_blocks:
            pair_feat = block(pair_feat)

        logits = self.head(pair_feat).squeeze(1)  # (B, L, L)
        return symmetrize(logits)

    def get_long_range_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for |i-j| >= min_separation."""
        return torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device),
            diagonal=self.cfg.min_separation,
        )


# ---------------------------------------------------------------------------
# BEACON official Top-L precision metric
# ---------------------------------------------------------------------------


def topL_precision(
    logits_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    min_separation: int = 23,
    fractions: Tuple[int, ...] = (1, 2, 5, 10),
) -> Dict[str, float]:
    """BEACON official Top-L precision: micro-averaged with 0.5 threshold.

    - Symmetrises and sigmoid-normalises logits per sample.
    - For each fraction k, takes the top-L/k predictions by probability.
    - Counts a prediction as TP only if its sigmoid score > 0.5.
    - Aggregates TP and FP across all samples (micro-averaging).

    Parameters
    ----------
    logits_list:
        Per-sample raw logit arrays of shape (L, L).
    labels_list:
        Per-sample binary label arrays of shape (L, L).
    min_separation:
        Minimum |i-j| for long-range evaluation.
    fractions:
        Denominators for L/k precision variants.

    Returns
    -------
    dict with keys ``top_l_precision``, ``top_l/2_precision``, etc.
    """
    tp_acc: Dict[int, float] = {f: 0.0 for f in fractions}
    fp_acc: Dict[int, float] = {f: 0.0 for f in fractions}

    for logits, labels in zip(logits_list, labels_list):
        labels = labels.astype(float)
        L = labels.shape[0]
        if L < min_separation + 1:
            continue

        # Symmetrise then sigmoid
        probs = torch.sigmoid(
            torch.from_numpy((logits + logits.T) / 2.0)
        ).numpy()

        lr_mask = np.zeros((L, L), dtype=bool)
        lr_mask[np.triu_indices(L, k=min_separation)] = True

        lr_probs = probs[lr_mask].ravel()
        lr_labels = labels[lr_mask].ravel()

        for f in fractions:
            k = max(1, min(L // f, lr_probs.size))
            top_idx = np.argsort(lr_probs)[-k:]
            top_probs = lr_probs[top_idx]
            top_labels = lr_labels[top_idx]

            above = top_probs > 0.5
            tp_acc[f] += float(top_labels[above].sum())
            fp_acc[f] += float((1.0 - top_labels[above]).sum())

    result: Dict[str, float] = {}
    for f in fractions:
        denom = tp_acc[f] + fp_acc[f]
        prec = tp_acc[f] / denom if denom > 0 else 0.0
        key = "top_l_precision" if f == 1 else f"top_l/{f}_precision"
        result[key] = float(prec)
    return result


__all__: List[str] = ["PythiaCMP", "topL_precision"]
