"""PythiaDMP: distance map prediction model.

Architecture (faithful to the BEACON distance-map benchmark reference model):
    One-hot → FD (frozen) → arm1 [K16, K10, K5] ────────────────────────┐
    One-hot → arm2 [K12, K7,  K5] ──────────────────────────────────────┼→ concat
                                                                         ↓
        Standard pairwise:
            outer_prod, outer_sum, outer_diff          (3 × total_1d)
        Distance-specific pairwise:
            weighted_diff = outer_diff × seq_dist_norm (  total_1d)
            weighted_sum  = outer_sum  × inv_seq_dist  (  total_1d)
            seq_dist_feat (1 channel:  |i-j| / L)
            decay_feat    (3 channels: exp(-|i-j|/10), /50, /100)
        Learned position encoding                      (dist_enc_dim)
        Total: 5 × total_1d + 4 + dist_enc_dim  (≈ 1348 with defaults)
                                                                         ↓
                pair_proj (Conv2d + BN + ReLU) → symmetrize
                                                                         ↓
                2D ResNet (BatchNorm + ReLU, multi-scale dilations)
                                                                         ↓
                    2-layer output head → sigmoid → symmetrize → (B, L, L)

    Output: [0, 1] normalised distances
    Loss:   HuberLoss(delta=0.1)
    Metric: R² computed globally (Pearson r² over all non-padded positions)

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.configs import DMPModelConfig
from pythia.models.common import (
    PositionalEncoding2DLearned,
    ResBlock2DBN,
    symmetrize,
)
from pythia.models.fixed_dilated_conv import FixedDilatedConv

# Multi-scale dilation pattern (same as CMP)
_DILATION_PATTERN: Tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8, 1, 1)
_DECAY_SCALES: Tuple[int, ...] = (10, 50, 100)


class PythiaDMP(nn.Module):
    """Distance map prediction model.

    Arms feed directly into pairwise features without an intermediate
    hidden-dim projection, and distance-specific features augment the
    standard outer-product representation.

    Parameters
    ----------
    cfg:
        DMPModelConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: Optional[DMPModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = DMPModelConfig()
        self.cfg = cfg

        # ------------------------------------------------------------------ #
        # Fixed dilated conv (always frozen)
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
        # Distance encoding (learned 16-bin embeddings)
        # ------------------------------------------------------------------ #
        self.dist_enc = PositionalEncoding2DLearned(cfg.dist_enc_dim)

        # ------------------------------------------------------------------ #
        # Pairwise projection
        # 5 × total_1d (prod, sum, diff, w_diff, w_sum)
        # + 4 (1 raw dist + 3 decay scales)
        # + dist_enc_dim
        # ------------------------------------------------------------------ #
        seq_dist_channels = 1 + len(_DECAY_SCALES)  # 4
        pairwise_in = total_1d * 5 + seq_dist_channels + cfg.dist_enc_dim
        self.pair_proj = nn.Sequential(
            nn.Conv2d(pairwise_in, cfg.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------ #
        # 2-D ResNet with multi-scale dilations
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
        distances : (B, L, L) in [0, 1]
        """
        B, _, L = onehot.shape

        # 1-D features
        h2 = self._align(self.arm2(onehot), L)               # (B, arm2[-1], L)
        h1 = self._align(self.arm1(self._run_fd(onehot)), L)  # (B, arm1[-1], L)
        h = self.transition_norm(torch.cat([h1, h2], dim=1))  # (B, total_1d, L)

        # Standard pairwise features
        outer_prod = torch.einsum("bci,bcj->bcij", h, h)
        outer_sum = h.unsqueeze(-1) + h.unsqueeze(-2)
        outer_diff = (h.unsqueeze(-1) - h.unsqueeze(-2)).abs()

        # Sequence distance matrix (raw integer distances)
        idx = torch.arange(L, device=onehot.device, dtype=torch.float32)
        seq_dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # (L, L)
        seq_dist_norm = seq_dist / float(L)                         # [0, 1]

        # Raw distance feature (1 channel)
        seq_dist_feat = seq_dist_norm.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        # Multi-scale exponential decay features (3 channels)
        decays = [torch.exp(-seq_dist / float(s)) for s in _DECAY_SCALES]
        decay_feat = torch.stack(decays, dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        # Distance-weighted difference (far pairs → large diff signal)
        weighted_diff = outer_diff * seq_dist_norm.unsqueeze(0).unsqueeze(0)

        # Inverse-distance-weighted sum (close pairs → large similarity signal)
        # Use raw seq_dist (not normalised) so denominator stays in a sensible range
        inv_seq_dist = 1.0 / (seq_dist + 1.0)
        weighted_sum = outer_sum * inv_seq_dist.unsqueeze(0).unsqueeze(0)

        # Learned position encoding
        dist_enc = self.dist_enc(L, onehot.device).expand(B, -1, -1, -1)

        # Concatenate all pairwise features: 5*total_1d + 4 + dist_enc_dim
        pair_feat = torch.cat([
            outer_prod,      # total_1d
            outer_sum,       # total_1d
            outer_diff,      # total_1d
            weighted_diff,   # total_1d
            weighted_sum,    # total_1d
            seq_dist_feat,   # 1
            decay_feat,      # 3
            dist_enc,        # dist_enc_dim
        ], dim=1)

        # Project + symmetrize before 2-D ResNet
        pair_feat = self.pair_proj(pair_feat)
        pair_feat = (pair_feat + pair_feat.transpose(-1, -2)) / 2

        for block in self.res_blocks:
            pair_feat = block(pair_feat)

        distances = self.head(pair_feat).squeeze(1)  # (B, L, L)
        distances = torch.sigmoid(distances)
        return symmetrize(distances)


# ---------------------------------------------------------------------------
# BEACON official DMP metric: global Pearson R²
# ---------------------------------------------------------------------------


def compute_dmp_metrics(
    preds_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    pad_value: float = -1.0,
) -> Dict[str, float]:
    """BEACON official DMP metric: global Pearson R² and MSE.

    - Concatenates all non-padded positions across samples (micro).
    - Computes a single global Pearson r², then squares it.

    Parameters
    ----------
    preds_list:
        Per-sample prediction arrays of shape (L, L), values in [0, 1].
    labels_list:
        Per-sample ground-truth arrays of shape (L, L); ``pad_value`` marks
        padded cells excluded from the metric.
    pad_value:
        Sentinel for padded positions (default -1).

    Returns
    -------
    dict with keys ``r2`` and ``mse``.
    """
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for preds, labels in zip(preds_list, labels_list):
        labels = labels.squeeze().astype(float)
        preds = preds.squeeze()
        valid = labels != pad_value
        all_preds.append(preds[valid].ravel())
        all_labels.append(labels[valid].ravel())

    if not all_preds:
        return {"r2": 0.0, "mse": 0.0}

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    mse = float(np.mean((y_true - y_pred) ** 2))

    if len(y_true) < 2 or y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        return {"r2": 0.0, "mse": mse}

    r_val = pearsonr(y_true, y_pred)[0]
    r2 = float(np.nan_to_num(r_val) ** 2)
    return {"r2": r2, "mse": mse}


def normalize_distance_map(
    dist: np.ndarray, max_distance: float = 20.0
) -> np.ndarray:
    """Symmetrize, clip, and normalize a raw Angstrom distance map to [0, 1].

    Used only by the legacy bpRNA-format code path.
    BEACON DistanceMap .npy files are already in [0, 1] and do not need this.
    """
    dist = np.nan_to_num(dist, nan=max_distance)
    dist = np.clip(dist, 0.0, max_distance)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return (dist / max_distance).astype(np.float32)


__all__: List[str] = ["PythiaDMP", "compute_dmp_metrics", "normalize_distance_map"]
