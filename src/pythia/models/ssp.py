"""PythiaSSP: secondary structure prediction model (notebook-faithful).

Architecture:
    one-hot -> FixedDilatedConv (frozen) -> arm1 [K16, K10, K5]
    one-hot -> arm2 [K12, K7, K5]
    -> cat -> pairwise features (outer_prod, outer_sum, outer_diff, learned_dist_enc)
    -> pair_proj [Conv2d + GroupNorm(8) + GELU] -> symmetrize
    -> 2-D ResNet (GroupNorm + GELU, dilations=[1,2,4,1,2,4,1,1]) x num_res_layers
    -> 2-layer head [Conv2d(H, H//2) + GN(8, H//2) + GELU + Conv2d(H//2, 1)]
    -> symmetrize
    Loss: BCEWithLogitsLoss   Metrics: global F1, precision, recall

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.configs import SSPModelConfig
from pythia.models.common import (
    PositionalEncoding2DLearned,
    ResBlock2D,
    symmetrize,
)
from pythia.models.fixed_dilated_conv import FixedDilatedConv

# Multi-scale dilation pattern for the 2-D ResNet stack.
_SSP_DILATIONS: List[int] = [1, 2, 4, 1, 2, 4, 1, 1]


# ---------------------------------------------------------------------------
# SSP model
# ---------------------------------------------------------------------------


class PythiaSSP(nn.Module):
    """Secondary structure prediction model (notebook-faithful).

    Parameters
    ----------
    cfg:
        SSPModelConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: Optional[SSPModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SSPModelConfig()
        self.cfg = cfg

        # Fixed dilated conv (frozen)
        self.fd_conv = FixedDilatedConv(
            in_channel=4,
            dil_start=cfg.dil_start,
            dil_end=cfg.dil_end,
            bulge_size=cfg.bulge_size,
            trainable=False,
            binarize_fd=cfg.binarize_fd,
        )

        # Arm 1: after FD, kernels=[16, 10, 5]
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

        # Arm 2: on one-hot, kernels=[12, 7, 5]
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

        combined_1d = cfg.arm1_widths[2] + cfg.arm2_widths[2]
        pairwise_in = combined_1d * 3 + cfg.dist_enc_dim

        # Optional BatchNorm on concatenated arm features before pairwise building
        self.transition_norm: nn.Module = (
            nn.BatchNorm1d(combined_1d) if cfg.use_transition_norm else nn.Identity()
        )

        # Learned distance encoding
        self.pos_enc = PositionalEncoding2DLearned(channels=cfg.dist_enc_dim)

        # Pairwise projection
        self.pair_proj = nn.Sequential(
            nn.Conv2d(pairwise_in, cfg.hidden_dim, kernel_size=1),
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
        )

        # 2-D ResNet with multi-scale dilations
        dilations = (_SSP_DILATIONS + [1] * cfg.num_res_layers)[: cfg.num_res_layers]
        self.res_blocks = nn.ModuleList(
            [
                ResBlock2D(
                    cfg.hidden_dim,
                    num_groups=8,
                    dropout=cfg.dropout,
                    dilation=dilations[i],
                )
                for i in range(cfg.num_res_layers)
            ]
        )

        # Two-layer output head
        self.head = nn.Sequential(
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, cfg.hidden_dim // 2),
            nn.GELU(),
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
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _run_fd(self, onehot: torch.Tensor) -> torch.Tensor:
        """Run FixedDilatedConv with symmetric padding, crop to L."""
        L = onehot.shape[2]
        pad = self.fd_conv.kernel_size // 2
        padded = F.pad(onehot, (pad, pad))
        fd_out = self.fd_conv(padded)
        if fd_out.shape[2] > L:
            start = (fd_out.shape[2] - L) // 2
            fd_out = fd_out[:, :, start : start + L]
        elif fd_out.shape[2] < L:
            fd_out = F.pad(fd_out, (0, L - fd_out.shape[2]))
        return fd_out

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        onehot:
            One-hot encoded RNA of shape (B, 4, L).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, L, L).
        """
        B, _, L = onehot.shape
        fd_out = self._run_fd(onehot)

        # Arm 1: after FD
        feat1 = self.arm1(fd_out)[:, :, :L]
        # Arm 2: on one-hot
        feat2 = self.arm2(onehot)[:, :, :L]

        # Concatenate, optionally normalise, then build pairwise features
        combined = self.transition_norm(
            torch.cat([feat1, feat2], dim=1)
        )  # (B, arm1_out+arm2_out, L)
        ci = combined.unsqueeze(3).expand(-1, -1, -1, L)
        cj = combined.unsqueeze(2).expand(-1, -1, L, -1)
        outer_prod = ci * cj
        outer_sum = ci + cj
        outer_diff = (ci - cj).abs()

        dist_enc = self.pos_enc(L, onehot.device).expand(B, -1, -1, -1)
        pair_feat = torch.cat([outer_prod, outer_sum, outer_diff, dist_enc], dim=1)

        # Project and symmetrize before ResNet
        x = self.pair_proj(pair_feat)
        x = (x + x.transpose(-1, -2)) * 0.5

        # 2-D ResNet
        for block in self.res_blocks:
            x = block(x)

        # Output head
        logits = self.head(x).squeeze(1)  # (B, L, L)
        return symmetrize(logits)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_ssp_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute global (micro-averaged) precision, recall and F1 for SSP.

    Parameters
    ----------
    logits:
        Raw logits of shape (B, L, L).
    labels:
        Binary ground-truth of shape (B, L, L).
    mask:
        Boolean mask of valid positions.
    threshold:
        Decision threshold for sigmoid output.

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits[mask])
        preds = (probs > threshold).long()
        lbls = labels[mask].long()

        tp = float((preds * lbls).sum())
        fp = float((preds * (1 - lbls)).sum())
        fn = float(((1 - preds) * lbls).sum())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


__all__: List[str] = [
    "PythiaSSP",
    "compute_ssp_metrics",
]
