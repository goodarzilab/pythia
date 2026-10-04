"""PythiaSSI: structural score imputation model (1-D per-position regression).

Architecture:
    one-hot -> FixedDilatedConv (frozen) -> squeeze-and-excite
    -> arm1 [K16, K10, K5] (no pooling)
    one-hot + observed-score channel + known-mask channel -> arm2 [K12, K7, K5]
    -> cat -> Conv1d(1x1) -> ResBlock1D x num_res_layers
    -> LayerNorm -> regression head -> per-position scalar

    hidden_dim     = 384
    num_res_layers = 10
    dil_start      = 2
    dil_end        = 36
    bulge_size     = 4
    dropout        = 0.307
    arm1_widths    = (128, 128, 128)
    arm2_widths    = (256, 128, 128)

    Loss: configurable (mse / huber / l1) on masked (imputed) positions
    Metric: R2 (Pearson r2)

Observation conditioning and squeeze-and-excite attention on the FD features
were the winning flags from an ablation study over the base architecture and
are therefore always enabled -- this is the single deployed PythiaSSI
architecture, not a togglable option.

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.configs import SSIModelConfig
from pythia.models.common import ResBlock1D
from pythia.models.fixed_dilated_conv import FixedDilatedConv


class _SEBlock1D(nn.Module):
    """Channel squeeze-and-excite for (B, C, L) tensors."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x.mean(dim=2))  # global avg pool -> (B, C)
        return x * scale.unsqueeze(2)


class PythiaSSI(nn.Module):
    """Structural score imputation model.

    Parameters
    ----------
    cfg:
        SSIModelConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: Optional[SSIModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SSIModelConfig()
        self.cfg = cfg

        # Frozen fixed dilated conv
        self.fd_conv = FixedDilatedConv(
            in_channel=4,
            dil_start=cfg.dil_start,
            dil_end=cfg.dil_end,
            bulge_size=cfg.bulge_size,
            trainable=False,
            binarize_fd=cfg.binarize_fd,
        )
        self.se = _SEBlock1D(self.fd_conv.out_channel)

        # Arm 1: CNN on (squeeze-excited) FD features
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

        # Arm 2: one-hot + observed score + known mask (6 channels)
        self.arm2 = nn.Sequential(
            nn.Conv1d(6, cfg.arm2_widths[0], kernel_size=12, padding=6),
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

        combined_dim = cfg.arm1_widths[2] + cfg.arm2_widths[2]
        self.proj = nn.Conv1d(combined_dim, cfg.hidden_dim, kernel_size=1)

        self.refinement = nn.ModuleList(
            [ResBlock1D(cfg.hidden_dim, cfg.dropout, dilation=1) for _ in range(cfg.num_res_layers)]
        )

        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def _run_fd(self, onehot: torch.Tensor) -> torch.Tensor:
        L = onehot.shape[2]
        pad = self.fd_conv.kernel_size // 2
        padded = F.pad(onehot, (pad, pad))
        fd_out = self.fd_conv(padded)
        if fd_out.shape[2] > L:
            fd_out = fd_out[:, :, :L]
        elif fd_out.shape[2] < L:
            fd_out = F.pad(fd_out, (0, L - fd_out.shape[2]))
        return fd_out

    def forward(
        self,
        onehot: torch.Tensor,
        obs_normed: torch.Tensor,
        obs_known: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        onehot:
            One-hot encoded RNA of shape (B, 4, L).
        obs_normed:
            Observed structural scores, zero at masked positions, shape (B, L).
        obs_known:
            Boolean mask: True at observed positions, shape (B, L).

        Returns
        -------
        torch.Tensor
            Per-position predictions of shape (B, L).
        """
        B, _, L = onehot.shape

        fd_out = self.se(self._run_fd(onehot))
        feat1 = self.arm1(fd_out)[:, :, :L]

        arm2_input = torch.cat(
            [onehot, obs_normed.unsqueeze(1), obs_known.float().unsqueeze(1)], dim=1
        )
        feat2 = self.arm2(arm2_input)[:, :, :L]

        min_l = min(feat1.shape[2], feat2.shape[2], L)
        x = self.proj(torch.cat([feat1[:, :, :min_l], feat2[:, :, :min_l]], dim=1))

        for block in self.refinement:
            x = block(x)

        # x: (B, H, L') -> (B, L', H)
        x = x.transpose(1, 2)
        if x.shape[1] < L:
            x = F.pad(x, (0, 0, 0, L - x.shape[1]))
        elif x.shape[1] > L:
            x = x[:, :L, :]

        x = self.final_norm(x)
        return self.head(x).squeeze(-1)  # (B, L)


__all__: List[str] = ["PythiaSSI"]
