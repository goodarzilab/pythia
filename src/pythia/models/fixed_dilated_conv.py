"""FixedDilatedConv: frozen RNA base-pairing convolutional layer.

This is the canonical single implementation used by all Pythia structural
models.  Weights are pre-filled with Watson-Crick and wobble pair scores
and are frozen by default (trainable=False).

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class FixedDilatedConv(nn.Module):
    """RNA base-pairing detector with pre-filled, frozen weights.

    For each dilation radius in [dil_start, dil_end] and each bulge variant
    the layer creates a dedicated 1-D convolution kernel that scores the
    likelihood of a Watson-Crick or wobble base pair at that distance.

    Pair weights (raw / binarized):
        A-U : 1.00 / 1   U-A : 1.25 / 1
        C-G : 1.50 / 1   G-C : 1.75 / 1
        G-U : 0.50 / 1   U-G : 0.75 / 1

    Parameters
    ----------
    in_channel:
        Number of input channels (4 for one-hot RNA).
    dil_start:
        Smallest dilation radius.
    dil_end:
        Largest dilation radius.
    bulge_size:
        Maximum bulge displacement considered.
    trainable:
        Whether to allow gradient updates.  Always False in production.
    binarize_fd:
        Replace all non-zero pair weights with 1.
    """

    def __init__(
        self,
        in_channel: int = 4,
        dil_start: int = 5,
        dil_end: int = 24,
        bulge_size: int = 2,
        trainable: bool = False,
        binarize_fd: bool = False,
    ) -> None:
        super().__init__()
        self.binarize_fd = binarize_fd
        self.in_channel = in_channel
        self.dil_start = dil_start
        self.dil_end = dil_end
        self.bulge_size = bulge_size
        self.kernel_size: int = int(dil_end - dil_start + 1) * 2
        self.mid: int = self.kernel_size // 2

        # out_channel will be set by _build_weights
        self.out_channel: int = 0
        self.weight_names: List[str] = []

        weights = self._build_weights()
        self.weights = nn.Parameter(weights)
        if not trainable:
            self.weights.requires_grad_(False)

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------

    def _build_weights(self) -> torch.Tensor:
        """Construct and return the weight tensor with pair-specific kernels."""
        A, C, G, U = 0, 1, 2, 3

        if self.binarize_fd:
            weight_codes: List[Tuple[int, int, float]] = [
                (A, U, 1.0),
                (U, A, 1.0),
                (C, G, 1.0),
                (G, C, 1.0),
                (G, U, 1.0),
                (U, G, 1.0),
            ]
        else:
            weight_codes = [
                (A, U, 1.00),
                (U, A, 1.25),
                (C, G, 1.50),
                (G, C, 1.75),
                (G, U, 0.50),
                (U, G, 0.75),
            ]

        # Pre-allocate generous buffer; will be trimmed at the end
        max_kernels = len(weight_codes) * (self.dil_end - self.dil_start + 1) * (
            self.bulge_size * 2 + 1
        )
        buf = torch.zeros(max_kernels, self.in_channel, self.kernel_size)
        kernel_idx = 0
        names: List[str] = []

        for nuc1, nuc2, adweight in weight_codes:
            for dil in range(self.dil_start, self.dil_end + 1):
                part_1 = round(dil / 2)
                part_2 = dil - part_1 + 1
                for each_bulge in range(1, self.bulge_size + 1):
                    if each_bulge == 1:
                        bulge_offsets = [-1, 0, 1]
                    else:
                        bulge_offsets = [-each_bulge, each_bulge]

                    for bulge in bulge_offsets:
                        idx_pair = np.array(
                            [
                                int(self.kernel_size / 2) - part_1,
                                int(self.kernel_size / 2) + part_2,
                            ]
                        )
                        if bulge > 0:
                            idx_pair[idx_pair > self.mid] += bulge
                        elif bulge < 0:
                            idx_pair[idx_pair < self.mid] -= bulge

                        # Clamp to valid range
                        idx_pair = idx_pair[idx_pair < self.kernel_size]
                        idx_pair = idx_pair[idx_pair > 0]

                        idx_zero = np.setdiff1d(
                            np.arange(self.kernel_size), idx_pair
                        )
                        buf[kernel_idx, :, idx_zero] = 0.0
                        if len(idx_pair) >= 1:
                            buf[kernel_idx, nuc1, idx_pair[0]] = adweight
                        if len(idx_pair) >= 2:
                            buf[kernel_idx, nuc2, idx_pair[-1]] = adweight

                        names.append(f"{nuc1}_{nuc2} dil {dil} bulge {bulge}")
                        kernel_idx += 1

        self.out_channel = kernel_idx
        self.weight_names = names
        return buf[:kernel_idx].clone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fixed dilated convolution.

        Parameters
        ----------
        x:
            Input tensor of shape (B, 4, L).

        Returns
        -------
        torch.Tensor
            Output of shape (B, out_channel, L').
        """
        out = nn.functional.conv1d(x, self.weights)
        if self.binarize_fd:
            # Keep only the maximum activation per position, binarize
            out_max = out.amax(dim=0, keepdim=True)
            out = (out >= out_max).float() * (out > 0).float()
        return out
