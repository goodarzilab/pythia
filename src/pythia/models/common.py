"""Shared building blocks used across Pythia structural models.

Components:
    ResBlock1D         -- 1-D residual block (BatchNorm + GELU)
    ResBlock2D         -- 2-D residual block (GroupNorm + GELU)
    PositionalEncoding2D -- sinusoidal distance encoding for pairwise grids

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1-D Residual Block
# ---------------------------------------------------------------------------


class ResBlock1D(nn.Module):
    """1-D residual block with BatchNorm and GELU activation.

    Parameters
    ----------
    channels:
        Number of input and output channels.
    dropout:
        Dropout probability applied inside the block.
    dilation:
        Dilation for the first conv (second conv always uses dilation=1).
    """

    def __init__(self, channels: int, dropout: float = 0.1, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=3,
                padding=dilation, dilation=dilation, bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ---------------------------------------------------------------------------
# 2-D Residual Block
# ---------------------------------------------------------------------------


class ResBlock2D(nn.Module):
    """2-D residual block with GroupNorm and GELU activation.

    Designed for (B, C, L, L) pairwise feature maps.

    Parameters
    ----------
    channels:
        Number of input and output channels.
    num_groups:
        Number of groups for GroupNorm.  Must divide `channels`.
    dropout:
        Dropout probability applied inside the block.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int = 8,
        dropout: float = 0.15,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        # Ensure num_groups divides channels
        while channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2

        self.net = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3,
                padding=dilation, dilation=dilation, bias=False,
            ),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ---------------------------------------------------------------------------
# 2-D Positional / Distance Encoding
# ---------------------------------------------------------------------------


class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2-D distance encoding for pairwise (i, j) position grids.

    Encodes |i - j| (sequence separation) using a set of sinusoidal functions
    at different scales and returns a (B, enc_dim, L, L) tensor.

    Parameters
    ----------
    enc_dim:
        Number of encoding channels (must be even).
    max_len:
        Maximum sequence length supported.
    """

    def __init__(self, enc_dim: int = 32, max_len: int = 1024) -> None:
        super().__init__()
        if enc_dim % 2 != 0:
            raise ValueError("enc_dim must be even.")
        self.enc_dim = enc_dim

        # Pre-compute 1-D encoding table (max_len, enc_dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, enc_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / enc_dim)
        )
        pe = torch.zeros(max_len, enc_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, enc_dim)

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        """Return distance encoding of shape (1, enc_dim, L, L).

        Parameters
        ----------
        L:
            Current sequence length.
        device:
            Target device for the output tensor.
        """
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        # |i - j| distances
        idx = torch.arange(L, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().clamp(max=pe.shape[0] - 1)
        # dist: (L, L)  look up in pe table
        enc = pe[dist.view(-1)].view(L, L, self.enc_dim)  # (L, L, enc_dim)
        enc = enc.permute(2, 0, 1).unsqueeze(0)  # (1, enc_dim, L, L)
        return enc


# ---------------------------------------------------------------------------
# Pairwise feature construction helpers
# ---------------------------------------------------------------------------


def make_pairwise_features(
    tokens: torch.Tensor,
    dist_enc: PositionalEncoding2D,
) -> torch.Tensor:
    """Build pairwise feature map from per-token features.

    Computes outer product, outer sum, outer difference, and a distance
    encoding, then concatenates them along the channel dimension.

    Parameters
    ----------
    tokens:
        Per-token features of shape (B, H, L).
    dist_enc:
        Positional encoding module.

    Returns
    -------
    torch.Tensor
        Pairwise feature tensor of shape (B, 3*H + enc_dim, L, L).
    """
    B, H, L = tokens.shape
    ti = tokens.unsqueeze(3).expand(B, H, L, L)  # (B, H, L, L)
    tj = tokens.unsqueeze(2).expand(B, H, L, L)  # (B, H, L, L)

    outer_prod = ti * tj
    outer_sum = ti + tj
    outer_diff = (ti - tj).abs()

    dist = dist_enc(L, tokens.device).expand(B, -1, L, L)

    return torch.cat([outer_prod, outer_sum, outer_diff, dist], dim=1)


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    """Average a (B, L, L) tensor with its transpose for symmetry."""
    return 0.5 * (x + x.transpose(-1, -2))


# ---------------------------------------------------------------------------
# BN-based 2-D Residual Block (pre-activation, optional dilation)
# ---------------------------------------------------------------------------


class ResBlock2DBN(nn.Module):
    """Pre-activation 2-D residual block with BatchNorm + ReLU.

    Matches the notebook's ResBlock2D used in PythiaCMP v2.
    Supports per-block dilation for multi-scale receptive fields.

    Parameters
    ----------
    channels:
        Number of input and output channels.
    dropout:
        Dropout probability.
    dilation:
        Dilation for the first conv.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return residual + x


# ---------------------------------------------------------------------------
# Learned distance encoding (binned embeddings)
# ---------------------------------------------------------------------------


class PositionalEncoding2DLearned(nn.Module):
    """Learned distance encoding using 16 distance bins.

    Distances 0-7 are exact; distances >=8 are binned log-scale up to bin 15.
    Returns a (1, channels, L, L) tensor.

    Parameters
    ----------
    channels:
        Embedding dimension.
    """

    NUM_BINS: int = 16

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dist_embed = nn.Embedding(self.NUM_BINS, channels)

    @staticmethod
    def _get_dist_bin(dist: torch.Tensor) -> torch.Tensor:
        """Map absolute integer distances to one of 16 bins."""
        b = torch.zeros_like(dist)
        b = torch.where(dist < 8, dist, b)
        b = torch.where((dist >= 8) & (dist < 16), torch.full_like(b, 8), b)
        b = torch.where((dist >= 16) & (dist < 32), torch.full_like(b, 9), b)
        b = torch.where((dist >= 32) & (dist < 64), torch.full_like(b, 10), b)
        b = torch.where((dist >= 64) & (dist < 128), torch.full_like(b, 11), b)
        b = torch.where((dist >= 128) & (dist < 256), torch.full_like(b, 12), b)
        b = torch.where((dist >= 256) & (dist < 512), torch.full_like(b, 13), b)
        b = torch.where((dist >= 512) & (dist < 1024), torch.full_like(b, 14), b)
        b = torch.where(dist >= 1024, torch.full_like(b, 15), b)
        return b.long()

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        """Return (1, channels, L, L) distance encoding."""
        idx = torch.arange(L, device=device)
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        dist_bins = self._get_dist_bin(dist)
        enc = self.dist_embed(dist_bins)       # (L, L, channels)
        return enc.permute(2, 0, 1).unsqueeze(0)  # (1, channels, L, L)
