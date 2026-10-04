"""Pydantic v2 configuration classes for all Pythia models and training runs.

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared / base configs
# ---------------------------------------------------------------------------


class FixedDilatedConvConfig(BaseModel):
    """Configuration for the FixedDilatedConv layer."""

    dil_start: int = Field(default=5, ge=1, description="Minimum dilation radius.")
    dil_end: int = Field(default=24, ge=1, description="Maximum dilation radius.")
    bulge_size: int = Field(default=2, ge=1, description="Number of bulge sizes.")
    binarize_fd: bool = Field(
        default=False, description="Binarize fixed dilated conv outputs."
    )
    trainable: bool = Field(
        default=False, description="Allow gradient updates to FD weights."
    )

    @model_validator(mode="after")
    def check_dil_range(self) -> "FixedDilatedConvConfig":
        if self.dil_end < self.dil_start:
            raise ValueError("dil_end must be >= dil_start.")
        return self


# ---------------------------------------------------------------------------
# RBP model config
# ---------------------------------------------------------------------------


class RBPModelConfig(BaseModel):
    """Architecture hyperparameters for PythiaRBP."""

    inputsize: int = Field(default=256, ge=1, description="Input sequence length.")
    init_channels: int = Field(
        default=64, ge=1, description="Width for arm1 conv layers."
    )
    kernel_size: int = Field(
        default=16, ge=1, description="First kernel size for arm1."
    )
    dp: float = Field(default=0.25, ge=0.0, le=1.0, description="Dropout rate.")
    dil_start: int = Field(default=5, ge=1)
    dil_end: int = Field(default=24, ge=1)
    bulge_size: int = Field(default=2, ge=1)
    binarize_fd: bool = Field(default=False)

    @model_validator(mode="after")
    def check_dil_range(self) -> "RBPModelConfig":
        if self.dil_end < self.dil_start:
            raise ValueError("dil_end must be >= dil_start.")
        return self


class RBPTrainConfig(BaseModel):
    """Training hyperparameters for PythiaRBP (pytorch-lightning)."""

    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    max_epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=64, ge=1)
    num_workers: int = Field(default=4, ge=0)
    patience: int = Field(default=5, ge=1)
    warmup_epochs: int = Field(default=2, ge=0)
    gradient_clip_val: float = Field(default=1.0, ge=0.0)
    precision: str = Field(default="16-mixed", description="Lightning precision flag.")
    pos_weight: float = Field(
        default=0.2,
        gt=0.0,
        description=(
            "Weight assigned to the negative (Unbound) class in CrossEntropyLoss. "
            "The positive (Bound) class always receives weight 1.0, so the default "
            "of 0.2 gives the positive class 5x more influence -- appropriate for "
            "the typical ~10:1 negative:positive class imbalance in CLIP-seq data."
        ),
    )


# ---------------------------------------------------------------------------
# Shared 2D backbone config
# ---------------------------------------------------------------------------


class DualArmConfig(BaseModel):
    """Dual-arm 1D backbone configuration shared by SSP / CMP / DMP."""

    dil_start: int = Field(default=5, ge=1)
    dil_end: int = Field(default=24, ge=1)
    bulge_size: int = Field(default=2, ge=1)
    binarize_fd: bool = Field(default=False)
    arm1_widths: Tuple[int, int, int] = Field(default=(128, 128, 128))
    arm2_widths: Tuple[int, int, int] = Field(default=(256, 128, 128))
    dropout: float = Field(default=0.15, ge=0.0, le=1.0)
    use_transition_norm: bool = Field(
        default=True,
        description=(
            "Insert a BatchNorm1d layer on the concatenated arm features "
            "before the pairwise projection. Enabled by default for structural "
            "tasks; not used by RBP."
        ),
    )

    @model_validator(mode="after")
    def check_dil_range(self) -> "DualArmConfig":
        if self.dil_end < self.dil_start:
            raise ValueError("dil_end must be >= dil_start.")
        return self


# ---------------------------------------------------------------------------
# SSP config
# ---------------------------------------------------------------------------


class SSPModelConfig(DualArmConfig):
    """Architecture hyperparameters for PythiaSSP."""

    hidden_dim: int = Field(default=256, ge=1)
    num_res_layers: int = Field(default=8, ge=1)
    # Learned distance bins (64 dims = hidden_dim // 4)
    dist_enc_dim: int = Field(default=64, ge=1)
    arm1_widths: Tuple[int, int, int] = Field(default=(128, 128, 128))
    arm2_widths: Tuple[int, int, int] = Field(default=(256, 128, 128))
    dil_end: int = Field(default=24, ge=1)
    dropout: float = Field(default=0.15)


class SSPTrainConfig(BaseModel):
    """Training hyperparameters for PythiaSSP."""

    lr: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_epochs: int = Field(default=100, ge=1)
    # batch_size=4, grad_accum=2 -> effective batch 8
    batch_size: int = Field(default=4, ge=1)
    batch_eval: int = Field(default=4, ge=1)
    grad_accum: int = Field(default=2, ge=1)
    num_workers: int = Field(default=4, ge=0)
    patience: int = Field(default=3, ge=1)
    warmup_epochs: int = Field(default=3, ge=0)
    max_len: int = Field(default=512, ge=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# CMP config
# ---------------------------------------------------------------------------


class CMPModelConfig(DualArmConfig):
    """Architecture hyperparameters for PythiaCMP."""

    hidden_dim: int = Field(default=224, ge=1)
    num_res_layers: int = Field(default=7, ge=1)
    # Learned distance bins
    dist_enc_dim: int = Field(default=64, ge=1)
    min_separation: int = Field(
        default=23, ge=1, description="Minimum |i-j| for long-range contacts."
    )
    # Tapered arm widths
    arm1_widths: Tuple[int, int, int] = Field(default=(128, 96, 64))
    arm2_widths: Tuple[int, int, int] = Field(default=(192, 128, 64))
    dil_end: int = Field(default=24, ge=1)
    dropout: float = Field(default=0.15)


class CMPTrainConfig(BaseModel):
    """Training hyperparameters for PythiaCMP."""

    lr: float = Field(default=5e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=1, ge=1)
    batch_eval: int = Field(default=1, ge=1)
    # Effective batch size = batch_size * grad_accum
    grad_accum: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    patience: int = Field(default=3, ge=1)
    warmup_epochs: int = Field(default=1, ge=0)
    max_len: int = Field(default=1024, ge=1)


# ---------------------------------------------------------------------------
# DMP config
# ---------------------------------------------------------------------------


class DMPModelConfig(DualArmConfig):
    """Architecture hyperparameters for PythiaDMP."""

    hidden_dim: int = Field(default=228, ge=1)
    num_res_layers: int = Field(default=7, ge=1)
    dist_enc_dim: int = Field(default=64, ge=1)
    arm1_widths: Tuple[int, int, int] = Field(default=(192, 128, 128))
    arm2_widths: Tuple[int, int, int] = Field(default=(256, 192, 128))
    dil_end: int = Field(default=24, ge=1)
    dropout: float = Field(default=0.15)
    # max_distance only relevant for legacy bpRNA normalization path
    max_distance: float = Field(default=20.0, gt=0.0)


class DMPTrainConfig(BaseModel):
    """Training hyperparameters for PythiaDMP."""

    lr: float = Field(default=5e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=1, ge=1)
    batch_eval: int = Field(default=1, ge=1)
    # Effective batch = batch_size x grad_accum
    grad_accum: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    patience: int = Field(default=5, ge=1)
    warmup_epochs: int = Field(default=5, ge=0)
    max_len: int = Field(default=1024, ge=1)
    huber_delta: float = Field(default=0.1, gt=0.0)
    # Early stopping cannot trigger before this many epochs have run.
    min_epochs: int = Field(default=1, ge=1)


# ---------------------------------------------------------------------------
# SSI config
# ---------------------------------------------------------------------------


class SSIModelConfig(BaseModel):
    """Architecture hyperparameters for PythiaSSI.

    Defaults match the deployed configuration found by hyperopt TPE search
    (50 trials, seed=42): val R2=0.522, test R2=0.396. The model always
    conditions on observed structural scores (arm2 receives the one-hot
    sequence plus the observed-score and known-mask channels) and applies
    squeeze-and-excite channel attention to the FixedDilatedConv features
    before arm1 -- both were the winning ablation flags and are therefore
    not exposed as toggles here.
    """

    hidden_dim: int = Field(default=384, ge=1)
    num_res_layers: int = Field(default=10, ge=1)
    arm1_widths: Tuple[int, int, int] = Field(default=(128, 128, 128))
    arm2_widths: Tuple[int, int, int] = Field(default=(256, 128, 128))
    dil_start: int = Field(default=2, ge=1)
    dil_end: int = Field(default=36, ge=1)
    bulge_size: int = Field(default=4, ge=1)
    binarize_fd: bool = Field(default=False)
    dropout: float = Field(default=0.307, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_dil_range(self) -> "SSIModelConfig":
        if self.dil_end < self.dil_start:
            raise ValueError("dil_end must be >= dil_start.")
        return self


class SSITrainConfig(BaseModel):
    """Training hyperparameters for PythiaSSI (deployed configuration)."""

    lr_feat: float = Field(default=7.071e-4, gt=0.0)
    lr_head: float = Field(default=2.598e-4, gt=0.0)
    weight_decay: float = Field(default=8.864e-3, ge=0.0)
    max_epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=64, ge=1)
    batch_eval: int = Field(default=128, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    num_workers: int = Field(default=4, ge=0)
    patience: int = Field(default=12, ge=1)
    warmup_epochs: int = Field(default=20, ge=0)
    max_len: int = Field(default=440, ge=1)
    loss_fn: str = Field(
        default="l1",
        description="Loss function: mse, huber, or l1.",
    )
    huber_delta: float = Field(
        default=0.1,
        gt=0.0,
        description="Delta parameter for Huber loss (only used when loss_fn=huber).",
    )
    grad_clip: float = Field(
        default=1.152,
        ge=0.0,
        description="Max gradient norm for clipping (0 disables clipping).",
    )


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------


class InferConfig(BaseModel):
    """Configuration for inference runs."""

    task: str = Field(
        description="One of: rbp, ssp, cmp, dmp, ssi.",
    )
    checkpoint: str = Field(description="Path to model checkpoint (.pt file).")
    input_csv: str = Field(description="Path to input CSV file.")
    output_csv: str = Field(description="Path to write predictions.")
    batch_size: int = Field(default=4, ge=1)
    num_workers: int = Field(default=2, ge=0)
    device: Optional[str] = Field(default=None, description="torch device string.")
    max_len: int = Field(default=512, ge=1)


# ---------------------------------------------------------------------------
# Convenience: all config classes for export
# ---------------------------------------------------------------------------

ALL_CONFIGS: List[type] = [
    FixedDilatedConvConfig,
    RBPModelConfig,
    RBPTrainConfig,
    SSPModelConfig,
    SSPTrainConfig,
    CMPModelConfig,
    CMPTrainConfig,
    DMPModelConfig,
    DMPTrainConfig,
    SSIModelConfig,
    SSITrainConfig,
    InferConfig,
]
