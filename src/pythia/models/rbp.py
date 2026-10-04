"""PythiaRBP: binary classification model for RNA-binding protein binding.

Architecture:
    one-hot (B, 4, L)
      -> FixedDilatedConv (frozen)
      -> AdaptiveMaxPool1d(60)
      -> BatchNorm + ReLU
      -> arm1 [K16, K10, K5] (from dil features, pooled)
    +
    one-hot (B, 4, L)
      -> arm2 [K12, K7, K5]
    ->  cat -> dense(64) -> dense(32) -> dense(2)
    Loss: CrossEntropyLoss

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.configs import RBPModelConfig, RBPTrainConfig
from pythia.models.fixed_dilated_conv import FixedDilatedConv

if TYPE_CHECKING:
    import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class ConvArm(nn.Module):
    """Three-layer 1-D convolutional arm with BatchNorm, ReLU, MaxPool.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    widths:
        Output channel widths for the three conv layers.
    kernels:
        Kernel sizes for the three conv layers.
    pools:
        Max-pool sizes for the three layers.
    dropout:
        Dropout probability applied after each ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        widths: Tuple[int, int, int] = (64, 64, 64),
        kernels: Tuple[int, int, int] = (16, 10, 5),
        pools: Tuple[int, int, int] = (2, 2, 2),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], kernel_size=kernels[0]),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pools[0]),
            nn.Conv1d(widths[0], widths[1], kernel_size=kernels[1]),
            nn.BatchNorm1d(widths[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pools[1]),
            nn.Conv1d(widths[1], widths[2], kernel_size=kernels[2]),
            nn.BatchNorm1d(widths[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pools[2]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class PythiaRBP(nn.Module):
    """Pythia RBP binary classification model.

    Parameters
    ----------
    cfg:
        Architecture hyperparameter config.
    """

    def __init__(self, cfg: Optional[RBPModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = RBPModelConfig()
        self.cfg = cfg

        # Fixed dilated conv (always frozen)
        self.conv_fd = FixedDilatedConv(
            in_channel=4,
            dil_start=cfg.dil_start,
            dil_end=cfg.dil_end,
            bulge_size=cfg.bulge_size,
            trainable=False,
            binarize_fd=cfg.binarize_fd,
        )

        # Pool to fixed size for arm1
        self.pool_fd = nn.AdaptiveMaxPool1d(60)
        self.bn_fd = nn.Sequential(
            nn.BatchNorm1d(60),
            nn.ReLU(),
        )

        # Arm 1: processes dilated (transposed) features
        self.arm1 = ConvArm(
            in_channels=60,
            widths=(cfg.init_channels, cfg.init_channels, cfg.init_channels),
            kernels=(16, 10, 5),
            pools=(2, 2, 2),
            dropout=cfg.dp,
        )

        # Arm 2: processes raw one-hot
        self.arm2 = ConvArm(
            in_channels=4,
            widths=(128, 64, 64),
            kernels=(12, 7, 5),
            pools=(2, 2, 2),
            dropout=cfg.dp,
        )

        # Compute linear input size dynamically
        lin_dim = self._compute_lin_dim()

        self.dense = nn.Sequential(
            nn.Linear(lin_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self._init_weights()

    def _compute_lin_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 4, self.cfg.inputsize)
            out = self._forward_features(x)
        return out.shape[1]

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Pad for FD conv so output spans full sequence
        pad = x.shape[2]
        x_pad = F.pad(x, (pad, pad))
        fd_out = self.conv_fd(x_pad)
        fd_pooled = self.pool_fd(fd_out)
        # Transpose: (B, channels, 60) -> (B, 60, channels)
        fd_pooled = fd_pooled.transpose(1, 2)
        fd_processed = self.bn_fd(fd_pooled)
        out1 = self.arm1(fd_processed)
        out2 = self.arm2(x)
        merged = torch.cat([out1, out2], dim=2)
        return merged.view(merged.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            One-hot encoded RNA of shape (B, 4, L).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 2).
        """
        features = self._forward_features(x)
        return self.dense(features)


# ---------------------------------------------------------------------------
# Lightning wrapper (imported lazily to avoid scipy at module load time)
# ---------------------------------------------------------------------------


def _make_rbp_lightning_module():
    """Create PythiaRBPModule class with lazy pytorch_lightning import."""
    import pytorch_lightning as pl  # noqa: PLC0415

    class _PythiaRBPModule(pl.LightningModule):
        """PyTorch Lightning module wrapping PythiaRBP for training.

        Parameters
        ----------
        model_cfg:
            Architecture config.
        train_cfg:
            Training hyperparameter config.
        """

        def __init__(
            self,
            model_cfg: Optional[RBPModelConfig] = None,
            train_cfg: Optional[RBPTrainConfig] = None,
        ) -> None:
            super().__init__()
            if model_cfg is None:
                model_cfg = RBPModelConfig()
            if train_cfg is None:
                train_cfg = RBPTrainConfig()

            self.model_cfg = model_cfg
            self.train_cfg = train_cfg
            self.model = PythiaRBP(model_cfg)
            # Weight negative (Unbound, class 0) by pos_weight and positive
            # (Bound, class 1) by 1.0, giving the positive class
            # 1 / pos_weight times more influence in the loss.
            class_weights = torch.tensor(
                [train_cfg.pos_weight, 1.0], dtype=torch.float32
            )
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.save_hyperparameters()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def _shared_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
        ) -> torch.Tensor:
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
            self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
            return loss

        def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
            return self._shared_step(batch, "train")

        def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> None:
            self._shared_step(batch, "val")

        def test_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> None:
            self._shared_step(batch, "test")

        def configure_optimizers(self) -> dict:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.train_cfg.lr,
                weight_decay=self.train_cfg.weight_decay,
            )
            steps_per_epoch: int = self.trainer.estimated_stepping_batches // max(
                1, self.train_cfg.max_epochs
            )
            warmup_steps = steps_per_epoch * self.train_cfg.warmup_epochs
            total_steps = steps_per_epoch * self.train_cfg.max_epochs

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (
                    1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        def get_class_probabilities(self, x: torch.Tensor) -> torch.Tensor:
            """Return softmax probabilities of shape (B, 2)."""
            self.eval()
            with torch.no_grad():
                return F.softmax(self(x), dim=1)

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path: str, **kwargs):  # type: ignore[override]
            return super().load_from_checkpoint(checkpoint_path, **kwargs)

    return _PythiaRBPModule


def PythiaRBPModule(  # noqa: N802
    model_cfg: Optional[RBPModelConfig] = None,
    train_cfg: Optional[RBPTrainConfig] = None,
):
    """Factory that instantiates the Lightning-backed RBP training module.

    Lightning is imported lazily so that the pure model classes can be used
    without pytorch_lightning installed.
    """
    cls = _make_rbp_lightning_module()
    return cls(model_cfg=model_cfg, train_cfg=train_cfg)


def get_rbp_model(
    model_cfg: Optional[RBPModelConfig] = None,
) -> PythiaRBP:
    """Convenience factory for PythiaRBP."""
    return PythiaRBP(model_cfg)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable, non-trainable) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


__all__: List[str] = [
    "ConvArm",
    "PythiaRBP",
    "PythiaRBPModule",
    "get_rbp_model",
    "count_parameters",
]
