"""Train PythiaRBP for binary RNA-binding protein binding prediction.

Uses PyTorch Lightning for training.

Example usage:
    python train_rbp.py \\
        --train-tsv /data/ELAVL1_trainingSet.tsv.gz \\
        --val-tsv   /data/ELAVL1_tuningSet.tsv.gz \\
        --test-tsv  /data/ELAVL1_validationSet.tsv.gz \\
        --output-dir /results/ELAVL1 \\
        --inputsize 256 --max-epochs 50 --batch-size 64

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from pythia.configs import RBPModelConfig, RBPTrainConfig
from pythia.data.rbp_dataset import RBPDataModule
from pythia.models.rbp import PythiaRBPModule


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PythiaRBP for RBP binding prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data arguments
    parser.add_argument(
        "--train-tsv", type=Path, required=True, help="Training TSV (gzip OK)."
    )
    parser.add_argument(
        "--val-tsv", type=Path, required=True, help="Validation / tuning TSV."
    )
    parser.add_argument(
        "--test-tsv", type=Path, default=None, help="Optional test TSV."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")

    # Model architecture
    parser.add_argument("--inputsize", type=int, default=256, help="Sequence length.")
    parser.add_argument("--init-channels", type=int, default=64, help="Arm1 conv width.")
    parser.add_argument("--kernel-size", type=int, default=16, help="Arm1 first kernel.")
    parser.add_argument("--dp", type=float, default=0.25, help="Dropout rate.")
    parser.add_argument("--dil-start", type=int, default=5, help="Min dilation.")
    parser.add_argument("--dil-end", type=int, default=24, help="Max dilation.")
    parser.add_argument("--bulge-size", type=int, default=2, help="FD bulge size.")
    parser.add_argument(
        "--binarize-fd",
        action="store_true",
        default=False,
        help="Binarize FixedDilatedConv outputs.",
    )

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=0.2,
        help=(
            "Weight for the negative (Unbound) class in CrossEntropyLoss. "
            "The positive class always has weight 1.0, so 0.2 gives the "
            "positive class 5× more influence (suitable for ~10:1 imbalance)."
        ),
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Lightning precision: '32', '16-mixed', 'bf16-mixed'.",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs.")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator (auto, gpu, cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = RBPModelConfig(
        inputsize=args.inputsize,
        init_channels=args.init_channels,
        kernel_size=args.kernel_size,
        dp=args.dp,
        dil_start=args.dil_start,
        dil_end=args.dil_end,
        bulge_size=args.bulge_size,
        binarize_fd=args.binarize_fd,
    )

    train_cfg = RBPTrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        pos_weight=args.pos_weight,
    )

    # Save configs
    cfg_path = args.output_dir / "configs.json"
    with open(cfg_path, "w") as fh:
        json.dump(
            {
                "model": model_cfg.model_dump(),
                "train": train_cfg.model_dump(),
            },
            fh,
            indent=2,
        )
    print(f"Configs saved to {cfg_path}")

    # Data
    dm = RBPDataModule(
        train_tsv=args.train_tsv,
        val_tsv=args.val_tsv,
        test_tsv=args.test_tsv,
        max_len=args.inputsize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    module = PythiaRBPModule(model_cfg=model_cfg, train_cfg=train_cfg)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="rbp_best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )
    logger = CSVLogger(save_dir=str(args.output_dir), name="lightning_logs")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(module, datamodule=dm)

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")

    # Test if test set provided
    if args.test_tsv is not None:
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()
