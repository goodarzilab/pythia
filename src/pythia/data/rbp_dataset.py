"""CSV/TSV-based datasets and DataModule for RBP binary classification.

Expected TSV format (gzip-compressed or plain):
    Input\\tResponse\\tSeqNames\\tMFEs
    UCACAUC...\\tUnbound\\tShuffled:chr6:...\\t0

File naming convention:
    {RBP}_trainingSet.tsv.gz
    {RBP}_tuningSet.tsv.gz
    {RBP}_validationSet.tsv.gz

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# One-hot encoding map for RNA
_RNA_MAP: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


def _one_hot(seq: str, length: int) -> np.ndarray:
    """Convert an RNA/DNA sequence to a (4, length) one-hot array."""
    arr = np.zeros((4, length), dtype=np.float32)
    seq = seq.upper()[:length]
    for i, ch in enumerate(seq):
        idx = _RNA_MAP.get(ch)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def _load_tsv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a TSV or TSV.gz file into a DataFrame."""
    path = Path(path)
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open  # type: ignore[assignment]
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RBPDataset(Dataset):
    """CSV/TSV-based RBP binary classification dataset.

    Parameters
    ----------
    tsv_path:
        Path to a (gzip-compressed) TSV file.
    max_len:
        Maximum sequence length; sequences are padded/truncated to this.
    label_col:
        Column name for the binary label (expects "Bound"/"Unbound" or 0/1).
    seq_col:
        Column name for the RNA sequence.
    """

    LABEL_MAP: Dict[str, int] = {"Bound": 1, "Unbound": 0, "1": 1, "0": 0}

    def __init__(
        self,
        tsv_path: Union[str, Path],
        max_len: int = 256,
        label_col: str = "Response",
        seq_col: str = "Input",
    ) -> None:
        self.max_len = max_len
        self.label_col = label_col
        self.seq_col = seq_col

        df = _load_tsv(tsv_path)
        if seq_col not in df.columns:
            raise KeyError(f"Column '{seq_col}' not found. Available: {df.columns.tolist()}")
        if label_col not in df.columns:
            raise KeyError(f"Column '{label_col}' not found. Available: {df.columns.tolist()}")

        self.sequences: List[str] = df[seq_col].astype(str).tolist()
        raw_labels = df[label_col].astype(str).tolist()
        self.labels: List[int] = [
            self.LABEL_MAP[lbl] if lbl in self.LABEL_MAP else int(float(lbl))
            for lbl in raw_labels
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].upper().replace("T", "U")
        onehot = torch.from_numpy(_one_hot(seq, self.max_len))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return onehot, label

    @classmethod
    def from_rbp_dir(
        cls,
        rbp_dir: Union[str, Path],
        rbp_name: str,
        split: str = "training",
        max_len: int = 256,
    ) -> "RBPDataset":
        """Construct dataset from the canonical RBP directory layout.

        Parameters
        ----------
        rbp_dir:
            Directory containing the TSV files.
        rbp_name:
            RBP name used to build the filename.
        split:
            One of 'training', 'tuning', 'validation'.
        max_len:
            Maximum sequence length.
        """
        split_map = {
            "training": "trainingSet",
            "tuning": "tuningSet",
            "validation": "validationSet",
        }
        suffix = split_map.get(split, split)
        fname = f"{rbp_name}_{suffix}.tsv.gz"
        fpath = Path(rbp_dir) / fname
        if not fpath.exists():
            fpath = fpath.with_suffix("")  # try without .gz
        return cls(fpath, max_len=max_len)


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------


try:
    import pytorch_lightning as _pl_lib
    _LightningDataModule = _pl_lib.LightningDataModule
except Exception:
    _pl_lib = None  # type: ignore[assignment]
    _LightningDataModule = object  # type: ignore[assignment, misc]


class RBPDataModule(_LightningDataModule):  # type: ignore[misc]
    """PyTorch Lightning DataModule for RBP binary classification.

    Parameters
    ----------
    train_tsv:
        Path to training TSV file.
    val_tsv:
        Path to validation (tuning) TSV file.
    test_tsv:
        Optional path to test TSV file.
    max_len:
        Maximum sequence length.
    batch_size:
        Training batch size.
    num_workers:
        DataLoader worker count.
    """

    def __init__(
        self,
        train_tsv: Union[str, Path],
        val_tsv: Union[str, Path],
        test_tsv: Optional[Union[str, Path]] = None,
        max_len: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_tsv = Path(train_tsv)
        self.val_tsv = Path(val_tsv)
        self.test_tsv = Path(test_tsv) if test_tsv else None
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_ds: Optional[RBPDataset] = None
        self._val_ds: Optional[RBPDataset] = None
        self._test_ds: Optional[RBPDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self._train_ds = RBPDataset(self.train_tsv, max_len=self.max_len)
            self._val_ds = RBPDataset(self.val_tsv, max_len=self.max_len)
        if stage in ("test", None) and self.test_tsv is not None:
            self._test_ds = RBPDataset(self.test_tsv, max_len=self.max_len)

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        ds = self._test_ds if self._test_ds is not None else self._val_ds
        assert ds is not None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


__all__: List[str] = ["RBPDataset", "RBPDataModule"]
