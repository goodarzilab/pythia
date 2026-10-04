"""CSV-based datasets for BEACON structural tasks: SSP, CMP, DMP, SSI.

CSV formats
-----------
SSP / CMP / DMP (bpRNA-style):
    Columns: data_name (TR0/VL0/TS0), file_name, seq, dot_string

    For SSP: dot_string converted to binary L×L contact matrix.
    For CMP: same as SSP but only |i-j| >= 23 contacts are used.
    For DMP: requires separate distance .npy files or inline distance column.

SSI:
    Columns: sequence, struct (space-sep floats, -1 for missing), struct_true

Maintainer: Mehran Karimzadeh <mehran.karimzade@gmail.com>
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_RNA_MAP: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


def _one_hot_rna(seq: str, length: int) -> np.ndarray:
    arr = np.zeros((4, length), dtype=np.float32)
    seq = seq.upper().replace("T", "U")[:length]
    for i, ch in enumerate(seq):
        idx = _RNA_MAP.get(ch)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def dot_bracket_to_matrix(db: str) -> np.ndarray:
    """Convert dot-bracket string to symmetric binary contact matrix.

    Supports nested brackets: (), [], {}, <>.

    Parameters
    ----------
    db:
        Dot-bracket string.

    Returns
    -------
    np.ndarray
        Float32 matrix of shape (len(db), len(db)).
    """
    n = len(db)
    mat = np.zeros((n, n), dtype=np.float32)
    stacks: Dict[str, List[int]] = {"(": [], "[": [], "{": [], "<": []}
    close_to_open: Dict[str, str] = {")": "(", "]": "[", "}": "{", ">": "<"}

    for i, ch in enumerate(db):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in close_to_open:
            opener = close_to_open[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                mat[i, j] = 1.0
                mat[j, i] = 1.0
    return mat


def _parse_split(data_name: str) -> str:
    """Map bpRNA split codes to canonical names."""
    mapping = {"TR0": "train", "VL0": "val", "TS0": "test"}
    return mapping.get(str(data_name).strip(), str(data_name).strip())


# ---------------------------------------------------------------------------
# SSP Dataset
# ---------------------------------------------------------------------------


class SSPDataset(Dataset):
    """Secondary structure prediction dataset.

    Reads a CSV with columns: data_name, file_name, seq, dot_string.
    Labels are L×L binary contact matrices derived from dot_string.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    split:
        One of 'train', 'val', 'test' (or TR0/VL0/TS0).
        If None, all rows are used.
    max_len:
        Maximum sequence length.
    seq_col:
        Column name for sequences.
    struct_col:
        Column name for dot-bracket structures.
    split_col:
        Column name for split labels.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        split: Optional[str] = None,
        max_len: int = 512,
        seq_col: str = "seq",
        struct_col: str = "dot_string",
        split_col: str = "data_name",
    ) -> None:
        df = pd.read_csv(csv_path)

        if split is not None:
            # normalise split codes
            canonical = _parse_split(split)
            df["_split_norm"] = df[split_col].apply(_parse_split)
            df = df[df["_split_norm"] == canonical].reset_index(drop=True)

        self.max_len = max_len
        self.seqs: List[str] = df[seq_col].astype(str).tolist()
        self.structs: List[str] = df[struct_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.seqs[idx].upper().replace("T", "U")
        db = self.structs[idx]
        L = min(len(seq), len(db), self.max_len)
        seq = seq[:L]
        db = db[:L]
        mat = dot_bracket_to_matrix(db)
        return {"seq": seq, "struct": mat, "length": L}


# ---------------------------------------------------------------------------
# CMP Dataset
# ---------------------------------------------------------------------------


class CMPDataset(Dataset):
    """Contact map prediction dataset (long-range contacts only).

    Same CSV format as SSP. The trainer applies a |i-j| >= min_separation
    mask during loss computation.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    split:
        Split selector.
    max_len:
        Maximum sequence length.
    min_separation:
        Minimum |i-j| for a contact to be counted (default 23).
    seq_col:
        Column name for sequences.
    struct_col:
        Column name for dot-bracket structures.
    split_col:
        Column name for split labels.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        split: Optional[str] = None,
        max_len: int = 512,
        min_separation: int = 23,
        seq_col: str = "seq",
        struct_col: str = "dot_string",
        split_col: str = "data_name",
    ) -> None:
        df = pd.read_csv(csv_path)
        if split is not None:
            canonical = _parse_split(split)
            df["_split_norm"] = df[split_col].apply(_parse_split)
            df = df[df["_split_norm"] == canonical].reset_index(drop=True)

        self.max_len = max_len
        self.min_separation = min_separation
        self.seqs: List[str] = df[seq_col].astype(str).tolist()
        self.structs: List[str] = df[struct_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.seqs[idx].upper().replace("T", "U")
        db = self.structs[idx]
        L = min(len(seq), len(db), self.max_len)
        seq = seq[:L]
        db = db[:L]
        mat = dot_bracket_to_matrix(db)
        # Apply min_separation mask: set short-range contacts to -1 (ignored)
        idx = np.arange(L)
        mask = np.abs(idx[:, None] - idx[None, :]) >= self.min_separation
        # Label: 1 where contact AND long-range, else -1 (ignored in loss)
        label = np.where(mask, mat[:L, :L], -1.0).astype(np.float32)
        return {"seq": seq, "struct": label, "length": L}


# ---------------------------------------------------------------------------
# CMP Dataset (BEACON ContactMap format — separate CSVs + pre-computed .npy)
# ---------------------------------------------------------------------------


class CMPNpyDataset(Dataset):
    """BEACON ContactMap dataset loading pre-computed .npy contact maps.

    CSV columns: ``id`` (used to locate the .npy file) and ``input`` (sequence).
    The contact map is returned as-is; no min-separation masking is applied
    here — training includes all non-padded pairs in the loss.

    Parameters
    ----------
    csv_path:
        Path to the CSV file (train.csv / val.csv / RFAM19.csv …).
    contact_map_dir:
        Directory containing ``{id}.npy`` contact maps.
    max_len:
        Sequences longer than this are truncated.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        contact_map_dir: Union[str, Path],
        max_len: int = 1024,
    ) -> None:
        df = pd.read_csv(csv_path)
        missing = {"id", "input"} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Got: {list(df.columns)}"
            )
        self.rows: List[Dict] = df[["id", "input"]].to_dict(orient="records")
        self.contact_map_dir = Path(contact_map_dir)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        rid = str(r["id"]).strip()
        seq = str(r["input"]).strip().upper().replace("T", "U")

        npy_path = self.contact_map_dir / f"{rid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Contact map not found: {npy_path}")

        struct = np.load(str(npy_path)).astype(np.float32)

        # Align seq and struct dims
        L = len(seq)
        if struct.shape[0] != L or struct.shape[1] != L:
            L = min(L, struct.shape[0], struct.shape[1])
            seq = seq[:L]
            struct = struct[:L, :L]

        # Enforce max_len
        L = min(L, self.max_len)
        seq = seq[:L]
        struct = struct[:L, :L]

        return {"seq": seq, "struct": struct, "length": L}


# ---------------------------------------------------------------------------
# DMP Dataset
# ---------------------------------------------------------------------------


class DMPDataset(Dataset):
    """Distance map prediction dataset.

    CSV format: data_name, file_name, seq, dot_string  OR  id, input.
    Distance maps are loaded from numpy .npy files.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    distance_dir:
        Directory containing {id}.npy distance maps.
    split:
        Split selector.
    max_len:
        Maximum sequence length.
    max_distance:
        Distances are clipped and normalised by this value.
    id_col:
        Column for sample identifiers (used to find .npy files).
    seq_col:
        Column for RNA sequences.
    split_col:
        Column for split labels.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        distance_dir: Union[str, Path],
        split: Optional[str] = None,
        max_len: int = 512,
        max_distance: float = 20.0,
        id_col: str = "file_name",
        seq_col: str = "seq",
        split_col: str = "data_name",
    ) -> None:
        df = pd.read_csv(csv_path)
        if split is not None:
            canonical = _parse_split(split)
            df["_split_norm"] = df[split_col].apply(_parse_split)
            df = df[df["_split_norm"] == canonical].reset_index(drop=True)

        self.distance_dir = Path(distance_dir)
        self.max_len = max_len
        self.max_distance = max_distance
        self.ids: List[str] = df[id_col].astype(str).tolist()
        self.seqs: List[str] = df[seq_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.seqs[idx].upper().replace("T", "U")
        rid = self.ids[idx]
        npy_path = self.distance_dir / f"{rid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Distance map not found: {npy_path}")

        dist = np.load(str(npy_path)).astype(np.float32)
        L = min(len(seq), dist.shape[0], dist.shape[1], self.max_len)
        seq = seq[:L]
        dist = dist[:L, :L]
        dist = np.nan_to_num(dist, nan=self.max_distance)
        dist = np.clip(dist, 0.0, self.max_distance)
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)
        dist = (dist / self.max_distance).astype(np.float32)
        return {"seq": seq, "struct": dist, "length": L}


# ---------------------------------------------------------------------------
# DMP Dataset (BEACON DistanceMap format — separate CSVs + pre-computed .npy)
# ---------------------------------------------------------------------------


class DMPNpyDataset(Dataset):
    """BEACON DistanceMap dataset loading pre-computed .npy distance maps.

    CSV columns: ``id`` (used to locate the .npy file) and ``input`` (sequence).
    The distance map is returned as-is — BEACON .npy files are already in
    [0, 1] and require no normalization.  Only the -1 padding sentinel is
    added by the collator.

    Parameters
    ----------
    csv_path:
        Path to the CSV file (train.csv / val.csv / RFAM19.csv …).
    distance_map_dir:
        Directory containing ``{id}.npy`` distance maps.
    max_len:
        Sequences longer than this are truncated.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        distance_map_dir: Union[str, Path],
        max_len: int = 1024,
    ) -> None:
        df = pd.read_csv(csv_path)
        missing = {"id", "input"} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Got: {list(df.columns)}"
            )
        self.rows: List[Dict] = df[["id", "input"]].to_dict(orient="records")
        self.distance_map_dir = Path(distance_map_dir)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        rid = str(r["id"]).strip()
        seq = str(r["input"]).strip().upper().replace("T", "U")

        npy_path = self.distance_map_dir / f"{rid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Distance map not found: {npy_path}")

        struct = np.load(str(npy_path)).astype(np.float32)

        # Align seq and struct dims
        L = len(seq)
        if struct.shape[0] != L or struct.shape[1] != L:
            L = min(L, struct.shape[0], struct.shape[1])
            seq = seq[:L]
            struct = struct[:L, :L]

        # Enforce max_len
        L = min(L, self.max_len)
        seq = seq[:L]
        struct = struct[:L, :L]

        return {"seq": seq, "struct": struct, "length": L}


# ---------------------------------------------------------------------------
# SSI Dataset
# ---------------------------------------------------------------------------


class SSIDataset(Dataset):
    """Structural score imputation dataset.

    CSV columns: sequence, struct (space-separated floats, -1 for missing),
    struct_true (space-separated floats).

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    max_len:
        Maximum sequence length.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        max_len: int = 440,
    ) -> None:
        df = pd.read_csv(csv_path, usecols=["sequence", "struct", "struct_true"])
        df["sequence"] = (
            df["sequence"].astype(str).str.upper().str.replace("T", "U", regex=False)
        )
        self.max_len = max_len
        self.seqs: List[str] = df["sequence"].tolist()
        self.struct_obs: List[np.ndarray] = [
            np.array([float(v) for v in str(x).split()], dtype=np.float32)
            for x in df["struct"].tolist()
        ]
        self.struct_gt: List[np.ndarray] = [
            np.array([float(v) for v in str(x).split()], dtype=np.float32)
            for x in df["struct_true"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.seqs[idx]
        obs = self.struct_obs[idx]
        gt = self.struct_gt[idx]
        L = min(len(seq), len(obs), len(gt), self.max_len)
        return {
            "seq": seq[:L],
            "obs": obs[:L],
            "tru": gt[:L],
            "length": L,
        }


# ---------------------------------------------------------------------------
# Shared collator for BEACON tasks
# ---------------------------------------------------------------------------


class BeaconCollator:
    """Collate variable-length BEACON samples into padded tensors.

    Works for SSP, CMP, and DMP tasks (2-D struct labels).
    For SSI, use SSICollator instead.

    Parameters
    ----------
    max_len:
        Maximum sequence length (sequences/labels are padded to this).
    task:
        One of 'ssp', 'cmp', 'dmp'.
    """

    def __init__(self, max_len: int = 512, task: str = "ssp") -> None:
        self.max_len = max_len
        self.task = task

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        lengths = [b["length"] for b in batch]
        l_max = min(max(lengths), self.max_len)
        b_size = len(batch)

        onehot = np.zeros((b_size, 4, l_max), dtype=np.float32)
        labels = np.full((b_size, l_max, l_max), fill_value=-1.0, dtype=np.float32)

        for i, item in enumerate(batch):
            seq = item["seq"]
            L = min(item["length"], l_max)
            onehot[i] = _one_hot_rna(seq, l_max)
            mat = item["struct"]
            ml = min(mat.shape[0], L)
            labels[i, :ml, :ml] = mat[:ml, :ml]

        return {
            "onehot": torch.from_numpy(onehot),
            "labels": torch.from_numpy(labels),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }


class SSICollator:
    """Collate SSIDataset samples into padded tensors.

    In addition to the imputation mask, produces the observed-score and
    known-mask channels PythiaSSI conditions on: ``obs_normed`` (observed
    scores, zero at masked positions) and ``obs_known`` (boolean mask of
    which positions were actually observed).

    Parameters
    ----------
    max_len:
        Maximum sequence length.
    """

    def __init__(self, max_len: int = 440) -> None:
        self.max_len = max_len

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        b_size = len(batch)
        lengths = [b["length"] for b in batch]
        l_max = min(max(lengths), self.max_len)

        onehot = np.zeros((b_size, 4, l_max), dtype=np.float32)
        y_true = np.full((b_size, l_max), -1.0, dtype=np.float32)
        mask_impute = np.zeros((b_size, l_max), dtype=np.bool_)
        obs_normed = np.zeros((b_size, l_max), dtype=np.float32)
        obs_known = np.zeros((b_size, l_max), dtype=np.bool_)

        for i, item in enumerate(batch):
            seq = item["seq"]
            L = min(item["length"], l_max)
            onehot[i] = _one_hot_rna(seq, l_max)
            y_true[i, :L] = item["tru"][:L]
            obs = item["obs"][:L]
            known = obs != -1.0
            mask_impute[i, :L] = ~known
            obs_normed[i, :L] = np.where(known, obs, 0.0)
            obs_known[i, :L] = known

        return {
            "onehot": torch.from_numpy(onehot),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "y_true": torch.from_numpy(y_true),
            "mask_impute": torch.from_numpy(mask_impute),
            "obs_normed": torch.from_numpy(obs_normed),
            "obs_known": torch.from_numpy(obs_known),
        }


def build_beacon_dataloader(
    dataset: Dataset,
    task: str,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 4,
    max_len: int = 512,
) -> DataLoader:
    """Build a DataLoader with the appropriate collator.

    Parameters
    ----------
    dataset:
        One of SSPDataset, CMPDataset, DMPDataset, or SSIDataset.
    task:
        One of 'ssp', 'cmp', 'dmp', 'ssi'.
    batch_size:
        Batch size.
    shuffle:
        Whether to shuffle.
    num_workers:
        DataLoader worker count.
    max_len:
        Maximum sequence length for padding.

    Returns
    -------
    DataLoader
    """
    if task == "ssi":
        collator: Any = SSICollator(max_len=max_len)
    else:
        collator = BeaconCollator(max_len=max_len, task=task)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


__all__: List[str] = [
    "SSPDataset",
    "CMPDataset",
    "CMPNpyDataset",
    "DMPDataset",
    "DMPNpyDataset",
    "SSIDataset",
    "BeaconCollator",
    "SSICollator",
    "dot_bracket_to_matrix",
    "build_beacon_dataloader",
]
