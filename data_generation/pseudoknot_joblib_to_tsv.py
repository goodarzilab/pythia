"""Convert a pseudoknot joblib dataset to train/tune/validation TSV splits.

Reads a joblib file with keys Input, Response (bool), Position, motif_seq
and writes three gzip-compressed TSV files compatible with the Pythia
RBP dataset format:

    {outdir}/pseudoknot_trainingSet.tsv.gz   (64 %)
    {outdir}/pseudoknot_tuningSet.tsv.gz     (16 %)
    {outdir}/pseudoknot_validationSet.tsv.gz (20 %)

Response booleans are converted to "Bound" / "Unbound" strings.
The random split uses seed 42 for reproducibility.

Usage
-----
    python pseudoknot_joblib_to_tsv.py \\
        --joblib output/pseudoknot.joblib \\
        --outdir output/dataSplits/pseudoknot
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SPLIT_RATIOS: List[float] = [0.64, 0.16, 0.20]
SPLIT_NAMES: List[str] = ["trainingSet", "tuningSet", "validationSet"]
RANDOM_SEED: int = 42


def split_joblib(
    joblibpath: Path,
    ratios: List[float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """Load a pseudoknot joblib and split into train/tune/validation DataFrames.

    Parameters
    ----------
    joblibpath:
        Path to the joblib file containing keys Input, Response, Position,
        and motif_seq.
    ratios:
        Fractional sizes for [training, tuning, validation]. Must sum to 1.
    seed:
        Random seed for reproducible shuffling.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keyed by split name ("trainingSet", "tuningSet", "validationSet"),
        each containing columns: Input, Response, Position, motif_seq.
    """
    log.info("Loading %s ...", joblibpath)
    raw = joblib.load(joblibpath)
    n = len(raw["Input"])
    log.info(
        "Loaded %d sequences  (bound=%d, unbound=%d)",
        n,
        raw["Response"].sum(),
        (~raw["Response"]).sum(),
    )

    rng = np.random.default_rng(seed)
    idxs = rng.permutation(n)

    # Compute per-split sizes; last split absorbs any rounding remainder
    sizes: List[int] = [int(r * n) for r in ratios]
    sizes[-1] = n - sum(sizes[:-1])

    # Convert bool Response -> "Bound" / "Unbound"
    response_str = np.where(raw["Response"], "Bound", "Unbound")

    splits: Dict[str, pd.DataFrame] = {}
    cursor = 0
    for name, size in zip(SPLIT_NAMES, sizes):
        sel = idxs[cursor : cursor + size]
        df = pd.DataFrame(
            {
                "Input": raw["Input"][sel],
                "Response": response_str[sel],
                "Position": raw["Position"][sel],
                "motif_seq": raw["motif_seq"][sel],
            }
        )
        splits[name] = df
        log.info(
            "  %s: %d rows  (bound=%d, unbound=%d)",
            name,
            len(df),
            (df["Response"] == "Bound").sum(),
            (df["Response"] == "Unbound").sum(),
        )
        cursor += size

    return splits


def write_splits(
    splits: Dict[str, pd.DataFrame],
    outdir: Path,
    name: str = "pseudoknot",
    overwrite: bool = False,
) -> None:
    """Write each split to a gzip-compressed TSV file.

    Parameters
    ----------
    splits:
        Output of split_joblib().
    outdir:
        Directory to write files into (created if absent).
    name:
        Dataset name prefix for filenames.
    overwrite:
        If False (default), raise ValueError when an output file already exists.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        outpath = outdir / f"{name}_{split_name}.tsv.gz"
        if outpath.exists() and not overwrite:
            raise ValueError(
                f"Output already exists (use --overwrite to replace): {outpath}"
            )
        df.to_csv(outpath, sep="\t", index=False, compression="gzip")
        log.info("Wrote %d rows -> %s", len(df), outpath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a pseudoknot joblib into train/tune/validation TSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--joblib",
        type=Path,
        default=Path("output/pseudoknot.joblib"),
        help="Path to the pseudoknot joblib file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output/dataSplits/pseudoknot"),
        help="Output directory for TSV splits.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pseudoknot",
        help="Prefix used in output filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = split_joblib(args.joblib, seed=args.seed)
    write_splits(splits, args.outdir, name=args.name, overwrite=args.overwrite)
    log.info("Done. Files written to %s", args.outdir)


if __name__ == "__main__":
    main()
