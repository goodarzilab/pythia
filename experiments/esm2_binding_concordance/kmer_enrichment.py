"""Dense k-mer enrichment profiles of ecCLIP peaks, one row per RBP.

For each split, a k-mer's count is the number of windows containing it at least
once. Counts are Laplace-smoothed (+1) over the full 4^k vocabulary and the
enrichment is log(f_peak / f_background), with f = count / total count. The
profile of an RBP is the mean over its splits; the Pearson correlation between
splits is reported as a reliability check.

Usage
-----
    python kmer_enrichment.py --rbps data/rbps.txt --windows work/windows \\
        --out data/kmer_enrichment_L20_k5.tsv.gz
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def presence_counts(path: Path, k: int, index: dict[str, int]) -> np.ndarray:
    counts = np.zeros(len(index), dtype=np.int64)
    for line in open(path):
        if line.startswith(">"):
            continue
        seq = line.strip().upper().replace("U", "T")
        seen = {index[seq[i : i + k]] for i in range(len(seq) - k + 1)
                if seq[i : i + k] in index}
        counts[list(seen)] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rbps", default="data/rbps.txt")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", default="data/kmer_enrichment_L20_k5.tsv.gz")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    vocab = ["".join(p) for p in itertools.product("ACGT", repeat=args.k)]
    index = {kmer: j for j, kmer in enumerate(vocab)}
    profiles, reliability = {}, {}
    for rbp in Path(args.rbps).read_text().split():
        splits = []
        for peaks in sorted((Path(args.windows) / rbp).glob("split*_peaks.fa")):
            pc = presence_counts(peaks, args.k, index) + 1
            bc = presence_counts(
                peaks.with_name(peaks.name.replace("_peaks", "_background")),
                args.k, index,
            ) + 1
            splits.append(np.log((pc / pc.sum()) / (bc / bc.sum())))
        splits = np.vstack(splits)
        profiles[rbp] = splits.mean(0)
        if len(splits) == 2:
            reliability[rbp] = np.corrcoef(splits)[0, 1]

    out = pd.DataFrame.from_dict(profiles, orient="index", columns=vocab)
    out.index.name = "rbp"
    out.to_csv(args.out, sep="\t")
    if reliability:
        r = pd.Series(reliability)
        print(f"split-half Pearson r: median {r.median():.3f}, "
              f"min {r.min():.3f} ({r.idxmin()})")


if __name__ == "__main__":
    main()
