"""Extract peak and flanking-background windows around ecCLIP CIMS sites.

For each RBP, peaks are shuffled (fixed seed), capped at --cap, and split into
--n-splits disjoint halves. For every peak an L-nt window centred on the CIMS site
is written, reverse-complemented on the minus strand, together with the two
adjacent L-nt windows as background. Background windows that leave the chromosome
or overlap any peak window of the same RBP (from the full, uncapped peak set) are
dropped.

Output: <outdir>/<RBP>/split<i>_peaks.fa and split<i>_background.fa

Usage
-----
    python extract_peak_windows.py --rbps data/rbps.txt \\
        --bed-dir <bedFiles> --genome hg38.fa --outdir work/windows
"""

from __future__ import annotations

import argparse
import bisect
import random
from collections import defaultdict
from pathlib import Path

import pysam

BED_PATTERN = "{rbp}_clearCLIP.pool.tag.uniq.del.CIMS.fdr10.bgfilter.bed"
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def read_sites(path: Path) -> list[tuple[str, int, str, str]]:
    sites = []
    for line in open(path):
        tok = line.rstrip("\n").split("\t")
        if len(tok) < 6:
            continue
        try:
            sites.append((tok[0], int(tok[1]), tok[3], tok[5]))
        except ValueError:
            continue
    return sites


def window(pos: int, L: int) -> tuple[int, int]:
    return pos - L // 2, pos + (L - L // 2)


def peak_index(sites, L: int) -> dict[str, list[tuple[int, int]]]:
    index = defaultdict(list)
    for chrom, pos, _, _ in sites:
        start, end = window(pos, L)
        if start >= 0:
            index[chrom].append((start, end))
    for chrom in index:
        index[chrom].sort()
    return index


def overlaps(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    i = bisect.bisect_left(intervals, (end,)) - 1
    while i >= 0 and intervals[i][1] > start:
        if intervals[i][0] < end:
            return True
        i -= 1
    return False


def write_windows(sites, index, fasta, sizes, L, peaks_out, background_out):
    n_peaks = n_background = 0
    with open(peaks_out, "w") as fp, open(background_out, "w") as fb:
        for chrom, pos, name, strand in sites:
            if chrom not in sizes:
                continue
            start, end = window(pos, L)
            if start < 0 or end > sizes[chrom]:
                continue
            seq = fasta.fetch(chrom, start, end).upper()
            fp.write(f">{name}\n{revcomp(seq) if strand == '-' else seq}\n")
            n_peaks += 1
            for side, s, e in (("u", start - L, start), ("d", end, end + L)):
                if s < 0 or e > sizes[chrom] or overlaps(s, e, index.get(chrom, [])):
                    continue
                bg = fasta.fetch(chrom, s, e).upper()
                fb.write(f">{name}_{side}\n{revcomp(bg) if strand == '-' else bg}\n")
                n_background += 1
    return n_peaks, n_background


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rbps", default="data/rbps.txt")
    ap.add_argument("--bed-dir", required=True)
    ap.add_argument("--bed-pattern", default=BED_PATTERN)
    ap.add_argument("--genome", required=True, help="indexed hg38 FASTA")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--cap", type=int, default=100_000)
    ap.add_argument("--n-splits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fasta = pysam.FastaFile(args.genome)
    sizes = dict(zip(fasta.references, fasta.lengths))
    for rbp in Path(args.rbps).read_text().split():
        sites = read_sites(Path(args.bed_dir) / args.bed_pattern.format(rbp=rbp))
        index = peak_index(sites, args.L)
        pool = list(sites)
        random.Random(args.seed).shuffle(pool)
        pool = pool[: args.cap]
        size = len(pool) // args.n_splits
        outdir = Path(args.outdir) / rbp
        outdir.mkdir(parents=True, exist_ok=True)
        counts = []
        for i in range(args.n_splits):
            counts.append(
                write_windows(
                    pool[i * size : (i + 1) * size], index, fasta, sizes, args.L,
                    outdir / f"split{i}_peaks.fa", outdir / f"split{i}_background.fa",
                )
            )
        print(rbp, len(sites), *counts, flush=True)


if __name__ == "__main__":
    main()
