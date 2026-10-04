"""Generate a synthetic sequence-motif benchmark dataset (SEQMOTIF).

Uses a fixed sequence motif (AUCGGCUAGACGCAC) to produce a labelled dataset:

- **Positive sequences**: the motif is injected at a random position.  In
  10 % of positives one nucleotide of the injected motif is randomly
  substituted to model real-world imperfect binding sites.

- **Negative sequences**: generated via a first-order Markov chain whose
  transition matrix is estimated from the positive sequences, preserving
  the same dinucleotide composition without containing the motif.

The output is compatible with ``pseudoknot_joblib_to_tsv.py`` for
splitting into train/tune/validation TSV files:

    {outdir}/sequence_motif_trainingSet.tsv.gz   (64 %)
    {outdir}/sequence_motif_tuningSet.tsv.gz     (16 %)
    {outdir}/sequence_motif_validationSet.tsv.gz (20 %)

Usage
-----
    python generate_sequence_motif_dataset.py \\
        --joblib-out output/sequence_motif.joblib \\
        --splits-outdir output/dataSplits/sequence_motif \\
        --num-examples 100000
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_MOTIF = "AUCGGCUAGACGCAC"
NOISE_FRACTION = 0.10  # fraction of positives with one nucleotide mutated

_NUCS = np.array(["A", "C", "G", "U"])
_NUC_IDX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}

DEFAULT_JOBLIB_OUT = Path("output/sequence_motif.joblib")
DEFAULT_SPLITS_OUTDIR = Path("output/dataSplits/sequence_motif")
DEFAULT_NUM_EXAMPLES = 100_000
DEFAULT_SEQ_LENGTH = 256
DEFAULT_IMBALANCE = 0.10
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def generate_random_backgrounds(
    n: int, seq_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate *n* uniformly random RNA sequences of *seq_length*."""
    mat = rng.choice(_NUCS, size=(n, seq_length))
    return np.array(["".join(row) for row in mat])


def compute_transition_matrix(sequences: np.ndarray) -> np.ndarray:
    """Estimate a 1st-order Markov transition matrix from *sequences*.

    Returns a (4, 4) float64 array where ``T[i, j]`` is the probability
    of nucleotide *j* following nucleotide *i* (rows normalised to sum to 1).
    Rows with zero observations fall back to a uniform distribution.
    """
    counts = np.zeros((4, 4), dtype=np.float64)
    for seq in sequences:
        for k in range(len(seq) - 1):
            a, b = seq[k], seq[k + 1]
            if a in _NUC_IDX and b in _NUC_IDX:
                counts[_NUC_IDX[a], _NUC_IDX[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    # Fall back to uniform where a row has no counts
    uniform_rows = (row_sums == 0).flatten()
    row_sums[uniform_rows] = 1.0
    counts[uniform_rows] = 0.25
    return counts / row_sums


def generate_markov_sequences(
    transition_matrix: np.ndarray,
    n: int,
    seq_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate *n* RNA sequences of *seq_length* via a 1st-order Markov chain.

    The first nucleotide of each sequence is sampled uniformly.  Subsequent
    nucleotides are drawn according to *transition_matrix*.
    """
    _NUCS_LIST = ["A", "C", "G", "U"]
    seqs = []
    for _ in range(n):
        first = str(rng.choice(_NUCS_LIST))
        chars = [first]
        for _ in range(seq_length - 1):
            last_idx = _NUC_IDX[chars[-1]]
            probs = transition_matrix[last_idx]
            chars.append(str(rng.choice(_NUCS_LIST, p=probs)))
        seqs.append("".join(chars))
    return np.array(seqs, dtype=f"|U{seq_length}")


def inject_motif(
    background: str,
    motif: str,
    insert_pos: int,
    noise: bool,
    rng: np.random.Generator,
) -> Tuple[str, str]:
    """Inject *motif* into *background* at *insert_pos*.

    If *noise* is True, one randomly chosen nucleotide in the motif is
    replaced by a uniformly sampled nucleotide (may be the same).

    Returns
    -------
    (modified_sequence, injected_motif_string)
    """
    motif_chars = list(motif)
    if noise:
        mut_pos = int(rng.integers(len(motif_chars)))
        motif_chars[mut_pos] = str(rng.choice(_NUCS))
    noisy_motif = "".join(motif_chars)
    seq = background[:insert_pos] + noisy_motif + background[insert_pos + len(motif):]
    return seq, noisy_motif


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    motif: str = FIXED_MOTIF,
    noise_fraction: float = NOISE_FRACTION,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    imbalance: float = DEFAULT_IMBALANCE,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Generate a labelled sequence-motif benchmark dataset.

    Parameters
    ----------
    motif:
        Fixed RNA sequence motif to inject into positive examples.
    noise_fraction:
        Fraction of positive examples in which one nucleotide of the
        injected motif is randomly substituted.
    num_examples:
        Total number of sequences to generate.
    seq_length:
        Length of each generated sequence (must be > ``len(motif) + 2``).
    imbalance:
        Fraction of sequences that are positive (motif-containing).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Keys: ``Input`` (str), ``Response`` (bool), ``Position`` (int),
        ``motif_seq`` (str).
    """
    rng = np.random.default_rng(seed)
    motif_len = len(motif)
    n_positive = int(num_examples * imbalance)
    n_negative = num_examples - n_positive

    log.info(
        "Generating %d sequences (motif=%s, motif_len=%d, "
        "positive=%d, negative=%d, noise_fraction=%.2f)",
        num_examples,
        motif,
        motif_len,
        n_positive,
        n_negative,
        noise_fraction,
    )

    # ------------------------------------------------------------------
    # 1. Generate positive sequences
    # ------------------------------------------------------------------
    log.info("Sampling positive background sequences ...")
    pos_backgrounds = generate_random_backgrounds(n_positive, seq_length, rng)
    pos_insert = rng.integers(1, seq_length - motif_len, size=n_positive)
    noise_mask = rng.random(n_positive) < noise_fraction

    pos_seqs = []
    pos_motif_seqs = []
    n_noisy = 0
    for k in range(n_positive):
        seq, injected = inject_motif(
            pos_backgrounds[k], motif, int(pos_insert[k]), bool(noise_mask[k]), rng
        )
        pos_seqs.append(seq)
        pos_motif_seqs.append(injected)
        if noise_mask[k]:
            n_noisy += 1

    log.info(
        "Injected motif into %d positives (%d with noise, %d exact)",
        n_positive,
        n_noisy,
        n_positive - n_noisy,
    )

    # ------------------------------------------------------------------
    # 2. Estimate dinucleotide transition matrix from positive sequences
    # ------------------------------------------------------------------
    log.info("Estimating dinucleotide transition matrix from positive sequences ...")
    transition_matrix = compute_transition_matrix(np.array(pos_seqs))
    log.info(
        "Transition matrix (rows=from, cols=to, ACGU):\n%s",
        np.array2string(transition_matrix, precision=3, suppress_small=True),
    )

    # ------------------------------------------------------------------
    # 3. Generate negative sequences via Markov chain
    # ------------------------------------------------------------------
    log.info("Generating %d negative sequences via Markov chain ...", n_negative)
    neg_seqs = generate_markov_sequences(transition_matrix, n_negative, seq_length, rng)

    # ------------------------------------------------------------------
    # 4. Assemble final arrays (positives first, then negatives; shuffled later
    #    by pseudoknot_joblib_to_tsv.py)
    # ------------------------------------------------------------------
    all_seqs = np.array(pos_seqs + list(neg_seqs), dtype=f"|U{seq_length}")
    labels = np.zeros(num_examples, dtype=bool)
    labels[:n_positive] = True
    positions = np.zeros(num_examples, dtype=np.int64)
    positions[:n_positive] = pos_insert
    motif_seqs_arr = np.zeros(num_examples, dtype="|U64")
    motif_seqs_arr[:n_positive] = pos_motif_seqs

    return {
        "Input": all_seqs,
        "Response": labels,
        "Position": positions,
        "motif_seq": motif_seqs_arr,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic SEQMOTIF benchmark joblib and optional "
            "train/tune/validation TSV splits using a fixed RNA sequence motif."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--motif",
        type=str,
        default=FIXED_MOTIF,
        help="Fixed RNA sequence motif to inject into positive examples.",
    )
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=NOISE_FRACTION,
        help="Fraction of positives with one nucleotide randomly substituted.",
    )
    parser.add_argument(
        "--joblib-out",
        type=Path,
        default=DEFAULT_JOBLIB_OUT,
        help="Output path for the generated joblib dataset.",
    )
    parser.add_argument(
        "--splits-outdir",
        type=Path,
        default=DEFAULT_SPLITS_OUTDIR,
        help=(
            "If given, run pseudoknot_joblib_to_tsv.py to write "
            "train/tune/validation TSV splits here."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="Total number of sequences to generate.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=DEFAULT_SEQ_LENGTH,
        help="Length of each generated sequence.",
    )
    parser.add_argument(
        "--imbalance",
        type=float,
        default=DEFAULT_IMBALANCE,
        help="Fraction of positive (motif-containing) sequences.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="sequence_motif",
        help="Dataset name prefix for TSV filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check output path
    args.joblib_out.parent.mkdir(parents=True, exist_ok=True)
    if args.joblib_out.exists() and not args.overwrite:
        raise ValueError(
            f"Output already exists (use --overwrite to replace): {args.joblib_out}"
        )

    # 1. Generate dataset
    dataset = generate_dataset(
        motif=args.motif,
        noise_fraction=args.noise_fraction,
        num_examples=args.num_examples,
        seq_length=args.seq_length,
        imbalance=args.imbalance,
        seed=args.seed,
    )

    # 2. Save joblib
    log.info("Saving dataset to %s ...", args.joblib_out)
    joblib.dump(dataset, args.joblib_out, compress=9)
    n = len(dataset["Input"])
    n_pos = int(dataset["Response"].sum())
    log.info(
        "Saved %d sequences (bound=%d, unbound=%d)",
        n,
        n_pos,
        n - n_pos,
    )

    # 3. Optionally run the TSV splitter
    if args.splits_outdir is not None:
        splitter = Path(__file__).parent / "pseudoknot_joblib_to_tsv.py"
        cmd = [
            sys.executable,
            str(splitter),
            "--joblib",
            str(args.joblib_out),
            "--outdir",
            str(args.splits_outdir),
            "--name",
            args.name,
            "--seed",
            str(args.seed),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        log.info("Running TSV splitter: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        log.info("TSV splits written to %s", args.splits_outdir)

    log.info("Done.")


if __name__ == "__main__":
    main()
