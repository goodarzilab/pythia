"""Generate synthetic structural-motif benchmark datasets (STRUCTMOTIF).

Two modes are supported:

``structure`` (default)
    Generates a sequence-independent structural motif benchmark.  A dot-
    bracket pattern (e.g. ``<<<<....>>>>``) defines the motif purely by
    secondary structure shape with no sequence constraint.  Positive examples
    enforce valid Watson-Crick and G-U wobble base pairs at all stem positions;
    the motif is placed at the centre of a 256 bp sequence.  All flanking
    positions and all negative sequences are generated via a first-order Markov
    chain whose transition matrix is estimated from the positive sequences,
    preserving the same dinucleotide composition.  Fully self-contained; no
    external data required.

``pyteiser``
    Reads structural motifs from a pyteiser-format binary seed file, injects
    a chosen motif (IUPAC-expanded) into background sequences, and writes
    the result as a compressed joblib file.  Requires a pyteiser seed file
    (not included in this release) -- see --binpath.

The output is compatible with ``pseudoknot_joblib_to_tsv.py`` for
splitting into train/tune/validation TSV files:

    {outdir}/structural_motif_trainingSet.tsv.gz   (64 %)
    {outdir}/structural_motif_tuningSet.tsv.gz     (16 %)
    {outdir}/structural_motif_validationSet.tsv.gz (20 %)

Usage
-----
    # structure mode (default, sequence-independent, no external data needed)
    python generate_structural_motif_dataset.py \\
        --mode structure \\
        --motif-pattern '<<<<....>>>>' \\
        --joblib-out output/structural_motif_seqindep.joblib \\
        --splits-outdir output/dataSplits/structural_motif \\
        --num-examples 100000

    # pyteiser mode (requires a private pyteiser seed file, bring your own)
    python generate_structural_motif_dataset.py \\
        --mode pyteiser \\
        --binpath /path/to/seeds_passed.bin \\
        --joblib-out output/structural_motif.joblib \\
        --splits-outdir output/dataSplits/structural_motif \\
        --motif-idx 10 \\
        --num-examples 100000
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IUPAC / binary encoding (mirrors glob_var.py from pyteiser)
# ---------------------------------------------------------------------------
_STEM = np.uint8(1)
_LOOP = np.uint8(2)

_INT_TO_IUPAC: Dict[int, str] = {
    1: "U",
    2: "C",
    3: "G",
    4: "A",
    5: "N",
    6: "Y",
    7: "R",
    8: "K",
    9: "M",
    10: "S",
    11: "W",
    12: "B",
    13: "D",
    14: "H",
    15: "V",
}

_COMP: Dict[int, int] = {
    1: 7,
    2: 3,
    3: 6,
    4: 1,
    5: 5,
    6: 7,
    7: 6,
    8: 5,
    9: 8,
    10: 12,
    11: 13,
    12: 5,
    13: 5,
    14: 13,
    15: 12,
}

_IUPAC_TO_NUCS: Dict[str, List[str]] = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "U": ["U"],
    "T": ["U"],
    "N": ["A", "C", "G", "U"],
    "Y": ["U", "C"],
    "R": ["A", "G"],
    "K": ["U", "G"],
    "M": ["A", "C"],
    "S": ["G", "C"],
    "W": ["A", "U"],
    "B": ["G", "U", "C"],
    "D": ["G", "A", "U"],
    "H": ["A", "C", "U"],
    "V": ["G", "C", "A"],
}

# ---------------------------------------------------------------------------
# Watson-Crick + G-U wobble complement map (structure mode)
# ---------------------------------------------------------------------------
_NUCS = np.array(["A", "C", "G", "U"])
_NUC_IDX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}

# For each 5'-arm nucleotide, the set of valid 3'-arm pairing partners
_WC_COMPLEMENTS: Dict[str, List[str]] = {
    "A": ["U"],
    "U": ["A", "G"],
    "G": ["C", "U"],
    "C": ["G"],
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BINPATH = Path("seeds_passed.bin")
DEFAULT_JOBLIB_OUT = Path("output/structural_motif.joblib")
DEFAULT_JOBLIB_OUT_STRUCT = Path("output/structural_motif_seqindep.joblib")
DEFAULT_SPLITS_OUTDIR = Path("output/dataSplits/structural_motif")
DEFAULT_MOTIF_IDX = 10
DEFAULT_NUM_EXAMPLES = 100_000
DEFAULT_SEQ_LENGTH = 256
DEFAULT_IMBALANCE = 0.10
DEFAULT_SEED = 42
DEFAULT_STRUCT_MOTIF = "<<<<....>>>>"


# ---------------------------------------------------------------------------
# Binary seed file parsing (pyteiser mode)
# ---------------------------------------------------------------------------

def load_motif_dict(binpath: Path) -> Dict[int, Dict]:
    """Parse a pyteiser binary seed file into a motif dictionary."""
    with open(binpath, "rb") as fh:
        bitstring = fh.read()

    motif_dict: Dict[int, Dict] = {}
    cur = 0
    mid = 0
    total = len(bitstring)

    while cur < total:
        stem_len = bitstring[cur]
        loop_len = bitstring[cur + 1]
        full_len = stem_len + loop_len

        seq_bin = np.frombuffer(
            bitstring[cur + 2 : cur + 2 + full_len], dtype=np.uint8
        )
        struct_bin = np.frombuffer(
            bitstring[cur + 2 + full_len : cur + 2 + 2 * full_len], dtype=np.uint8
        )

        stem_cnt = int(np.sum(struct_bin == _STEM))
        loop_cnt = int(np.sum(struct_bin == _LOOP))
        linear_len = 2 * stem_cnt + loop_cnt

        linear_seq = np.zeros(linear_len, dtype=np.uint8)
        li, ri = 0, linear_len - 1
        for i in range(full_len):
            nt = int(seq_bin[i])
            linear_seq[li] = nt
            if struct_bin[i] == _STEM:
                linear_seq[ri] = _COMP.get(nt, 5)
                li += 1
                ri -= 1
            else:
                li += 1

        iupac = "".join(_INT_TO_IUPAC.get(int(x), "N") for x in linear_seq)
        motif_dict[mid] = {
            "Sequence": iupac,
            "stem_length": stem_len,
            "loop_length": loop_len,
        }
        cur += 2 + 2 * full_len + 16
        mid += 1

    log.info("Parsed %d motifs from %s", mid, binpath)
    return motif_dict


# ---------------------------------------------------------------------------
# Sequence utilities (shared)
# ---------------------------------------------------------------------------

def generate_random_backgrounds(
    n: int, seq_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate *n* uniformly random RNA sequences of *seq_length*."""
    mat = rng.choice(_NUCS, size=(n, seq_length))
    return np.array(["".join(row) for row in mat])


def compute_transition_matrix(sequences: np.ndarray) -> np.ndarray:
    """Estimate a 1st-order Markov transition matrix from *sequences*.

    Returns a (4, 4) float64 array where ``T[i, j]`` is the probability of
    nucleotide *j* following nucleotide *i*.  Rows with zero observations
    fall back to a uniform distribution.
    """
    counts = np.zeros((4, 4), dtype=np.float64)
    for seq in sequences:
        for k in range(len(seq) - 1):
            a, b = seq[k], seq[k + 1]
            if a in _NUC_IDX and b in _NUC_IDX:
                counts[_NUC_IDX[a], _NUC_IDX[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
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
    """Generate *n* RNA sequences of *seq_length* via a 1st-order Markov chain."""
    nucs_list = ["A", "C", "G", "U"]
    seqs = []
    for _ in range(n):
        first = str(rng.choice(nucs_list))
        chars = [first]
        for _ in range(seq_length - 1):
            last_idx = _NUC_IDX[chars[-1]]
            probs = transition_matrix[last_idx]
            chars.append(str(rng.choice(nucs_list, p=probs)))
        seqs.append("".join(chars))
    return np.array(seqs, dtype=f"|U{seq_length}")


# ---------------------------------------------------------------------------
# Sequence utilities (pyteiser mode)
# ---------------------------------------------------------------------------

def sample_motif_seqs(
    motif_iupac: str, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample *n* concrete sequences from an IUPAC motif pattern."""
    motif_len = len(motif_iupac)
    choices_per_pos = [_IUPAC_TO_NUCS.get(c, ["N"]) for c in motif_iupac]
    out = np.zeros(n, dtype="|U64")
    for i in range(n):
        chars = [str(rng.choice(choices_per_pos[j])) for j in range(motif_len)]
        out[i] = "".join(chars)
    return out


# ---------------------------------------------------------------------------
# Dot-bracket parsing and motif generation (structure mode)
# ---------------------------------------------------------------------------

def parse_dot_bracket(pattern: str) -> Tuple[Dict[int, int], List[int]]:
    """Parse a dot-bracket pattern into base-pair indices and free positions.

    Supports ``<`` / ``>`` as stem characters and ``.`` as loop/unpaired.

    Returns
    -------
    pairs : dict
        Mapping from each paired position to its partner (bidirectional).
    free : list
        Positions with no pairing constraint (loop / unpaired).
    """
    stack: List[int] = []
    pairs: Dict[int, int] = {}
    for i, c in enumerate(pattern):
        if c == "<":
            stack.append(i)
        elif c == ">":
            if not stack:
                raise ValueError(
                    f"Unmatched '>' at position {i} in pattern '{pattern}'"
                )
            open_pos = stack.pop()
            pairs[open_pos] = i
            pairs[i] = open_pos
    if stack:
        raise ValueError(
            f"Unmatched '<' at positions {stack} in pattern '{pattern}'"
        )
    free = [i for i, c in enumerate(pattern) if c == "."]
    return pairs, free


def generate_structure_motif_instance(
    motif_len: int,
    pairs: Dict[int, int],
    free: List[int],
    rng: np.random.Generator,
) -> str:
    """Generate one concrete RNA sequence that satisfies *pairs* constraints.

    For each stem pair (i, j) with i < j:
    - Position *i* is sampled uniformly from {A, C, G, U}.
    - Position *j* is sampled from the set of valid Watson-Crick / G-U wobble
      partners of the nucleotide at *i*.

    Free positions are sampled uniformly.
    """
    motif = ["N"] * motif_len
    processed: set = set()
    nucs_list = ["A", "C", "G", "U"]

    for pos in range(motif_len):
        if pos in processed:
            continue
        if pos in pairs:
            partner = pairs[pos]
            if partner > pos:
                nuc_5 = str(rng.choice(nucs_list))
                partners_3 = _WC_COMPLEMENTS[nuc_5]
                nuc_3 = str(rng.choice(partners_3))
                motif[pos] = nuc_5
                motif[partner] = nuc_3
                processed.add(pos)
                processed.add(partner)
        else:
            motif[pos] = str(rng.choice(nucs_list))
            processed.add(pos)

    return "".join(motif)


# ---------------------------------------------------------------------------
# Dataset generation (pyteiser mode)
# ---------------------------------------------------------------------------

def generate_dataset(
    motif_dict: Dict[int, Dict],
    motif_idx: int = DEFAULT_MOTIF_IDX,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    imbalance: float = DEFAULT_IMBALANCE,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Generate synthetic sequences with injected structural motifs (pyteiser mode)."""
    rng = np.random.default_rng(seed)
    motif_iupac = motif_dict[motif_idx]["Sequence"]
    motif_len = len(motif_iupac)
    n_positive = int(num_examples * imbalance)
    n_negative = num_examples - n_positive

    log.info(
        "Generating %d sequences (motif_idx=%d, iupac=%s, motif_len=%d, "
        "positive=%d, negative=%d)",
        num_examples,
        motif_idx,
        motif_iupac,
        motif_len,
        n_positive,
        n_negative,
    )

    log.info("Sampling background sequences ...")
    backgrounds = generate_random_backgrounds(num_examples, seq_length, rng)

    log.info("Sampling motif instances ...")
    motif_seqs = sample_motif_seqs(motif_iupac, n_positive, rng)

    pos_indices = np.round(
        np.linspace(0, num_examples - 1, n_positive)
    ).astype(int)
    pos_insert = rng.integers(1, seq_length - motif_len, size=n_positive)

    labels = np.zeros(num_examples, dtype=bool)
    positions = np.zeros(num_examples, dtype=np.int64)
    motifseqs = np.zeros(num_examples, dtype="|U64")

    log.info("Injecting motifs into positive sequences ...")
    seqs = list(backgrounds)
    for k, idx in enumerate(pos_indices):
        ins = pos_insert[k]
        m = motif_seqs[k]
        s = seqs[idx]
        seqs[idx] = s[:ins] + m + s[ins + motif_len :]
        labels[idx] = True
        positions[idx] = ins
        motifseqs[idx] = m

    out_seqs = np.array(seqs, dtype=f"|U{seq_length}")

    return {
        "Input": out_seqs,
        "Response": labels,
        "Position": positions,
        "motif_seq": motifseqs,
    }


# ---------------------------------------------------------------------------
# Dataset generation (structure mode)
# ---------------------------------------------------------------------------

def generate_structural_dataset(
    motif_pattern: str = DEFAULT_STRUCT_MOTIF,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    imbalance: float = DEFAULT_IMBALANCE,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Generate a sequence-independent structural motif benchmark dataset.

    Positive examples have a concrete RNA sequence injected at the centre of
    *seq_length* bp that satisfies all base-pairing constraints in
    *motif_pattern* (Watson-Crick + G-U wobble pairs at stem positions; free
    nucleotides elsewhere).  All remaining positions are drawn from a
    first-order Markov chain estimated from the positive sequences, so that
    positive and negative sequences share the same dinucleotide composition.

    Parameters
    ----------
    motif_pattern:
        Dot-bracket string defining the structural motif, e.g. ``<<<<....>>>>``.
        ``<`` / ``>`` mark paired positions; ``.`` marks unpaired positions.
    num_examples:
        Total number of sequences to generate.
    seq_length:
        Length of each generated sequence.
    imbalance:
        Fraction of positive (structure-enforcing) sequences.
    seed:
        Random seed.

    Returns
    -------
    Dict with keys ``Input``, ``Response``, ``Position``, ``motif_seq``.
    """
    rng = np.random.default_rng(seed)
    pairs, free = parse_dot_bracket(motif_pattern)
    motif_len = len(motif_pattern)
    insert_pos = (seq_length - motif_len) // 2

    if motif_len >= seq_length:
        raise ValueError(
            f"motif_len={motif_len} must be less than seq_length={seq_length}"
        )

    n_positive = int(num_examples * imbalance)
    n_negative = num_examples - n_positive

    log.info(
        "Structure mode: pattern=%s, motif_len=%d, insert_pos=%d (centre), "
        "positive=%d, negative=%d",
        motif_pattern,
        motif_len,
        insert_pos,
        n_positive,
        n_negative,
    )

    # ------------------------------------------------------------------
    # 1. Build positive sequences
    # ------------------------------------------------------------------
    log.info("Generating positive background sequences ...")
    pos_backgrounds = generate_random_backgrounds(n_positive, seq_length, rng)

    log.info("Generating structure-enforced motif instances ...")
    pos_seqs = []
    pos_motif_seqs = []
    for k in range(n_positive):
        motif_instance = generate_structure_motif_instance(
            motif_len, pairs, free, rng
        )
        bg = pos_backgrounds[k]
        seq = bg[:insert_pos] + motif_instance + bg[insert_pos + motif_len :]
        pos_seqs.append(seq)
        pos_motif_seqs.append(motif_instance)

    log.info("Built %d positive sequences.", n_positive)

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
    # 4. Assemble final arrays
    # ------------------------------------------------------------------
    all_seqs = np.array(pos_seqs + list(neg_seqs), dtype=f"|U{seq_length}")
    labels = np.zeros(num_examples, dtype=bool)
    labels[:n_positive] = True
    positions = np.full(num_examples, fill_value=0, dtype=np.int64)
    positions[:n_positive] = insert_pos
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
            "Generate a synthetic STRUCTMOTIF benchmark joblib and optional "
            "train/tune/validation TSV splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["structure", "pyteiser"],
        default="structure",
        help=(
            "Dataset generation mode. 'structure' (default): generate a "
            "sequence-independent benchmark from a dot-bracket motif pattern "
            "(requires --motif-pattern), fully self-contained. 'pyteiser': "
            "inject a motif from a pyteiser binary seed file (requires "
            "--binpath, a private file not included in this release)."
        ),
    )
    # pyteiser-mode arguments
    parser.add_argument(
        "--binpath",
        type=Path,
        default=DEFAULT_BINPATH,
        help="[pyteiser mode] Pyteiser binary seed file (.bin).",
    )
    parser.add_argument(
        "--motif-idx",
        type=int,
        default=DEFAULT_MOTIF_IDX,
        help="[pyteiser mode] Index of the motif to inject (0-based).",
    )
    # structure-mode arguments
    parser.add_argument(
        "--motif-pattern",
        type=str,
        default=DEFAULT_STRUCT_MOTIF,
        help=(
            "[structure mode] Dot-bracket pattern defining the structural motif "
            "(use '<' for 5'-stem, '>' for 3'-stem, '.' for loop/unpaired)."
        ),
    )
    # shared arguments
    parser.add_argument(
        "--joblib-out",
        type=Path,
        default=None,
        help=(
            "Output path for the generated joblib dataset. "
            f"Defaults to {DEFAULT_JOBLIB_OUT} (pyteiser) or "
            f"{DEFAULT_JOBLIB_OUT_STRUCT} (structure)."
        ),
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
        default="structural_motif",
        help="Dataset name prefix for TSV filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve default joblib output path per mode
    if args.joblib_out is None:
        args.joblib_out = (
            DEFAULT_JOBLIB_OUT_STRUCT if args.mode == "structure" else DEFAULT_JOBLIB_OUT
        )

    args.joblib_out.parent.mkdir(parents=True, exist_ok=True)
    if args.joblib_out.exists() and not args.overwrite:
        raise ValueError(
            f"Output already exists (use --overwrite to replace): {args.joblib_out}"
        )

    if args.mode == "structure":
        log.info(
            "Structure mode: motif_pattern=%s", args.motif_pattern
        )
        dataset = generate_structural_dataset(
            motif_pattern=args.motif_pattern,
            num_examples=args.num_examples,
            seq_length=args.seq_length,
            imbalance=args.imbalance,
            seed=args.seed,
        )
    else:
        # pyteiser mode
        if not args.binpath.exists():
            raise FileNotFoundError(
                f"Pyteiser seed file not found: {args.binpath}"
            )
        motif_dict = load_motif_dict(args.binpath)
        if args.motif_idx not in motif_dict:
            raise ValueError(
                f"motif_idx={args.motif_idx} not found; "
                f"valid range: 0-{max(motif_dict)}"
            )
        log.info(
            "Using motif %d: %s",
            args.motif_idx,
            motif_dict[args.motif_idx]["Sequence"],
        )
        dataset = generate_dataset(
            motif_dict=motif_dict,
            motif_idx=args.motif_idx,
            num_examples=args.num_examples,
            seq_length=args.seq_length,
            imbalance=args.imbalance,
            seed=args.seed,
        )

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
