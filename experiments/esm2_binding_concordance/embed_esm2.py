"""Embed the RBPs with ESM2 and pool per protein.

Each full-length protein is passed through esm2_t33_650M_UR50D once. Two
representations are written per requested layer:

    esm2_layer<L>_whole.tsv.gz   mean over all residues (BOS/EOS excluded)
    esm2_layer<L>_rbd.tsv.gz     mean over the residues in data/rbd_spans.tsv

The domain representation keeps the full-protein context and only restricts the
residues that are averaged.

Usage
-----
    python embed_esm2.py --fasta data/rbp_proteins.fasta \\
        --spans data/rbd_spans.tsv --outdir data --layers 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch


def read_fasta(path: str) -> list[tuple[str, str]]:
    records, name, seq = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(seq)))
            name, seq = line[1:].split("|")[0], []
        elif line:
            seq.append(line)
    if name is not None:
        records.append((name, "".join(seq)))
    return records


def parse_spans(path: str) -> dict[str, list[int]]:
    table = pd.read_csv(path, sep="\t")
    residues = {}
    for rbp, spans in zip(table.rbp, table.spans):
        keep = []
        for span in str(spans).split(","):
            s, e = (int(x) for x in span.split("-"))
            keep.extend(range(s, e + 1))
        residues[rbp] = keep
    return residues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fasta", default="data/rbp_proteins.fasta")
    ap.add_argument("--spans", default="data/rbd_spans.tsv")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--layers", type=int, nargs="+", default=[30])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    records = read_fasta(args.fasta)
    residues = parse_spans(args.spans)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter()

    pooled = {(layer, kind): [] for layer in args.layers for kind in ("whole", "rbd")}
    for i, (name, seq) in enumerate(records, 1):
        _, _, tokens = batch_converter([(name, seq)])
        with torch.inference_mode():
            out = model(tokens.to(args.device), repr_layers=args.layers)
        # token 0 is BOS, so token index == 1-based residue index
        rbd = torch.tensor([r for r in residues[name] if r <= len(seq)])
        for layer in args.layers:
            rep = out["representations"][layer][0]
            pooled[layer, "whole"].append(rep[1 : len(seq) + 1].mean(0).numpy())
            pooled[layer, "rbd"].append(rep[rbd].mean(0).numpy())
        if i % 10 == 0 or i == len(records):
            print(f"{i}/{len(records)}", flush=True)

    outdir = Path(args.outdir)
    names = [name for name, _ in records]
    for (layer, kind), rows in pooled.items():
        mat = np.vstack(rows)
        df = pd.DataFrame(
            mat, index=pd.Index(names, name="rbp"),
            columns=[f"ESM_{j:04d}" for j in range(mat.shape[1])],
        )
        df.to_csv(outdir / f"esm2_layer{layer}_{kind}.tsv.gz", sep="\t")


if __name__ == "__main__":
    main()
