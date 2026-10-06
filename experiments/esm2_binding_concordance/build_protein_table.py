"""Resolve the 67 ecCLIP RBPs to UniProt canonical sequences and RNA-binding domains.

For each RBP in data/rbps.txt this queries UniProt (reviewed, human) and writes:

    data/rbp_proteins.tsv     rbp, uniprot, gene, length, protein_name, sequence
    data/rbp_proteins.fasta   canonical full-length sequences
    data/rbd_spans.tsv        residue spans averaged for the domain embedding

Domain spans come from UniProt DOMAIN / ZN_FING / REPEAT features. A protein's
spans are its annotated RNA-binding domains; if it has none, other RNA-contacting
domains (helicase, deaminase, ...) are used; if it has no annotated domain at all,
the full-length protein is used.

The files shipped in data/ were built against UniProt release 2026_03. Running this
against a later release may change annotations.

Usage
-----
    python build_protein_table.py --rbps data/rbps.txt --outdir data
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ecCLIP target name -> UniProt gene symbol, where they differ
ALIASES = {"ADAR1": "ADAR", "LIN28b": "LIN28B"}

# accessions fixed where a symbol is ambiguous or matches several reviewed entries
PINNED = {
    "IMP3": "Q9NV31",  # U3 snoRNP protein IMP3, not IGF2BP3 (O00425)
    "RBM10": "P98175",
}

RBD_KEYWORDS = [
    "RRM", "XRRM", "KH", "DRBM", "CSD", "YTH", "SM", "CCHC", "RANBP2",
    "LA-TYPE", "PUA", "THUMP", "SAP", "DZF", "PAZ", "PIWI", "AGENET",
    "SPOC", "NTF2", "PABC", "C2H2", "G-PATCH", "TUDOR", "AKAP95", "HTH",
]
AUX_KEYWORDS = [
    "HELICASE", "DEAMINASE", "EDITASE", "KINASE", "METHYLTRANS", "TRM1",
    "GTPASE", "TR-TYPE", "UPF1", "CH-RICH", "Z-BINDING", "SAM", "OCRE",
    "B30.2", "UBIQUITIN", "SUZ", "IQ",
]
FIELDS = (
    "accession,id,gene_primary,protein_name,length,sequence,"
    "ft_domain,ft_zn_fing,ft_repeat"
)
FEATURE = re.compile(r'(DOMAIN|ZN_FING|REPEAT) (\d+)\.\.(\d+);(?: /note="([^"]*)")?')


def domain_class(note: str) -> tuple[str, str]:
    note = (note or "").upper()
    for key in RBD_KEYWORDS:
        if key in note:
            return key, "rbd"
    for key in AUX_KEYWORDS:
        if key in note:
            return key, "aux"
    return "OTHER", "other"


def features(text) -> list[tuple[int, int, str]]:
    if not isinstance(text, str):
        return []
    return [(int(m[2]), int(m[3]), m[4] or "") for m in FEATURE.finditer(text)]


def query_uniprot(genes: list[str], retries: int = 3) -> tuple[pd.DataFrame, str]:
    query = " OR ".join(f"gene_exact:{g}" for g in sorted(set(genes)))
    params = {
        "query": f"({query}) AND organism_id:9606 AND reviewed:true",
        "fields": FIELDS,
        "format": "tsv",
        "size": "500",
    }
    for attempt in range(retries):
        try:
            r = requests.get(
                "https://rest.uniprot.org/uniprotkb/search", params=params, timeout=120
            )
            r.raise_for_status()
            release = r.headers.get("x-uniprot-release", "unknown")
            return pd.read_csv(io.StringIO(r.text), sep="\t"), release
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rbps", default="data/rbps.txt")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rbps = Path(args.rbps).read_text().split()
    genes = {rbp: ALIASES.get(rbp, rbp) for rbp in rbps}
    raw, release = query_uniprot(list(genes.values()))
    raw["primary"] = raw["Gene Names (primary)"].fillna("").map(
        lambda s: [x.strip() for x in s.split(";") if x.strip()]
    )
    print(f"UniProt release {release}: {len(raw)} reviewed entries")

    proteins, missing = [], []
    for rbp, gene in genes.items():
        hits = raw[raw.primary.map(lambda names, g=gene: g in names)]
        if rbp in PINNED:
            hits = hits[hits.Entry == PINNED[rbp]]
        if hits.empty:
            missing.append(rbp)
            continue
        best = hits.sort_values(["Length", "Entry"], ascending=[False, True]).iloc[0]
        proteins.append(best)
    if missing:
        sys.exit(f"no UniProt entry for: {missing}")

    table, spans = [], []
    for rbp, p in zip(rbps, proteins):
        feats = (
            features(p["Domain [FT]"])
            + features(p["Zinc finger"])
            + features(p["Repeat"])
        )
        by_class: dict[str, list] = {"rbd": [], "aux": []}
        families: dict[str, set] = {"rbd": set(), "aux": set()}
        for s, e, note in feats:
            family, cls = domain_class(note)
            if cls in by_class:
                by_class[cls].append((s, e))
                families[cls].add(family)
        if by_class["rbd"]:
            rule, use, fam = "rbd", by_class["rbd"], families["rbd"]
        elif by_class["aux"]:
            rule, use, fam = "aux_fallback", by_class["aux"], families["aux"]
        else:
            rule, use, fam = "whole_protein_fallback", [(1, int(p.Length))], set()
        merged = merge_spans(use)
        n_res = sum(e - s + 1 for s, e in merged)
        table.append(
            dict(
                rbp=rbp,
                uniprot=p.Entry,
                gene=p["Gene Names (primary)"],
                length=int(p.Length),
                protein_name=p["Protein names"],
                sequence=p.Sequence,
            )
        )
        spans.append(
            dict(
                rbp=rbp,
                uniprot=p.Entry,
                rule=rule,
                n_spans=len(merged),
                rbd_len=n_res,
                frac_of_protein=round(n_res / int(p.Length), 3),
                families=";".join(sorted(fam)),
                spans=",".join(f"{s}-{e}" for s, e in merged),
            )
        )

    table = pd.DataFrame(table)
    spans = pd.DataFrame(spans)
    assert table.uniprot.nunique() == len(rbps)
    table.to_csv(outdir / "rbp_proteins.tsv", sep="\t", index=False)
    spans.to_csv(outdir / "rbd_spans.tsv", sep="\t", index=False)
    with open(outdir / "rbp_proteins.fasta", "w") as fh:
        for r in table.itertuples():
            fh.write(f">{r.rbp}|{r.uniprot}\n{r.sequence}\n")
    print(spans.rule.value_counts().to_string())


if __name__ == "__main__":
    main()
