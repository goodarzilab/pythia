#!/usr/bin/env bash
# Reproduce Fig. 1c,d (RNA-binding-domain embeddings) and Supp. Fig. 1i,j
# (whole-protein embeddings).
#
# By default the figures are drawn from the inputs shipped in data/.
# FULL=1 rebuilds those inputs first, which needs network access (UniProt, ESM2
# weights), the ecCLIP CIMS BED files and an indexed hg38 FASTA:
#
#   FULL=1 BED_DIR=/path/to/bedFiles GENOME=/path/to/hg38.fa ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${FULL:-0}" == 1 ]]; then
    : "${BED_DIR:?set BED_DIR to the directory with the CIMS BED files}"
    : "${GENOME:?set GENOME to an indexed hg38 FASTA}"
    WORK=${WORK:-work}
    python build_protein_table.py --rbps data/rbps.txt --outdir data
    python embed_esm2.py --fasta data/rbp_proteins.fasta --spans data/rbd_spans.tsv \
        --outdir data --layers 30
    python extract_peak_windows.py --rbps data/rbps.txt --bed-dir "$BED_DIR" \
        --genome "$GENOME" --outdir "$WORK/windows" --L 20 --cap 100000 --n-splits 2
    python kmer_enrichment.py --rbps data/rbps.txt --windows "$WORK/windows" \
        --out data/kmer_enrichment_L20_k5.tsv.gz --k 5
fi

python concordance.py --preferences data/kmer_enrichment_L20_k5.tsv.gz \
    --embedding rbd=data/esm2_layer30_rbd.tsv.gz \
    --embedding whole=data/esm2_layer30_whole.tsv.gz \
    --out results/concordance.tsv --metric cosine --k 6 --n-perm 9999 --seed 0

python plot_concordance.py --embedding data/esm2_layer30_rbd.tsv.gz --name rbd \
    --label "RNA-binding-domain" --letters c,d --out results/fig1cd_rbd_embeddings
python plot_concordance.py --embedding data/esm2_layer30_whole.tsv.gz --name whole \
    --label "whole-protein" --letters i,j --out results/figS1ij_whole_protein_embeddings
