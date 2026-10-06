# ESM2 protein embeddings vs ecCLIP binding preferences

Code for Fig. 1c,d (RNA-binding-domain embeddings) and Supp. Fig. 1i,j
(whole-protein embeddings): how well distances between ESM2 embeddings of the 67
ecCLIP RBPs agree with distances between their RNA binding preferences.

## Method

- **Proteins.** UniProt canonical sequences (`build_protein_table.py`). IMP3 is the
  U3 snoRNP protein IMP3 (Q9NV31), not IGF2BP3.
- **Embeddings.** `esm2_t33_650M_UR50D`, layer 30, averaged over the residues of
  the annotated RNA-binding domains (UniProt DOMAIN/ZN_FING/REPEAT features) or
  over all residues (`embed_esm2.py`). Proteins without an annotated RNA-binding
  domain fall back to other annotated RNA-contacting domains (9 RBPs) or to the
  full-length protein (EMG1, IMP3, SF3B1, SF3B3); see `data/rbd_spans.tsv`.
- **Binding preferences.** Log enrichment of all 1,024 5-mers in 20-nt windows
  centred on CIMS sites relative to the two flanking windows, using up to 100,000
  sites per RBP split into two disjoint halves whose profiles are averaged
  (`extract_peak_windows.py`, `kmer_enrichment.py`).
- **Concordance.** After z-scoring each space: modified RV coefficient, Mantel test
  (Spearman ρ on pairwise cosine distances) and distance correlation, each against
  9,999 permutations of the RBP labels (`concordance.py`).
- **Dendrograms.** Ward linkage on Euclidean distances, cut at k = 6; both trees
  are coloured by the protein-embedding clusters (`plot_concordance.py`).

## Running

```bash
./run.sh
```

draws both figures from the inputs in `data/`. To rebuild those inputs from
UniProt, ESM2 and the ecCLIP CIMS BED files (`<RBP>_clearCLIP.pool.tag.uniq.del.CIMS.fdr10.bgfilter.bed`):

```bash
FULL=1 BED_DIR=/path/to/bedFiles GENOME=/path/to/hg38.fa ./run.sh
```

Besides the packages in the top-level `pyproject.toml`, the full rebuild needs
`fair-esm` (2.0.0), `pysam` and `requests`. Embedding runs on CPU in a few
minutes.

## Files

| File | Contents |
|---|---|
| `data/rbps.txt` | the 67 RBPs |
| `data/rbp_proteins.tsv`, `data/rbp_proteins.fasta` | UniProt accessions and sequences (release 2026_03) |
| `data/rbd_spans.tsv` | residue spans averaged for the domain embeddings |
| `data/esm2_layer30_rbd.tsv.gz`, `data/esm2_layer30_whole.tsv.gz` | 67 × 1,280 embeddings |
| `data/kmer_enrichment_L20_k5.tsv.gz` | 67 × 1,024 binding-preference profiles |
| `results/concordance.tsv` | statistics, null mean, null 95th percentile, permutation p |
| `results/fig1cd_rbd_embeddings.*` | Fig. 1c,d |
| `results/figS1ij_whole_protein_embeddings.*` | Supp. Fig. 1i,j |

## Results

| Embedding | modified RV | Mantel ρ | distance correlation |
|---|---|---|---|
| RNA-binding domain | 0.110 (p = 0.0002) | 0.083 (p = 0.0018) | 0.605 (p = 0.0001) |
| whole protein | 0.121 (p = 0.0001) | 0.117 (p = 0.0001) | 0.641 (p = 0.0001) |

p = 0.0001 is the smallest value attainable with 9,999 permutations.
