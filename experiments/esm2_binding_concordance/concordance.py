"""Concordance between ESM2 protein embeddings and ecCLIP binding preferences.

Both matrices are z-scored per feature. Statistics, each tested against a null
built by permuting the RBP labels of the preference matrix:

    RV_modified  modified RV coefficient (Smilde et al. 2009)
    mantel_rho   Spearman correlation between the pairwise distance vectors
    dcor         distance correlation R (Szekely et al. 2007)
    ari          adjusted Rand index between Ward clusterings cut at --k

Distances for the Mantel test and distance correlation use --metric; the Ward
clusterings use Euclidean distance, as in the dendrograms.

Usage
-----
    python concordance.py --preferences data/kmer_enrichment_L20_k5.tsv.gz \\
        --embedding rbd=data/esm2_layer30_rbd.tsv.gz \\
        --embedding whole=data/esm2_layer30_whole.tsv.gz \\
        --out results/concordance.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

STATISTICS = ["RV_modified", "mantel_rho", "dcor", "ari"]


def zscore(df: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(StandardScaler().fit_transform(df.values.astype(float)))


def centred_gram(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    K = Xc @ Xc.T
    np.fill_diagonal(K, 0.0)
    return K


def double_centre(D: np.ndarray) -> np.ndarray:
    return D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def ranked(D: np.ndarray, tri) -> np.ndarray:
    R = np.zeros_like(D)
    R[tri] = rankdata(D[tri])
    return R + R.T


def ward(X: np.ndarray, k: int) -> np.ndarray:
    return fcluster(linkage(pdist(X), method="ward"), k, criterion="maxclust")


def concordance(X, Y, metric, k, n_perm, seed):
    """Observed statistics and permutation p-values for one pair of spaces."""
    n = len(X)
    tri = np.triu_indices(n, 1)
    Kx, Ky = centred_gram(X), centred_gram(Y)
    Dx = squareform(pdist(X, metric=metric))
    Dy = squareform(pdist(Y, metric=metric))
    Ax, By = double_centre(Dx), double_centre(Dy)
    Rx, Ry = ranked(Dx, tri), ranked(Dy, tri)
    cx, cy = ward(X, k), ward(Y, k)
    norm_rv = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    norm_dc = np.sqrt(np.sum(Ax * Ax) * np.sum(By * By))

    def stats(p=None):
        if p is None:
            Ky_, By_, Ry_, cy_ = Ky, By, Ry, cy
        else:
            g = np.ix_(p, p)
            Ky_, By_, Ry_, cy_ = Ky[g], By[g], Ry[g], cy[p]
        dcor_sq = np.sum(Ax * By_) / norm_dc
        return [
            float(np.sum(Kx * Ky_) / norm_rv),
            pearson(Rx[tri], Ry_[tri]),
            float(np.sqrt(max(dcor_sq, 0.0))),
            float(adjusted_rand_score(cx, cy_)),
        ]

    observed = np.array(stats())
    rng = np.random.default_rng(seed)
    null = np.array([stats(rng.permutation(n)) for _ in range(n_perm)])
    p = (np.sum(null >= observed, axis=0) + 1) / (n_perm + 1)
    return observed, null, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preferences", required=True)
    ap.add_argument("--embedding", action="append", required=True,
                    help="name=path, may be given more than once")
    ap.add_argument("--out", default="results/concordance.tsv")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = pd.read_csv(args.preferences, sep="\t", index_col=0)
    rows = []
    for spec in args.embedding:
        name, path = spec.split("=", 1)
        E = pd.read_csv(path, sep="\t", index_col=0)
        rbps = [r for r in E.index if r in P.index]
        X, Y = zscore(E.loc[rbps]), zscore(P.loc[rbps])
        obs, null, p = concordance(X, Y, args.metric, args.k, args.n_perm, args.seed)
        for j, stat in enumerate(STATISTICS):
            rows.append(dict(
                embedding=name, statistic=stat, n_rbps=len(rbps),
                value=round(obs[j], 4),
                null_mean=round(float(null[:, j].mean()), 4),
                null_p95=round(float(np.percentile(null[:, j], 95)), 4),
                perm_p=round(float(p[j]), 5),
            ))
        print(name, " ".join(f"{s}={v:.3f} (p={q:.4f})"
                             for s, v, q in zip(STATISTICS, obs, p)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
