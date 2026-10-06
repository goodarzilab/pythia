"""Plot ESM2-embedding vs ecCLIP-preference concordance (Fig. 1c,d; Supp. Fig. 1i,j).

First panel: concordance statistics against their permutation null (grey band, up
to the 95th percentile), and pairwise cosine distances in the two spaces.
Second panel: circular Ward dendrograms of the RBPs built from the protein
embeddings (left) and the binding preferences (right). Both trees are coloured by
the protein-embedding clusters cut at --k.

Usage
-----
    python plot_concordance.py --embedding data/esm2_layer30_rbd.tsv.gz \\
        --name rbd --label "RNA-binding-domain" --letters c,d \\
        --out results/fig1cd_rbd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BAR, NULL_BAND = PALETTE[0], "#eceae4"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"
DISPLAY_NAME = {"LIN28b": "LIN28B"}
STAT_LABEL = {"dcor": "distance correlation", "RV_modified": "modified RV",
              "mantel_rho": "Mantel ρ"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def zscore(df: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(StandardScaler().fit_transform(df.values.astype(float)))


def circular_dendrogram(ax, Z, names, leaf_colour, title, gap=0.035):
    n = len(names)
    members = {}
    for i, (a, b, _, _) in enumerate(Z):
        a, b = int(a), int(b)
        members[i] = ({a} if a < n else members[a - n]) | ({b} if b < n else members[b - n])

    def link_colour(node):
        colours = {leaf_colour[names[leaf]] for leaf in members[node - n]}
        return colours.pop() if len(colours) == 1 else MUTED

    dd = dendrogram(Z, no_plot=True, link_color_func=link_colour)
    span, xmax = 2 * np.pi * (1 - gap), 10.0 * n
    ymax = max(max(d) for d in dd["dcoord"])

    def theta(x):
        return np.pi / 2 - span * x / xmax

    def radius(y):
        return 1.0 - 0.97 * y / ymax

    for (x1, _, _, x2), (y1, ym, _, y2), colour in zip(
        dd["icoord"], dd["dcoord"], dd["color_list"]
    ):
        for x, y in ((x1, y1), (x2, y2)):
            t = theta(x)
            ax.plot([radius(y) * np.cos(t), radius(ym) * np.cos(t)],
                    [radius(y) * np.sin(t), radius(ym) * np.sin(t)],
                    color=colour, lw=0.7, solid_capstyle="round")
        arc = np.linspace(theta(x1), theta(x2), 40)
        ax.plot(radius(ym) * np.cos(arc), radius(ym) * np.sin(arc),
                color=colour, lw=0.7)

    for j, leaf in enumerate(dd["leaves"]):
        name = names[leaf]
        t = theta(5.0 + 10.0 * j)
        deg = np.degrees(t)
        right = -90 <= deg <= 90
        ax.text(1.035 * np.cos(t), 1.035 * np.sin(t), DISPLAY_NAME.get(name, name),
                rotation=deg if right else deg + 180, rotation_mode="anchor",
                ha="left" if right else "right", va="center", fontsize=5.2,
                color=leaf_colour[name])
    ax.set_xlim(-1.33, 1.33)
    ax.set_ylim(-1.33, 1.33)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8.2, color=INK, y=1.04)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--preferences", default="data/kmer_enrichment_L20_k5.tsv.gz")
    ap.add_argument("--concordance", default="results/concordance.tsv")
    ap.add_argument("--name", required=True, help="embedding name in --concordance")
    ap.add_argument("--label", required=True, help="e.g. 'RNA-binding-domain'")
    ap.add_argument("--letters", default="c,d")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()
    first, second = (x.strip() for x in args.letters.split(","))

    E = pd.read_csv(args.embedding, sep="\t", index_col=0)
    P = pd.read_csv(args.preferences, sep="\t", index_col=0)
    names = [r for r in E.index if r in P.index]
    X, Y = zscore(E.loc[names]), zscore(P.loc[names])
    Zx, Zy = linkage(pdist(X), "ward"), linkage(pdist(Y), "ward")
    clusters = fcluster(Zx, args.k, "maxclust")
    # assign palette colours in the order clusters appear around the protein tree
    order = list(dict.fromkeys(clusters[i] for i in dendrogram(Zx, no_plot=True)["leaves"]))
    colour = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}
    leaf_colour = {name: colour[c] for name, c in zip(names, clusters)}

    cc = pd.read_csv(args.concordance, sep="\t")
    cc = cc[cc.embedding == args.name].set_index("statistic")

    fig = plt.figure(figsize=(7.4, 6.9))
    gs = GridSpec(2, 2, height_ratios=[0.60, 1], hspace=0.30, wspace=0.20, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    stats = ["dcor", "RV_modified", "mantel_rho"]
    for y, stat in zip(range(len(stats))[::-1], stats):
        r = cc.loc[stat]
        ax.barh(y, r.null_p95, height=0.62, color=NULL_BAND, edgecolor="none", zorder=1)
        ax.barh(y, r.value, height=0.45, color=BAR, zorder=3)
        inside = r.value > 0.45
        ax.text(r.value - 0.02 if inside else r.value + 0.02, y,
                f"{r.value:.3f}   p = {r.perm_p:g}", va="center",
                ha="right" if inside else "left", fontsize=7,
                color="white" if inside else INK2, zorder=4)
    ax.set_yticks(range(len(stats))[::-1])
    ax.set_yticklabels([STAT_LABEL[s] for s in stats])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("value", color=INK2)
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=NULL_BAND, label="range expected by chance\n"
                             "(permutation null, to 95th pct)")],
              frameon=False, fontsize=6.5, loc="center right",
              bbox_to_anchor=(1.02, 0.33), handlelength=1.4)
    ax.set_title(f"{first}   {args.label} embeddings vs ecCLIP binding preference",
                 loc="left", fontsize=8.5, pad=6)

    ax = fig.add_subplot(gs[0, 1])
    dx, dy = pdist(X, metric=args.metric), pdist(Y, metric=args.metric)
    ax.scatter(dx, dy, s=4.5, c=BAR, alpha=0.25, linewidths=0, rasterized=True)
    ax.set_xlabel(f"{args.label} embedding distance ({args.metric})", color=INK2)
    ax.set_ylabel("binding preference distance", color=INK2)
    r = cc.loc["mantel_rho"]
    ax.text(0.035, 0.97, f"ρ = {r.value:+.3f},  p = {r.perm_p:g}\n"
            f"n = {len(names)} RBPs, {len(dx):,} pairs",
            transform=ax.transAxes, va="top", fontsize=7, color=INK2, linespacing=1.4)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    left, right = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    circular_dendrogram(left, Zx, names, leaf_colour, f"ESM2 {args.label} embeddings")
    circular_dendrogram(right, Zy, names, leaf_colour, "ecCLIP k-mer binding preferences")
    left.text(-1.30, 1.40, second, fontsize=10, fontweight="bold")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for letter, axes in ((first, fig.axes[:2]), (second, fig.axes[2:])):
        box = Bbox.union([a.get_tightbbox(renderer) for a in axes])
        box = box.transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.10)
        for ext in ("pdf", "svg"):
            fig.savefig(out.parent / f"{out.name}_panel_{letter}.{ext}", bbox_inches=box)
    plt.close(fig)


if __name__ == "__main__":
    main()
