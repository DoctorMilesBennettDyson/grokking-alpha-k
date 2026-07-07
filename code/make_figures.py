"""
make_figures.py -- Publication figures for the alpha(K) paper, from committed CSVs.

Reads results/ CSVs (no GPU, no re-training) and writes to ../../figures/ (repo: figures/).
Palette: Okabe-Ito subset, CVD-validated (blue #0072B2, vermillion #D55E00, green #009E73).
Heatmap uses cividis (perceptually uniform, colorblind-safe).

Figures:
  fig1_alpha_vs_K.png       -- headline: |alpha| vs K, 6 points
  fig2_master_heatmap.png   -- t_grok across p x wd (log color)
  fig3_regime_contrast.png  -- grokking gap (depth-1) vs no gap (depth-2)
  fig4_negative_results.png -- H4 batch-size bars + SGD-vs-AdamW loss (weight collapse)
"""
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE.parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})


def read_curve(path):
    steps, tr, te, trl = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            tr.append(float(row["train_acc"]))
            te.append(float(row["test_acc"]))
            trl.append(float(row["train_loss"]))
    return np.array(steps), np.array(tr), np.array(te), np.array(trl)


# --- data from results/phase7/REPORT.md master table (frozen) ---
ALPHA = [(31, 480, 0.926), (53, 1404, 0.866), (67, 2244, 0.861),
         (97, 4704, 0.731), (113, 6384, 0.736), (149, 11100, 0.577)]
PRIMES = [31, 53, 67, 97, 113, 149]
WDS = [0.01, 0.1, 0.3, 1.0, 3.0]
TGROK = {  # p: {wd: mean t_grok}
    31:  {0.01: 198367, 0.1: 39660, 0.3: 10267, 1.0: 2767, 3.0: 1200},
    53:  {0.01: 153700, 0.1: 32767, 0.3: 9433,  1.0: 2567, 3.0: 1350},
    67:  {0.01: 91300,  0.1: 23233, 0.3: 6733,  1.0: 1933, 3.0: 767},
    97:  {0.01: 30133,  0.1: 9267,  0.3: 4100,  1.0: 1314, 3.0: 471},
    113: {0.01: 31767,  0.1: 6800,  0.3: 2933,  1.0: 1067, 3.0: 500},
    149: {0.01: 13233,  0.1: 5767,  0.3: 2333,  1.0: 833,  3.0: 617},
}
H4 = [(256, 1067), (512, 1314), (1024, 1467)]  # phase8 H4


def fig1():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ks = [k for _, k, _ in ALPHA]
    A = [a for _, _, a in ALPHA]
    ax.plot(Ks, A, "-", color=GREY, lw=1, alpha=0.5, zorder=1)
    ax.plot(Ks, A, "o", color=BLUE, ms=9, zorder=3)
    for p, k, a in ALPHA:
        ax.annotate(f"p={p}", (k, a), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("K   (training-set size = 0.5·p², log scale)")
    ax.set_ylabel("|α|   exponent of  t_grok ∝ wd^α")
    ax.set_title("Weight-decay's grip on grokking weakens as the task grows", fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, which="both", alpha=0.15, lw=0.6)
    fig.tight_layout(); fig.savefig(FIG / "fig1_alpha_vs_K.png"); plt.close(fig)


def fig2():
    M = np.array([[TGROK[p][wd] for wd in WDS] for p in PRIMES], dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    im = ax.imshow(M, aspect="auto", cmap="cividis_r",
                   norm=LogNorm(vmin=M.min(), vmax=M.max()))
    ax.set_xticks(range(len(WDS))); ax.set_xticklabels(WDS)
    ax.set_yticks(range(len(PRIMES))); ax.set_yticklabels([f"p={p}" for p in PRIMES])
    ax.set_xlabel("weight decay"); ax.set_ylabel("modulus (task size K increases downward)")
    ax.set_title("Grokking time across the task × regularization grid", fontsize=11)
    for i in range(len(PRIMES)):
        for j in range(len(WDS)):
            v = M[i, j]
            lum = LogNorm(vmin=M.min(), vmax=M.max())(v)
            txt = f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if lum > 0.55 else "black")
    cb = fig.colorbar(im, ax=ax); cb.set_label("t_grok (steps, log)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_master_heatmap.png"); plt.close(fig)


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    # left: classic grokking (p=97, wd=0.01) -- big memorize->generalize gap
    s, tr, te, _ = read_curve(RES / "phase5/wd0.01_s0.csv")
    axes[0].plot(s, tr, color=BLUE, lw=2, label="train acc")
    axes[0].plot(s, te, color=VERM, lw=2, label="test acc")
    axes[0].set_title("Depth-1, wd=0.01 (p=97): grokking gap", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8.5, loc="center right")
    # right: depth-2 -- train and test rise together, NO gap
    s2, tr2, te2, _ = read_curve(RES / "phase8/H1_depth2_wd0.1_s0.csv")
    axes[1].plot(s2, tr2, color=BLUE, lw=2, label="train acc")
    axes[1].plot(s2, te2, color=VERM, lw=2, label="test acc")
    axes[1].set_title("Depth-2, wd=0.1 (p=97): no memorization phase", fontsize=10.5)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("step (log)"); ax.grid(True, which="both", alpha=0.15, lw=0.6)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("The wd power law survives a regime where grokking itself disappears (§4.3)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig3_regime_contrast.png"); plt.close(fig)


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    # left: H4 batch-size bars
    bss = [str(b) for b, _ in H4]; vals = [v for _, v in H4]
    bars = axes[0].bar(bss, vals, color=BLUE, width=0.6, zorder=3)
    axes[0].bar_label(bars, fmt="%d", padding=3, fontsize=9)
    axes[0].set_title("H4: larger batch → slower grokking\n(noise accelerates; §4.4)", fontsize=10)
    axes[0].set_xlabel("batch size"); axes[0].set_ylabel("t_grok (steps)")
    axes[0].set_ylim(0, max(vals) * 1.18); axes[0].grid(True, axis="y", alpha=0.15, lw=0.6)
    # right: SGD (frozen at ln p) vs AdamW loss
    s_a, _, _, trl_a = read_curve(RES / "phase8/H3_initxavier_s20.csv")   # AdamW, groks
    s_s, _, _, trl_s = read_curve(RES / "phase8/H2_sgd_wd1.0_s0.csv")     # SGD, collapses
    axes[1].plot(s_a, trl_a, color=GREEN, lw=2, label="AdamW (wd=1.0)")
    axes[1].plot(s_s, trl_s, color=VERM, lw=2, label="SGD-momentum (wd=1.0)")
    axes[1].axhline(math.log(99), color=GREY, ls=":", lw=1.2, label="uniform loss ln(99)")
    axes[1].set_title("H2: SGD weight-norm collapse\ntrain loss frozen at chance (§4.2)", fontsize=10)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("train loss")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(True, alpha=0.15, lw=0.6)
    axes[1].set_xlim(0, 20000)
    fig.suptitle("Two mechanistic negative results", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig4_negative_results.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("Figures written to", FIG.resolve())
    for f in sorted(FIG.glob("fig*.png")):
        print("  ", f.name, f"{f.stat().st_size//1024} KB")
