"""
fit_alpha_K.py -- Transparent reconstruction of the theory_extension eq. (8) fit.

Context (see theory_extension_v0.2.md section 1.4):
  v0.1 section 4.3 published pre-registered predictions for alpha at the
  intermediate primes (p=53, 67, 113) BEFORE Phase 7 data existed:
      alpha(53) = -0.83, alpha(67) = -0.79, alpha(113) = -0.67  (+/- 0.05 CI,
      falsification line: |observed - predicted| > 0.10 rejects eq. 8)
  but the stated parameter set (A=2.04, beta=4.5e-5, C=1) does not reproduce
  those numbers under any log convention we tried, and no fit script was saved.

This script is the missing artifact, after the fact and clearly labeled as such:
  1. Solves (A, beta) EXACTLY from the two anchors used in v0.1
     (Phase 6 slopes at p=31 and p=149), with C=1 frozen, under both
     log10 and natural-log conventions for the eq. (8) denominator.
  2. Prints predicted alpha at the intermediates for each convention,
     against BOTH the published section 4.3 numbers and Phase 7 observations.
  3. Repeats the solve with the updated (post-Phase-7) anchors to expose
     anchor instability.
  4. Renders figures/alpha_vs_K_pred_vs_obs.png.

Eq. (8):  alpha(K) = -A * (1 - beta*K) / (C + 0.5*log(K))

Run:  python fit_alpha_K.py     (no GPU; writes figure + prints tables)
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG_DIR = HERE.parent.parent / "figures"   # paper_04_emergence_prediction/figures
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data (provenance: results/phase7/REPORT.md master table, generated 2026-04-28)
# ---------------------------------------------------------------------------

# Observed slope(p) with all Phase 4-7 data combined:
OBSERVED = {  # p: (K=n_train, slope)
    31:  (480,   -0.926),
    53:  (1404,  -0.866),
    67:  (2244,  -0.861),
    97:  (4704,  -0.731),
    113: (6384,  -0.736),
    149: (11100, -0.577),
}

# Anchors used by v0.1 (Phase 6 values, BEFORE Phase 7 added seeds):
PHASE6_ANCHORS = {31: (480, -0.86), 149: (11100, -0.57)}

# Published pre-registered predictions (v0.1 section 4.3 -- the binding numbers):
PUBLISHED = {53: -0.83, 67: -0.79, 113: -0.67}
FALSIFICATION_LINE = 0.10

C_FIXED = 1.0


def solve_two_anchor(anchors, logfn):
    """Exact solve of A, beta from two (K, alpha) anchors with C fixed."""
    (K1, a1), (K2, a2) = anchors.values()
    d1 = C_FIXED + 0.5 * logfn(K1)
    d2 = C_FIXED + 0.5 * logfn(K2)
    r1 = abs(a1) * d1          # = A*(1 - beta*K1)
    r2 = abs(a2) * d2          # = A*(1 - beta*K2)
    ratio = r2 / r1
    beta = (1.0 - ratio) / (K2 - ratio * K1)
    A = r1 / (1.0 - beta * K1)
    return A, beta


def predict(A, beta, K, logfn):
    return -A * (1.0 - beta * K) / (C_FIXED + 0.5 * logfn(K))


def report(tag, anchors, logfn, logname):
    A, beta = solve_two_anchor(anchors, logfn)
    print(f"\n=== {tag} | convention: {logname} | C={C_FIXED} ===")
    print(f"  solved: A = {A:.4f}, beta = {beta:.4e}")
    print(f"  {'p':>4} {'K':>6} {'pred':>8} {'published':>10} {'observed':>9} "
          f"{'|d_pub|':>8} {'|d_obs|':>8} {'falsif(0.10)':>12}")
    worst = 0.0
    for p in (53, 67, 113):
        K, obs = OBSERVED[p]
        pr = predict(A, beta, K, logfn)
        d_pub = abs(pr - PUBLISHED[p])
        d_obs = abs(pr - obs)
        worst = max(worst, d_obs)
        verdict = "PASS" if d_obs <= FALSIFICATION_LINE else "FAIL"
        print(f"  {p:>4} {K:>6} {pr:>8.3f} {PUBLISHED[p]:>10.2f} {obs:>9.3f} "
              f"{d_pub:>8.3f} {d_obs:>8.3f} {verdict:>12}")
    print(f"  -> worst |pred-obs| = {worst:.3f} "
          f"({'SURVIVES' if worst <= FALSIFICATION_LINE else 'EXCEEDS'} the 0.10 line)")
    return A, beta


def main():
    print("Transparent reconstruction of theory_extension eq. (8) fit")
    print("(labels: this script was written 2026-07-07, AFTER Phase 7 data;")
    print(" the binding pre-registered predictions remain the published 4.3 numbers)")

    # The two conventions, Phase-6 anchors (what v0.1 had available):
    A10, b10 = report("Phase-6 anchors", PHASE6_ANCHORS, math.log10, "log10")
    Aln, bln = report("Phase-6 anchors", PHASE6_ANCHORS, math.log, "ln")

    # Updated anchors (post-Phase-7) -- exposes anchor instability:
    upd = {31: (480, OBSERVED[31][1]), 149: (11100, OBSERVED[149][1])}
    report("UPDATED anchors (post-Phase-7)", upd, math.log10, "log10")

    # Published-number verdict (independent of reconstruction):
    print("\n=== Verdict against the PUBLISHED (binding) 4.3 numbers ===")
    for p in (53, 67, 113):
        obs = OBSERVED[p][1]
        d = abs(obs - PUBLISHED[p])
        print(f"  p={p}: published {PUBLISHED[p]:.2f}  observed {obs:.3f}  "
              f"|d|={d:.3f}  {'PASS' if d <= FALSIFICATION_LINE else 'FAIL'}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"  # Okabe-Ito, validated
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)

    Ks = [OBSERVED[p][0] for p in sorted(OBSERVED)]
    obs_mag = [abs(OBSERVED[p][1]) for p in sorted(OBSERVED)]

    # eq. (8) curves from Phase-6 anchors (both conventions), thin lines
    import numpy as np
    Kgrid = np.logspace(math.log10(400), math.log10(13000), 200)
    ax.plot(Kgrid, [abs(predict(A10, b10, k, math.log10)) for k in Kgrid],
            color=GREEN, lw=2, ls="-", label="eq. (8) from Phase-6 anchors (log10)")
    ax.plot(Kgrid, [abs(predict(Aln, bln, k, math.log)) for k in Kgrid],
            color=GREEN, lw=2, ls="--", label="eq. (8) from Phase-6 anchors (ln)")
    Aupd, bupd = solve_two_anchor(
        {31: (480, OBSERVED[31][1]), 149: (11100, OBSERVED[149][1])}, math.log10)
    ax.plot(Kgrid, [abs(predict(Aupd, bupd, k, math.log10)) for k in Kgrid],
            color="#666666", lw=1.5, ls=":",
            label="eq. (8) from post-Phase-7 anchors (matches published ≤0.006)")

    # published pre-registered predictions with the 0.10 falsification band
    pub_K = [OBSERVED[p][0] for p in (53, 67, 113)]
    pub_a = [abs(PUBLISHED[p]) for p in (53, 67, 113)]
    ax.errorbar(pub_K, pub_a, yerr=FALSIFICATION_LINE, fmt="D", color=VERM,
                ms=7, capsize=4, lw=1.5,
                label="published §4.3 predictions ±0.10 (provenance contested, v0.2 §1)")

    # observed points, direct-labeled with p
    ax.plot(Ks, obs_mag, "o", color=BLUE, ms=8, label="observed slope (Phases 4–7)")
    for p in sorted(OBSERVED):
        K, a = OBSERVED[p]
        ax.annotate(f"p={p}", (K, abs(a)), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#444444")

    ax.set_xscale("log")
    ax.set_xlabel("K  (training-set size, log scale)")
    ax.set_ylabel("|α|   magnitude of log–log slope  t_grok ∝ wd^α")
    ax.set_title("Grokking exponent |α| vs task size K: eq. (8) reconstructions vs Phase 7 data",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()

    out = FIG_DIR / "alpha_vs_K_pred_vs_obs.png"
    fig.savefig(out)
    print(f"\nFigure written: {out}")


if __name__ == "__main__":
    main()
