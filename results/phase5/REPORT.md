# Phase 5 — REPORT

Generated: 2026-04-26 (manual analysis by Opus 4.7)

---

## Purpose

Address two weaknesses flagged in Phase 4 review:

- **(A) Baseline robustness**: Phase 4 baseline s1 had final_test_acc=0.705 — grokking
  threshold triggered but not sustained. N=3 was insufficient. Phase 5 adds 5 new seeds,
  raising N=7 clean runs.
- **(B) Weight-decay sweep**: weight_decay was fixed at 1.0 in all prior phases. This is
  the canonical grokking regularizer (Power et al. 2022) and its absence was the main
  gap a reviewer would flag.

---

## (A) Baseline — N=7 clean (Phase 4 + Phase 5 combined)

| source | seed | t_grok | final_test_acc | clean |
|---|---|---|---|---|
| phase4 | 0 | 1200 | 1.0000 | yes |
| phase4 | 1 | 1200 | 0.7052 | **NO — excluded** |
| phase4 | 2 | 1400 | 1.0000 | yes |
| phase5 | 10 | 1600 | 1.0000 | yes |
| phase5 | 11 | 1400 | 1.0000 | yes |
| phase5 | 12 | 1400 | 1.0000 | yes |
| phase5 | 13 | 1200 | 1.0000 | yes |
| phase5 | 14 | 1000 | 1.0000 | yes |

**Updated baseline: mean = 1314 steps, std = 195, N=7 (clean grokks only).**

The Phase 4 reported baseline mean was 1267 (N=3, including the outlier). The updated
mean is 1314 (+3.7%). The outlier was a case where the grokking threshold (test_acc ≥ 0.95)
was momentarily satisfied but the model did not reach stable generalization by step 100K.

**Conclusion (A):** baseline is robust. The N=7 runs cluster around 1000–1600 with low
variance. The outlier phenomenon is real but rare (~1 in 8 seeds). For Paper 04, report
mean=1314 ± 195 (N=7) and footnote the exclusion criterion.

---

## (B) Weight-decay sweep

All other hyperparameters identical to baseline (p=97, depth=1, dropout=0, ls=0,
steps=100K). weight_decay varied over 3 orders of magnitude.

### Per-run data

| wd | s0 | s1 | s2 | s1_acc |
|---|---|---|---|---|
| 0.01 | 32300 | 20800 | 35600 | 0.9995, 0.9979, 1.0 |
| 0.10 | 8900 | 7900 | 11000 | 1.0, 1.0, 1.0 |
| 0.30 | 4000 | 4200 | 4100 | 1.0, 1.0, 0.9998 |
| 1.0 (baseline) | — | — | — | (see Section A) |
| 3.0 | 400 | 500 ⚠️ | 500 | 1.0, **0.416**, 1.0 |

⚠️ wd=3.0 s1: final_test_acc=0.416 — same outlier pattern as baseline s1. Threshold hit
momentarily but not sustained. Excluded from mean.

### Aggregated (clean grokks only)

| weight_decay | N_clean | mean t_grok | std | vs baseline | ratio |
|---|---|---|---|---|---|
| 0.01 | 3 | 29567 | 7770 | 22.5× slower | — |
| 0.10 | 3 | 9267 | 1582 | 7.1× slower | 3.2× vs wd=0.01 |
| 0.30 | 3 | 4100 | 100 | 3.1× slower | 2.3× vs wd=0.10 |
| **1.0** | **7** | **1314** | **195** | **baseline** | 3.1× vs wd=0.30 |
| 3.0 | 2 | 450 | 71 | 2.9× faster | 2.9× vs wd=1.0 |

**Total speedup from wd=0.01 to wd=3.0: ~65×.**

### t_train_saturated analysis

Critical observation: weight_decay does NOT change t_train_saturated.

| wd | t_train_saturated (typical) | t_grok | gap |
|---|---|---|---|
| 0.01 | 400 | 29567 | 29167 |
| 0.10 | 400 | 9267 | 8867 |
| 0.30 | 400 | 4100 | 3700 |
| 1.0 | 400 | 1314 | 914 |
| 3.0 | 400–500 | 450 | ~50 |

Weight decay controls the size of the generalization gap, not the speed of memorization.
High wd → near-zero gap → grokking almost simultaneous with training saturation.

---

## Comparison: weight decay vs dropout vs label smoothing

| intervention | mechanism | direction | max effect |
|---|---|---|---|
| dropout ↑ | disrupts memorization shortcuts (stochastic) | ↓ t_grok | ~40% speedup |
| label smoothing ↑ | softens target signal | ↑ t_grok | ~89% slowdown |
| weight decay ↑ | penalizes large weights (structural) | ↓ t_grok | **65× speedup** |

Weight decay dominates by 2 orders of magnitude. This is consistent with Power et al. 2022
who showed wd as the primary grokking control variable, but our sweep shows the full range
more cleanly.

### Power law fit

Plotting log(t_grok) vs log(wd):

| log(wd) | log(t_grok) |
|---|---|
| -2.00 (wd=0.01) | 4.47 |
| -1.00 (wd=0.10) | 3.97 |
| -0.52 (wd=0.30) | 3.61 |
| 0.00 (wd=1.0) | 3.12 |
| 0.48 (wd=3.0) | 2.65 |

Linear regression slope ≈ -0.73. Approximate fit: **t_grok ∝ wd^(-0.73)**.

Not a clean integer power law, but a strong negative power law over 3 orders of magnitude.

---

## Interpretation for Paper 04 framework

### Three distinct mechanisms, three distinct variables

The experiments across Phase 4 and Phase 5 reveal that regularization is not monolithic.
Three interventions map to three different components of the emergence inequality
L·μ ≤ log(σ):

**Weight decay → reduces effective L (circuit complexity)**
High wd penalizes large weights, forcing distributed representations. Complex memorization
circuits (high L) cannot form — they would be penalized too heavily. The generalization
circuit (lower L, exploiting algebraic structure) becomes the only viable attractor.
Mechanism: structural pressure reducing circuit length, not noise.

**Dropout → raises μ for memorization circuits specifically**
Dropout randomly zeroes activations, disrupting the sharp, specific patterns that
characterize memorization circuits. The generalization circuit (distributed, redundant)
is more robust to dropout. Effective noise μ is higher for memorization than for
generalization. Net effect: the ratio log(σ)/μ is higher for the generalization circuit.
Mechanism: noise injection that differentially damages memorization.

**Label smoothing → reduces σ (effective signal strength)**
Label smoothing makes targets probabilistic rather than discrete. For modular arithmetic
(a+b mod p), the correct answer is uniquely determined — label smoothing artificially
introduces uncertainty about the target. The effective advantage of the generalization
circuit (which correctly identifies the unique answer) is reduced. σ falls.
Mechanism: signal degradation, not noise or complexity.

### Predicted Paper 04 v0.2 claim

The emergence inequality L·μ ≤ log(σ) — where L is circuit complexity, μ is effective
noise per circuit bit, and σ is the selective advantage of the target circuit — can be
modulated independently via three regularization axes:

- wd controls L
- dropout controls μ (differentially for memorization vs generalization)
- label smoothing controls σ

This makes the framework more testable: each axis should have distinct experimental
signatures, and we now have data for all three.

---

## Robustness caveats (for reviewers)

1. **N=2 for wd=3.0 clean**: one of 3 seeds hit false positive. More seeds at wd=3.0
   needed to characterize the stability of the near-zero gap regime.

2. **Single p=97**: all regularization experiments are on one modulus. The power law
   should be replicated at p=31 or p=149 to confirm it's not p-specific.

3. **Threshold sensitivity**: the t_grok measure depends on a 95% threshold. The outliers
   (baseline s1, wd3.0 s1) suggest momentary threshold crossings that don't persist.
   A stricter criterion (e.g., sustained 95% for 200 steps) would filter these automatically
   and raise the reported means slightly.

4. **Single architecture**: all experiments use the same transformer (depth=1, same d_model).
   Whether the power law t_grok ∝ wd^(-0.73) holds across architectures is unknown.

---

## Recommended next actions

**For Paper 04 v0.2 (do now):**
- Update baseline mean from 1267 to 1314 (N=7, with exclusion criterion documented).
- Add Section: "Decomposition of regularization effects on grokking speed" with the three-
  mechanism framework (wd→L, dropout→μ, ls→σ).
- Add Figure: weight_decay sweep curve (log-log plot, power law fit).
- Add Figure: three-panel comparison (wd, dropout, ls) showing opposite directions.

**For Phase 6 (optional, would strengthen paper):**
- Replicate wd sweep at p=31 and p=149 (9 more runs each, ~3 hrs each).
- Test wd=5.0 and wd=10.0 to find if curve saturates.
- Add sustained-threshold criterion to reduce outlier sensitivity.

---

*Phase 5 complete. All 17 runs successful. Report generated 2026-04-26.*
