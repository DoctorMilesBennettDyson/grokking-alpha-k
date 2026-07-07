# Pre-Registration — Experiments P1 and P4 (Phase 9)

**Date of signing:** 2026-07-07
**Author:** Pablo L. Rainieri Blasco (execution assistant: Claude / Fable 5)
**Status:** FIXED. The public timestamp of this document is the git commit hash of the
commit that first adds this file to the `grokking-alpha-k` repository, pushed to GitHub
**before** any P1/P4 run is launched. Later modification only via dated changelog below.

**Context.** `theory/theory_extension_v0.2.md` §1 documents that the previous theory
pre-registration (v0.1 §4.3) was found compromised (predictions numerically match a fit
using data they claimed to predate). This document is the replacement confirmatory
instrument, written under stricter rules:

1. The prediction-generating model and its parameters are committed as code
   (`code/fit_alpha_K.py`) **before** data collection.
2. This file is pushed publicly before the runner starts.
3. All runs are reported (clean, outlier, DNF), per the analysis rules of
   `prereg/PRE_REGISTRATION_FASE_II.md`, which are re-used verbatim (sustained
   t_grok = first step with test_acc ≥ 0.95 held 3 consecutive evals; outlier if
   final_test_acc < 0.95 or t_sustained > t_first + 1000; DNF excluded from slopes;
   log-log regression over clean cell means).

---

## Motivation (from theory_extension_v0.2 §3.4 and §4)

Phase 8's H4 result (t_grok monotonically increasing in batch size) contradicts the
v0.1 reading of σ as gradient SNR and motivates the v0.2 refinement, eq. (10):

```
log t_grok ≈ L·μ_0(L)·exp(-γ·K/L) / f(T),   T = lr / bs,   f increasing
```

Two falsifiable consequences are tested here. **Setup common to all runs:** modular
addition, p=97, depth=1, n_layers=2, d_model=128, AdamW (betas 0.9/0.98), Xavier init,
steps=200,000, eval_every=100, train_frac=0.5, warmup=100, seeds {0, 1, 2},
train_v3.py engine, RTX 3070, PyTorch 2.4.1+cu118.

---

## P1 — Learning-rate mirror of the batch-size effect

**Hypothesis:** if the noise term is a temperature T = lr/bs, then at fixed bs raising
lr must act like lowering bs: **t_grok decreases monotonically with lr.**

**Runs (9):** bs=512, wd=1.0, lr ∈ {5e-4, 1e-3, 2e-3} × seeds {0,1,2}.
Run names: `P1_lr{lr}_s{seed}`. (The lr=1e-3 cell replicates the Phase 4/5 baseline,
1314 ± 195; it doubles as an internal replication check.)

**Pre-registered acceptance (qualitative, binding):**
mean t_grok_sustained(5e-4) > mean(1e-3) > mean(2e-3), computed over clean runs.

**Pre-registered quantitative bands (exploratory, non-binding):** scaled from the
observed bs-doubling effects (+23%, +12%):
- t(2e-3) / t(1e-3) ∈ [0.65, 0.90]
- t(5e-4) / t(1e-3) ∈ [1.10, 1.55]

**Falsification:** ordering violated (or reversed) with ≥2 clean seeds per cell →
the temperature reading of eq. (10) is wrong for lr, and the H4 result must be
attributed to a bs-specific mechanism (e.g., steps-per-epoch coverage), not to T.
If lr=2e-3 diverges systematically (≥2/3 seeds), the cell is reported DNF and the
test is declared **inconclusive at this lr range** — not rescued by picking a new lr.

## P4 — Separability: the wd exponent α must not depend on batch size

**Hypothesis:** in eq. (10), T multiplies the whole numerator, so temperature shifts
t_grok's *level* but not the wd exponent. **α measured at bs=256 equals α at bs=1024
within tolerance.**

**Runs (24 new):** lr=1e-3, bs ∈ {256, 1024} × wd ∈ {0.01, 0.1, 0.3, 3.0} × seeds
{0,1,2}. Run names: `P4_bs{bs}_wd{wd}_s{seed}`.
**Declared reuse:** the wd=1.0 cells at bs=256 and bs=1024 are taken from Phase 8 H4
(`results/phase8/H4_*`, 3 clean seeds each, same procedure, collected 2026-07-06 —
before this pre-registration, declared here rather than re-run).

**Pre-registered acceptance (binding):**
|α(bs=256) − α(bs=1024)| ≤ 0.15, each α from the log-log regression over its five
clean wd-cell means. Reference: α(bs=512) = −0.731 (Phases 4-7).

**Falsification:** |Δα| > 0.15 → eq. (10)'s separability is wrong; wd and batch size
interact; the framework requires an explicit interaction term and the paper reports
the temperature refinement as refuted at first test.

**Secondary (descriptive, non-binding):** whether each of α(256), α(1024) falls inside
[−0.94, −0.54], and the sign of Δα.

---

## Decision table (fixed now)

| P1 | P4 | Interpretation for the paper |
|---|---|---|
| pass | pass | Temperature reading survives its first two tests; eq. (10) promoted to "supported working model" (still not confirmed mechanism) |
| pass | fail | T affects level via lr as predicted, but separability fails → interaction model needed |
| fail | pass | bs effect is not temperature; likely epoch-coverage artifact → σ/T term dropped entirely |
| fail | fail | v0.2 refinement refuted; paper reports the empirical maps only |

No outcome changes the empirical α(K) results; this instrument tests **only** the
v0.2 theoretical refinement.

## Execution

- Runner: `code/phase9.py` (idempotent, crash-safe, same pattern as phase8.py).
  Order: P1 (9 runs, ~3 h) → P4 bs=256 (12 runs, ~4 h) → P4 bs=1024 (12 runs, ~7 h).
- Total ≈ 14 h wall-clock on the RTX 3070.
- Auto-analysis writes `results/phase9/REPORT.md` evaluating exactly the criteria above.

## Changelog

(none)
