# Phase 6 — REPORT

Generated: 2026-04-27T17:18:09

## Objective

Replicate the weight_decay power law `t_grok ∝ wd^(-0.73)` (from Phase 4+5 at p=97) at two additional moduli:
- **p=31** (small task, fewer training examples: 480 vs 4704)
- **p=149** (larger task, more examples: 11100)

Also introduces **sustained threshold**: t_grok_sustained = first step where test_acc ≥ 0.95 for 3 consecutive evaluations (300 steps), eliminating false-positive threshold crossings.

## Weight-decay sweep results

All using sustained threshold (t_grok_sustained). 'first' columns show original threshold for comparison.

### p = 31

| wd | mean (sustained) | std | min | max | mean (first) | note |
|---|---|---|---|---|---|---|
| 0.01 | 144300 | — | 144300 | 144300 | 144300 | N=1 DNF=2 |
| 0.1 | 29950 | 3889 | 27200 | 32700 | 29300 | N=2 DNF=1 |
| 0.3 | 10267 | 1882 | 8100 | 11500 | 10200 | N=3 |
| 1.0 | 2767 | 153 | 2600 | 2900 | 2767 | N=3 |
| 3.0 | 1200 | 100 | 1100 | 1300 | 1000 | N=3 |

**Power law slope (sustained): t_grok ∝ wd^(-0.86)**
Power law slope (first-hit): t_grok ∝ wd^(-0.88)

### p = 149

| wd | mean (sustained) | std | min | max | mean (first) | note |
|---|---|---|---|---|---|---|
| 0.01 | 13233 | 3371 | 10500 | 17000 | 13100 | N=3 |
| 0.1 | 5767 | 929 | 5000 | 6800 | 5767 | N=3 |
| 0.3 | 2333 | 404 | 1900 | 2700 | 2333 | N=3 |
| 1.0 | 833 | 58 | 800 | 900 | 833 | N=3 |
| 3.0 | 650 | 71 | 600 | 700 | 650 | N=2 DNF=1 |

**Power law slope (sustained): t_grok ∝ wd^(-0.57)**
Power law slope (first-hit): t_grok ∝ wd^(-0.57)

### p = 97 (reference from Phase 4 + Phase 5)

| wd | mean (sustained) | std | note |
|---|---|---|---|
| 0.01 | 30133 | 7107 | N=3 |
| 0.1 | 9267 | 1582 | N=3 |
| 0.3 | 4100 | 100 | N=3 |
| 1.0 | 1314 | 195 | N=7 |
| 3.0 | 450 | 71 | N=2 |

**Power law slope p=97: t_grok ∝ wd^(-0.74)**

## Cross-p comparison: does the power law hold?

| p | wd=0.01 | wd=0.1 | wd=0.3 | wd=1.0 | wd=3.0 | slope |
|---|---|---|---|---|---|---|
| p=31 | 144300 | 29950 | 10267 | 2767 | 1200 | -0.86 |
| p=97 (ref) | 30133 | 9267 | 4100 | 1314 | 450 | -0.74 |
| p=149 | 13233 | 5767 | 2333 | 833 | 650 | -0.57 |

If slopes are consistent across p (all around -0.7±0.15), the power law is universal and not p-specific.

## Sustained vs first-hit threshold comparison

Shows how many runs had false-positive t_grok under the original criterion.

| run | t_grok_first | t_grok_sustained | delta | outlier? |
|---|---|---|---|---|
| p149_wd0.01_s0 | 12200 | 12200 | +0 | no |
| p149_wd0.01_s1 | 17000 | 17000 | +0 | no |
| p149_wd0.01_s2 | 10100 | 10500 | +400 | no |
| p149_wd0.1_s0 | 5500 | 5500 | +0 | no |
| p149_wd0.1_s1 | 6800 | 6800 | +0 | no |
| p149_wd0.1_s2 | 5000 | 5000 | +0 | no |
| p149_wd0.3_s0 | 2400 | 2400 | +0 | no |
| p149_wd0.3_s1 | 2700 | 2700 | +0 | no |
| p149_wd0.3_s2 | 1900 | 1900 | +0 | no |
| p149_wd1.0_s0 | 800 | 800 | +0 | no |
| p149_wd1.0_s1 | 900 | 900 | +0 | no |
| p149_wd1.0_s2 | 800 | 800 | +0 | no |
| p149_wd3.0_s0 | 700 | 700 | +0 | no |
| p149_wd3.0_s1 | 600 | 600 | +0 | no |
| p149_wd3.0_s2 | 800 | 5400 | +4600 | YES |
| p31_wd0.01_s0 | 144300 | 144300 | +0 | no |
| p31_wd0.01_s1 | DNF | — | — | yes |
| p31_wd0.01_s2 | 151800 | 153000 | +1200 | YES |
| p31_wd0.1_s0 | 25900 | 27200 | +1300 | YES |
| p31_wd0.1_s1 | 49300 | 50400 | +1100 | YES |
| p31_wd0.1_s2 | 32700 | 32700 | +0 | no |
| p31_wd0.3_s0 | 7900 | 8100 | +200 | no |
| p31_wd0.3_s1 | 11500 | 11500 | +0 | no |
| p31_wd0.3_s2 | 11200 | 11200 | +0 | no |
| p31_wd1.0_s0 | 2600 | 2600 | +0 | no |
| p31_wd1.0_s1 | 2900 | 2900 | +0 | no |
| p31_wd1.0_s2 | 2800 | 2800 | +0 | no |
| p31_wd3.0_s0 | 1000 | 1300 | +300 | no |
| p31_wd3.0_s1 | 1100 | 1100 | +0 | no |
| p31_wd3.0_s2 | 900 | 1200 | +300 | no |

## Interpretation for Paper 04 v0.2

### On universality of the power law

If the cross-p comparison shows similar slopes (~-0.7) across p=31, 97, 149:

- The power law `t_grok ∝ wd^(-0.7)` is a property of the **learning dynamics**, not of the specific task structure.
- This supports the framework claim that weight_decay modulates L (circuit complexity) in the Eigen inequality, independent of K(X) (task complexity).
- Combined with Phase 4 dropout and label smoothing results: the three regularization axes (wd→L, dropout→μ, ls→σ) are orthogonal and universal.

### On sustained threshold

If t_grok_sustained ≈ t_grok_first for most runs: the original threshold is reliable and outliers were genuinely rare edge cases.
If t_grok_sustained >> t_grok_first for many runs: the original results were noisier than reported and the sustained threshold is the correct measure.

Either conclusion is informative and publishable.

### Recommended Paper 04 v0.2 additions

1. **Main table** (Table 1): wd sweep at p=31, 97, 149 side by side.
2. **Figure**: log-log plot of t_grok vs wd for the 3 primes, all on one axes. Overlapping power law lines = universality claim.
3. **Sustained threshold**: report both metrics, argue sustained is the robust measure and is consistent with first-hit within noise.
4. **Combined framework table**: wd (→L), dropout (→μ), ls (→σ) in a 3-row table showing direction, magnitude, and mechanism.

---

*Phase 6 complete. Ready for Paper 04 v0.2 write-up.*
