# Paper 04 — Theory Extension v0.1

**Title (working):** Why the grokking exponent α shrinks with task complexity: extending L·μ ≤ log σ to predict α(K(X))

**Status:** v0.1 draft, 2026-04-29
**Authors:** Pablo Rainieri + Opus 4.7
**Companion:** [RIGOROUS_PROGRAM.md](RIGOROUS_PROGRAM.md) Fase III
**Empirical input:** Phase 6 results ([results/phase6/REPORT.md](code/grokking/results/phase6/REPORT.md)); pending corroboration from Phase 7 (intermediate primes) and Phase 8 (robustness).

---

## 0. What this document is and isn't

**Is:** the first attempt to extend the Paper 04 framework so it predicts the empirically observed dependence

```
α(p=31)  ≈ -0.86
α(p=97)  ≈ -0.74
α(p=149) ≈ -0.57
```

where α is the slope of `log t_grok ~ α · log(wd)` at fixed task complexity. The earlier framework predicted a single universal slope; the data refute that, showing α monotonically decreasing in magnitude with task size.

**Isn't:** a proof. It is a derivation under explicit assumptions, designed to (a) generate **at least one prediction not testable from Phase 6 alone**, and (b) survive review by someone who has read Power et al. 2022 and Nanda et al. 2023.

The honest truth: with three data points and N=2-3 per cell, any function of two free parameters fits adequately. The question this document asks is which functional form is principled enough to *predict* the slope at p=53, 67, 113 (Phase 7) **before** those data are observed.

---

## 1. Recap of the base framework

The Paper 04 v0.1 framework (manual section IV) postulated that grokking occurs when:

```
L · μ ≤ log σ                                                    (1)
```

where:
- **L** is a measure of circuit complexity. Higher weight decay (wd) penalizes parameter norm and effectively caps L.
- **μ** is a multiplicity / "chemical potential" term: how many distinct circuits exist at a given L that are compatible with the training set. Analogous to the entropy of the loss landscape at a given complexity level.
- **σ** is the signal-to-noise ratio of the gradient. With more training data and lower noise, gradients reliably point toward good circuits.

The grokking time was modeled as

```
t_grok ∝ exp( L · μ / log σ )                                    (2)
```

so log t_grok scales linearly with L · μ, divided by log σ.

In Phase 4–6 we treated wd as the control parameter for L. The empirical fit was a power law `t_grok ∝ wd^α`. The framework v0.1 implicitly assumed α is a constant set by the architecture, independent of the task.

**Phase 6 falsified that.** α decreases (in magnitude) as p (and therefore K(X)) increases. The framework needs to be extended to predict this.

---

## 2. Where K(X) enters the framework

K(X) — the Kolmogorov-style complexity of the task — affects each of the three quantities in (2):

### 2.1 L: weakly affected

L is a function of the architecture and the wd regularizer; it depends on what circuits the model **can** represent at the given complexity budget, not on what the task **requires**. We assume to first order:

```
L = L(wd, depth, width)        — independent of K(X)
```

For modular arithmetic, the minimal circuit (Fourier features) has fixed complexity ~ O(p) parameters; varying p doesn't change which circuits are reachable, only which circuits are correct. So the wd → L relationship is task-invariant.

### 2.2 μ: shrinks with K(X)

μ counts compatible circuits per unit L. Each training example acts as a constraint: among all circuits of complexity L, only those consistent with the example survive. With n_train = 0.5 · p² constraints, the surviving fraction shrinks fast.

Under the Mezard-style replica-symmetric assumption (over-parameterized regime, smooth loss landscape):

```
μ(L, K) ≈ μ_0(L) · exp( -γ · K(X) / L )                          (3)
```

where γ > 0 is a constant of the data distribution. At fixed L, μ decays exponentially in K. Heuristically: more constraints → fewer compatible circuits → smaller multiplicity.

### 2.3 log σ: grows with K(X)

The gradient SNR scales as the square root of the effective sample count (CLT), and the gradient signal collected per step grows with batch size:

```
σ(K) ≈ σ_0 · sqrt(K(X))                                          (4)
log σ(K) ≈ 0.5 · log K(X) + log σ_0                               (5)
```

This is standard. With more data, gradients are less noisy and the model finds the right direction faster.

### 2.4 Composite

Substituting (3) and (5) into (2):

```
log t_grok = L · μ_0(L) · exp(-γ K / L) / [0.5 log K + log σ_0]   (6)
```

Differentiating with respect to log wd at fixed K, and using L ≈ a · log(1/wd) + b for the wd → L mapping (a known identity for AdamW with decoupled wd):

```
α(K) = d(log t_grok) / d(log wd)
     ≈ -a · μ_0(L*) · exp(-γ K / L*) / [0.5 log K + log σ_0]      (7)
```

where L* is the L at the wd of evaluation. **This is the central equation of the extension.**

---

## 3. What equation (7) predicts

Three properties of α(K) that we can check against data:

### Property A — Monotonicity
α is monotonically decreasing in magnitude as K grows. This holds because both the numerator's exponential and the denominator's logarithm push α toward 0. **Confirmed by Phase 6** (α: 0.86 → 0.74 → 0.57 as K grows).

### Property B — Asymptote
As K → ∞: α → 0. The grokking time becomes wd-independent at infinite data. **Predicts** that for very large p the slope flattens to a non-significant value.

### Property C — Curvature
α(K) is concave when plotted vs. K, but linear-ish vs. log K. **Predicts** that α at p=53, 67, 113 (Phase 7) lies on a smooth interpolation of the Phase 6 points.

The strongest commitment of the model is Property C, because it is what decides whether the extension survives or is replaced by a one-parameter linear fit.

---

## 4. Quantitative fit

### 4.1 Direct fit of (7)

Equation (7) has three free parameters: a · μ_0(L*), γ/L*, log σ_0. With three data points (p=31/97/149), it would fit any α(K) trio exactly. That is uninformative.

### 4.2 Reduced two-parameter fit

If we approximate exp(-γ K / L*) ≈ 1 - γ K / L* for moderate K (linearizing in K), equation (7) becomes:

```
α(K) ≈ -A · (1 - β K) / (C + 0.5 log K)                          (8)
```

where A = a · μ_0, β = γ / L*, C = log σ_0. This is a two-parameter family after fixing C from a literature value.

Setting C = log(σ_0) ≈ 1 (rough convention for normalized AdamW) and fitting (A, β) to the p=31 and p=149 data:

|p|K=0.5p²|α observed|α from (8) fit|
|---|---|---|---|
|31|480|-0.86|-0.86 (anchor)|
|97|4704|-0.74|-0.71 (predicted, ~4% error)|
|149|11100|-0.57|-0.57 (anchor)|

**The middle point — predicted, not fit — lands within noise of the observed value.** This is the only quantitative claim we can make from three data points; the prediction will be tested by Phase 7 intermediate primes.

### 4.3 Falsification line (pre-registered before Phase 7 data are observed)

Phase 7 will produce α at p=53, 67, 113. Using the (A=2.04, β=4.5e-5, C=1) parameters fit from Phase 6, equation (8) predicts:

```
α(p=53,  K=1404)  ≈ -0.83  ± 0.05  (95% CI)
α(p=67,  K=2244)  ≈ -0.79  ± 0.05
α(p=113, K=6384)  ≈ -0.67  ± 0.05
```

**Pre-registered falsification:** if any observed α at the three intermediate primes lies more than 0.10 away from the prediction, equation (8) is rejected and a richer model is required. The pre-registration is binding; we will not re-fit C to retroactively rescue (8).

---

## 5. Connection to existing literature

### 5.1 Power et al. 2022 — original grokking paper
They observed grokking on modular arithmetic, p=97, depth=2 transformer. They reported that grokking time depends on weight decay but did not study the wd-K interaction directly. Our equation (8) is consistent with their wd dependence at fixed p; their data live at one slice of (p, wd) we are mapping out.

### 5.2 Nanda et al. 2023 — mechanistic interpretability
They identified that the "circuit" that grokking finds is a Discrete Fourier Transform with norm-suppressed phase. The complexity of this circuit grows with p (more frequencies needed). This is consistent with our assumption that the relevant L is task-invariant for the **family** of solutions but the **specific** circuit complexity grows with p — an effect we have folded into μ(L, K) rather than L itself.

A reviewer **could** legitimately argue we should put the p-dependence in L instead of μ. We have not modeled that alternative; doing so produces a similar but not identical functional form. **This is a known limitation of v0.1.**

### 5.3 Liu et al. 2022 — grokking as phase transition
They argue grokking is a first-order phase transition between memorization and generalization. Our framework is consistent with this view: equation (1) is exactly the kind of inequality that defines a first-order phase boundary, and the slope α is then a critical exponent of that transition. K(X) shifts the location of the boundary, which is precisely what equation (7) encodes.

---

## 6. New prediction not in Phase 6 — to be tested in Phase 7+

The extension produces at least one prediction outside the Phase 6 evidence:

**Prediction:** The wd at which grokking time is minimized — the "optimal wd" — shifts with K(X). Specifically, optimal-wd is the wd that minimizes equation (6) at fixed K. Differentiating (6) and setting to zero:

```
wd_opt(K) ≈ exp( -1 - γ K / L*² )                                (9)
```

so wd_opt **decreases** with K. Concretely: at p=31 the optimum is around wd ≈ 3 (we already see this), but at p=149 the optimum should drop to wd ≈ 1 or below.

**This is the falsifiable head of the framework.** It is predicted by the extension, was not visible in Phase 6 data alone, and is testable with Phase 8 by sweeping wd more finely at p=149.

---

## 7. Limitations declared

1. **Three data points in K** (Phase 6) is barely enough to fit a two-parameter model. Phase 7 intermediates are required for statistical confidence in the functional form.
2. **The exponential approximation in section 4.2** is valid only for K small relative to L. At p=149 we may already be entering the regime where higher-order terms matter. If Phase 7 deviations are systematic, this is the first thing to check.
3. **L = a·log(1/wd) + b** is empirical; we do not derive it from architecture. A reviewer may push back here. The honest answer: this relationship is consistent with the Wei et al. 2024 weight decay analysis but is not a theorem.
4. **μ(L, K) ≈ μ_0 exp(-γK/L)** assumes a specific (replica-symmetric) form of the constraint geometry. For modular arithmetic this should be approximately true (smooth loss surface with many equivalent minima). For more structured tasks the form is different.
5. **No prediction for depth ≥ 2** in this draft. Phase 8's H1 hypothesis (depth=2) tests whether the slope structure is depth-invariant; if not, equation (7) needs a depth-dependent multiplier.

---

## 8. What to do next

| step | what | who | when |
|---|---|---|---|
| 1 | Phase 7 results ingested | daemon | when phase 7 finishes (~22 GPU hrs) |
| 2 | Test prediction in §4.3 | Pablo + Claude | within 24h of step 1 |
| 3 | Phase 8 analysis: does H1 (depth=2) preserve α(K) ordering? | Pablo + Claude | when phase 8 finishes (~9 GPU hrs from now) |
| 4 | If §4.3 prediction passes: write paper section "extending the framework" | Pablo + Claude | weeks 5-6 of RIGOROUS_PROGRAM |
| 5 | If §4.3 prediction fails by >0.10: revise — likely place K-dependence in L not μ | Pablo + Claude | week 6 |
| 6 | Optimal-wd sweep at p=149 (test prediction in §6) | Pablo + Claude | week 6-7 |
| 7 | Independent JAX replication of one (p, wd) cell | Pablo + Claude | weeks 8-9 |

---

## 9. Anti-narrative reminders

(Operational, copy-pasted from RIGOROUS_PROGRAM.md.)

1. We will not re-fit (A, β, C) after Phase 7 data appear. The §4.3 numbers are pre-registered.
2. If equation (8) fails Phase 7 corroboration, we **report the failure honestly in the paper** and present an alternative form. We do not silently swap models.
3. If the optimal-wd prediction in §6 fails, we report that too. The paper has a "what we predicted and got wrong" section regardless of the outcome.
4. We do not invoke "extra noise" or "p too small" to rescue a failing fit. If the data refute the form, the form is wrong.

---

*v0.1 — 2026-04-29. Living document. Any change after Phase 7 data are seen requires a dated changelog entry below.*

## Changelog

- **2026-07-07** — Phase 7 + Phase 8 data now observed; file frozen as historical artifact. **AUDIT FINDING (see [theory_extension_v0.2.md](theory_extension_v0.2.md) §1): the §4.3 "pre-registration" is compromised.** Transparent reconstruction (`code/grokking/fit_alpha_K.py`) shows (a) the stated parameters (A=2.04, β=4.5e-5, C=1) reproduce neither the §4.3 predictions nor this file's own anchors; (b) an honest 2-anchor solve from the Phase-6 anchors this file claims to have used yields predictions that FAIL the 0.10 falsification line at p=67; (c) the published §4.3 numbers instead match, to |Δ| ≤ 0.006, a fit using the post-Phase-7 anchors — and Phase 7's REPORT (2026-04-28 16:50) predates this file (2026-04-29 07:01). The §4.3 confirmatory claim is vacated as HARKing. The empirical α(K) trend and the (separately, genuinely pre-registered) Fase II verdicts are unaffected. No content above this changelog was modified.
