# Paper 04 — Theory Extension v0.2

**Title (working):** Why the grokking exponent α shrinks with task complexity: extending L·μ ≤ log σ to predict α(K(X))

**Status:** v0.2, 2026-07-07 — evaluation of v0.1 pre-registered predictions against Phase 7 data + integration of Phase 8 (Fase II) robustness verdicts.
**Authors:** Pablo Rainieri + Opus 4.7 (v0.1) + Fable 5 (v0.2 evaluation)
**Relation to v0.1:** [theory_extension_v0.1.md](theory_extension_v0.1.md) is **frozen** as the pre-registration artifact. This document evaluates its §4.3 predictions and extends the framework with Fase II evidence. Per v0.1 §9, parameters (A, β, C) are **not** re-fit.

---

## 1. Verdict on the v0.1 §4.3 predictions: **PRE-REGISTRATION COMPROMISED — confirmatory claim VACATED**

v0.1 §4.3 claimed to predict α at the three intermediate primes *"before Phase 7 data are observed"*, with a binding falsification line of 0.10. Against the published numbers, the comparison is:

| p | K = n_train | α published (§4.3) | α observed (Phase 7) | Δ | within 0.10? |
|---|---|---|---|---|---|
| 53 | 1404 | -0.83 ± 0.05 | **-0.866** | 0.036 | yes |
| 67 | 2244 | -0.79 ± 0.05 | **-0.861** | 0.071 | yes |
| 113 | 6384 | -0.67 ± 0.05 | **-0.736** | 0.066 | yes |

Nominally 3/3 PASS. **However, the transparent reconstruction (`code/grokking/fit_alpha_K.py`, written 2026-07-07) vacates this as a confirmatory result:**

1. **The stated parameters don't reproduce anything.** (A=2.04, β=4.5e-5, C=1) fails to reproduce the §4.3 predictions *and* fails to reproduce v0.1's own anchors, under both log10 and ln conventions. No fit script was saved.
2. **A clean re-derivation from the Phase-6 anchors v0.1 claims to have used does NOT yield the published numbers.** Exact 2-anchor solve (log10, C=1): predictions -0.772 / -0.734 / -0.638 — deviations from §4.3 of 0.058 / 0.056 / 0.032. Under this honest reconstruction the falsification test **FAILS at p=67** (|pred-obs| = 0.127 > 0.10); under the ln convention it fails at all three points (worst 0.147).
3. **The published numbers match a fit that uses the post-Phase-7 anchors.** Exact 2-anchor solve from the *updated* slopes (p=31: -0.926, p=149: -0.577 — values that only exist with Phase 7's added seeds) yields -0.828 / -0.784 / -0.666, matching the published §4.3 numbers to **|Δ| ≤ 0.006 on all three points**.
4. **The timeline permits contamination.** Phase 7's REPORT was generated 2026-04-28 16:50; `theory_extension_v0.1.md` is dated 2026-04-29 (file mtime 07:01). The data the document claims not to have seen existed on disk ~14 hours before the document was written.

**Conclusion (2)+(3)+(4):** the §4.3 "predictions" are numerically indistinguishable from a post-hoc fit to data that already included Phase 7, mislabeled as a pre-registration. This is HARKing per the project's own methodology standards (Doc 128 Anexo A; Manual 139). The claim "eq. (8) survived pre-registered falsification" **must not appear in the paper**. What legitimately remains:

- eq. (8) as a **descriptive 2-parameter fit**: from Phase-6 anchors it tracks the 6-point curve with residuals 0.03–0.13 — adequate as description, zero confirmatory weight.
- The **empirical α(K) decreasing trend itself** (Section 2), which does not depend on eq. (8) at all.
- The **Fase II pre-registration is clean and unaffected**: `PRE_REGISTRATION_FASE_II.md` was signed 2026-04-27 and last touched 2026-04-28 21:55, three minutes before Phase 8 execution began (21:58) and months before the 2026-07 runs. H1/H3/H4 verdicts stand as genuine pre-registered tests.
- Confirmatory standing for the theory must now come from the **new predictions in §5 (P1–P4)**, pre-registered externally (OSF) before any run, with the fit script committed first.

**Secondary observations (subordinate to the above):** all three observed slopes are steeper than every Phase-6-anchored reconstruction (systematic bias ~0.06–0.10, same sign), consistent with the linearization exp(-γK/L) ≈ 1-γK degrading with K; and the p=31 anchor is fragile (its wd=0.01 cell is 3/8 clean, slope moved -0.86 → -0.926 with added seeds).

## 2. Updated empirical picture: slope(p) with 6 points (Phase 7)

| p | n_train | slope (log-log) |
|---|---|---|
| 31 | 480 | -0.926 |
| 53 | 1404 | -0.866 |
| 67 | 2244 | -0.861 |
| 97 | 4704 | -0.731 |
| 113 | 6384 | -0.736 |
| 149 | 11100 | -0.577 |

**Property A (monotonicity, v0.1 §3):** the broad decreasing trend in |α| holds, but strict monotonicity does **not**: there are two local plateaus (53→67: Δ=0.005; 97→113: Δ=-0.005, a tiny reversal). Both are far inside cell-level noise (N=3/cell). Honest statement for the paper: *monotone trend confirmed at the resolution available; strict monotonicity unresolved at N=3.* Phase 7's own REPORT flagged this ("slope(p) tiene estructura no trivial").

**Property B (asymptote α→0):** untested; requires p ≥ 197. Remains a forward prediction.

**Property C (smooth interpolation):** supported within the 0.10 tolerance; weakened by the systematic bias of §1.1.

## 3. What Fase II (Phase 8, completed 2026-07-07) does to the framework

Full data: `code/grokking/results/phase8/REPORT.md`; diagnostic trail: `results/phase8_debug/` + PRE_REGISTRATION_FASE_II.md changelog 2026-07-06.

### 3.1 H3 (init: Kaiming/Xavier ratio 1.28 ∈ [0.5, 1.5]) — supports the framework
t_grok level shifts by +28% under Kaiming; slope structure untouched. The mechanism is not an initialization artifact. Additionally, Xavier seeds 20-24 replicated the Phase 4+5 baseline (1280±84 vs 1314±195) — an independent-seed replication of the anchor cell.

### 3.2 H1 (task composition, depth=2: slope -0.658 ∈ [-0.94, -0.54]) — confirmed *numerically*, with a regime caveat that matters theoretically
In every clean depth-2 cell, `t_train_saturated == t_grok` and train/test accuracy track each other throughout: **there is no memorization phase, hence no grokking dynamic** (n_train=250K ≈ 0.6× parameter count; memorization is impossible). Yet the wd power law survives with a slope inside the pre-registered interval.

**Theoretical implication (new, v0.2):** eq. (2) models t_grok as the time for circuit *selection* after memorization. If the same power law appears in a regime with no memorization phase, then either (a) the wd → convergence-time power law is generic to regularized training and not grokking-specific — which would *weaken* the claim that α is a grokking critical exponent — or (b) the circuit-selection mechanism operates identically whether or not a memorization solution coexists. Distinguishing (a) from (b) is now an open question of the framework. The paper must not present H1 as "grokking is depth-invariant"; it must present it as "the slope survives a regime change, which is itself informative."

### 3.3 H2 (SGD: 15/15 cells DNF at chance) — bounds the framework's validity domain
Mechanism established by differential diagnostic (2026-07-06): coupled L2 + momentum at the pre-registered (lr, wd) values collapses the weight norm (42 → 0.02 within 1K steps) before gradient signal can compete; decoupled SGDW at wd=1.0 also collapses. SGD with wd=0 memorizes normally — the training path is healthy.

**Implication:** the mapping L ≈ a·log(1/wd) + b (v0.1 §7.3) is **specific to adaptive optimizers with decoupled decay**. In plain SGD, wd does not modulate circuit complexity — it destroys the network. Equation (7)'s scope is hereby bounded: *α(K) is defined for AdamW-class optimizers.* This is reported as a scoped claim, not rescued as universal.

### 3.4 H4 (batch size: t_grok = 1067 → 1314 → 1467, monotonic ↑) — **exposes a sign error in the naive σ reading**
v0.1 §2.3 treats σ as gradient SNR and places log σ in the denominator of eq. (2): higher SNR → faster grokking. If minibatch SNR ∝ √bs, eq. (2) predicts larger batch → **faster** grokking. Observed: larger batch → **slower** (monotonic, 3/3 clean per cell).

The pre-registered H4 hypothesis itself anticipated the observed direction via the *noise-as-exploration* reading (smaller batch → more SGD noise → faster grokking). The data therefore force a reinterpretation:

- **σ in eq. (2) must be split into two distinct quantities:**
  - `σ_data` — signal from dataset size K. More data → fewer compatible wrong circuits → faster grokking. Supported by the master table: at fixed wd=1.0, t_grok falls monotonically with p (2767 → 833 across the 6 primes). This effect can live entirely in μ(L, K); the σ ∝ √K assumption (v0.1 eq. 4) is redundant with μ and can be dropped.
  - `T_sgd ∝ lr/bs` — minibatch temperature. Higher temperature → faster escape toward the generalizing circuit (Langevin-style basin escape). This term enters with the **opposite sign** to the SNR reading.
- Quantitatively: per-doubling t_grok ratios are +23% (256→512, inside the pre-registered 20-80% band) and +12% (512→1024, **below** the band). The increments *halve* per doubling — inconsistent with a pure exp(c·bs) escape model, closer to a saturating power law. N=3/cell is too sparse to fit; reported as-is.

**This is the most theoretically productive result of Fase II:** it corrects the functional form before the paper overcommits to it.

## 4. v0.2 candidate refinement (proposal — NOT fit to data yet)

Keep the α(K) structure that survived (eq. 8's numerator/denominator competition), remove the redundant σ(K) term, add an explicit temperature term:

```
log t_grok ≈ L·μ_0(L)·exp(-γ·K/L) / f(T),   T = lr/bs,   f increasing      (10)
```

with α(K) = ∂ log t_grok / ∂ log wd unchanged in form (the K-dependence still enters through μ), and the bs/lr dependence now carried by f(T) with the empirically-correct sign.

Commitments:
- (A, β) from v0.1 remain frozen; the §1.4 recomputation is a transparency fix, not a refit.
- f(T) is *not* fit to the 3-point H4 data. It generates the predictions below instead.

## 5. New pre-registrable predictions (to run ONLY under a new dated pre-registration)

| # | Prediction | Test | Cost |
|---|---|---|---|
| P1 | **lr mirror of H4:** at fixed bs=512, wd=1.0, p=97, t_grok decreases monotonically with lr over a stable range (T = lr/bs ↑ → faster) | lr ∈ {5e-4, 1e-3, 2e-3}, 3 seeds = 9 runs | ~3h GPU |
| P2 | **wd_opt(K) shift** (inherited from v0.1 §6, still untested): optimal wd decreases with K | fine wd sweep ≥3.0 at p=31 vs p=149 | ~6h GPU |
| P3 | **Asymptote (Property B):** α at p=197 shallower than -0.50 | 5 wd × 3 seeds at p=197 | ~8h GPU |
| P4 | **T-invariance of α:** the slope α(p=97) measured at bs=256 equals α at bs=1024 within ±0.15 (temperature shifts t_grok's *level*, not the wd exponent) — this is the cleanest discriminator between eq. (10) and "wd and bs interact" | 2 × (5 wd × 3 seeds) = 30 runs | ~15h GPU |

P4 is the strongest: if α depends on bs, eq. (10)'s separability is wrong and the framework needs an interaction term.

### 5.1 RESULTS — Phase 9 (P1 + P4), completed 2026-07-08

Pre-registration `prereg/PRE_REGISTRATION_P1_P4.md`, public commit `320f7799` (2026-07-07),
pushed before any run. 33/33 runs clean, zero DNF. Full report: `results/phase9/REPORT.md`.

**P1 — CONFIRMED (binding criterion).** t_grok(lr) at bs=512, wd=1.0:
2567 (5e-4) → 1333 (1e-3) → 933 (2e-3), monotone decreasing as predicted (higher lr =
higher temperature = faster grokking). Exploratory bands: t(2e-3)/t(1e-3) = 0.70 (inside
[0.65, 0.90]); t(5e-4)/t(1e-3) = **1.93, outside [1.10, 1.55]** — the low-lr slowdown is
*stronger* than the temperature model predicted. Direction confirmed; magnitude at low lr
under-predicted.

**P4 — REJECTED (binding criterion). This is the important result.**
α is **not** batch-size-invariant. Measured slopes:

| bs | α (log-log, 5 wd cells, 3 seeds each) |
|---|---|
| 256 | -0.636 |
| 512 | -0.731 (reference, Phases 4-7) |
| 1024 | -0.802 |

|α(256) − α(1024)| = **0.166 > 0.15 tolerance → separability REFUTED**. Moreover the
dependence is not noise: **|α| increases monotonically with batch size** across all three
points (0.636 → 0.731 → 0.802). Larger batches make grokking time *more* sensitive to
weight decay. eq. (10) — where T multiplies the whole numerator and therefore cannot touch
the wd exponent — is falsified at first test. **A wd × bs (equivalently wd × T) interaction
term is required.**

**Reconciling P1 and P4.** P1 shows the temperature *direction* is right for the overall
level (lr↑ → faster). P4 shows T additionally reshapes the wd response (bs↑ → steeper α).
So T is not a separable prefactor; it enters inside the K/L competition. The minimal
consistent revision:

```
log t_grok ≈ L·μ_0(L)·exp(-γ·K/L) / f(T)  with  γ = γ(T) increasing in bs        (11)
```

i.e. batch size modulates the effective task-constraint coupling γ. Note P1 varied lr (not
bs) and tested only the level, so lr's effect on α is still untested — a clean asymmetry to
close (see revised P-list §5.2). **(11) is post-hoc to P4 and carries zero confirmatory
weight; it is the next thing to pre-register, not a result.**

### 5.2 Remaining pre-registrable predictions (unchanged; still open)

- **P2** — wd_opt(K) shift (v0.1 §6): optimal wd decreases with K. ~6h GPU.
- **P3** — asymptote: α at p=197 shallower than -0.50. ~8h GPU.
- **P1′ (new, from §5.1)** — does α depend on lr the way it depends on bs? Measure α(p=97)
  at lr=5e-4 vs lr=2e-3 (fixed bs=512). If |Δα| tracks the bs result under matched T-ratio,
  the interaction is a genuine function of T; if not, bs and lr act through different
  channels and (11) is too simple. This is the sharpest next test of (11).

## 6. Consolidated status of the framework after Fases I–II

| Component | Status |
|---|---|
| α(K) monotone-trend (Property A) | Supported (6 points; strict monotonicity unresolved) — **the solid empirical core** |
| Eq. (8) quantitative prediction (§4.3) | **VACATED — pre-registration compromised (§1)**; eq. (8) demoted to descriptive 2-parameter fit |
| σ as gradient SNR (v0.1 §2.3) | **Refuted in its bs-reading by H4**; replaced by temperature term in v0.2 proposal |
| Temperature separability, eq. (10) | **REFUTED by P4** (genuinely pre-registered, commit 320f7799): α depends on bs (0.636→0.731→0.802). Interaction term needed, eq. (11) |
| Temperature direction (level), P1 | **CONFIRMED** (lr↑→faster); low-lr magnitude under-predicted |
| L(wd) mapping | Scoped to AdamW-class optimizers (H2: SGD collapses) |
| Init robustness | Supported (H3, genuinely pre-registered) |
| Depth/composition | Slope survives, but regime changes (H1, genuinely pre-registered) — open question §3.2 |
| Optimal-wd prediction (v0.1 §6) | Still untested (P2) |

**Sober summary for the paper:** the publishable core is **empirical**: a 6-point α(K) map with genuine pre-registered robustness checks (init, batch size, optimizer-scope, regime boundary at depth-2), two mechanistic negative results (SGD weight collapse; σ-as-SNR sign error), and — new empirical result from P4 — a **monotone wd × batch-size interaction** (|α| grows with bs). The theory now has an honest confirmatory record: of two genuinely pre-registered predictions (public commit, code-generated, timestamped before data), **P1 passed and P4 failed** — the separable temperature model is refuted, an interaction model (11) proposed and *not yet* tested. Contrast with the earlier §4.3 incident (a compromised pre-registration, disclosed §1). Under Lakatos: the empirical hard core is solid; the theoretical protective belt has a documented degenerate episode and now one clean falsification driving the next revision. **A framework that publicly predicted, pre-registered, and reported its own 50% hit rate is more credible than one that reports only wins** — that asymmetry is the paper's methodological contribution.

## 7. Limitations (updated from v0.1 §7)

1. v0.1 limitations 1-4 stand (3-point fit origin, linearization validity, empirical L(wd), replica-symmetric μ).
2. **The §4.3 pre-registration is compromised (§1); no confirmatory claim survives from it.** The fit script (`fit_alpha_K.py`) is now committed; all future predictions are generated by committed code and registered externally (OSF) before data collection.
3. The temperature refinement (§4) is post-hoc relative to H4 — it is honest *hypothesis generation*, and P1/P4 are its pre-registrable tests. It must not be presented as a confirmed mechanism.
4. All conclusions live on modular addition, d_model=128, 2-layer encoder, p ≤ 149. No claim beyond this family until P3+ and (eventually) a second task family.
5. Every timestamp claim in this document is auditable from file mtimes and the repo's sync commits; the paper's data-availability section should state this explicitly.

## 8. Next operational steps

| step | what | GPU? |
|---|---|---|
| 1 | Recompute (A, β, C) transparently, commit fit script (`code/grokking/fit_alpha_K.py`) | no |
| 2 | Generate paper figures: α(K) predicted-vs-observed; master-table heatmap; H4 bar | no |
| 3 | Draft paper v1 skeleton (IMRaD) with §6 table as results backbone | no |
| 4 | If Pablo approves: pre-register P1/P4 (cheapest/strongest) and run | yes (~18h) |
| 5 | Fase IV: JAX replication of 2 anchor cells + arXiv preprint | yes (~4h) |

---

## Changelog

- **2026-07-08 — Phase 9 results integrated** (§5.1). P1 CONFIRMED (temperature direction), P4 REJECTED (α depends on bs: 0.636/0.731/0.802 → separability refuted, interaction eq. (11) proposed). 33/33 runs clean. Status table (§6) and summary updated. Public pre-registration commit 320f7799 predates all runs.
- **2026-07-07 — v0.2 created** (Fable 5 session). Initial evaluation found §4.3 nominally passing (3/3 within 0.10). Transparent reconstruction via `fit_alpha_K.py` then **vacated the confirmatory claim**: stated parameters reproduce nothing; Phase-6-anchor re-derivation fails the falsification line at p=67; published numbers match a post-Phase-7-anchor fit to ≤0.006; timeline permits contamination (Phase 7 REPORT 2026-04-28 16:50 < v0.1 mtime 2026-04-29 07:01). §1 rewritten same day to the vacated verdict — both states of this evaluation are preserved in this changelog for the audit trail. Integrated Phase 8 verdicts (§3), temperature refinement (§4), P1–P4 (§5). Figure: `figures/alpha_vs_K_pred_vs_obs.png`.
