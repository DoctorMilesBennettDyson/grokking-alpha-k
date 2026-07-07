# The grokking exponent shrinks with task size: an empirical map of α(K) with pre-registered robustness checks

**Pablo L. Rainieri Blasco**
Independent researcher — Tucumán, Argentina
Draft v0.1 — 2026-07-07 — code and raw data: `github.com/DoctorMilesBennettDyson/grokking-alpha-k`

> **Draft status:** results of the pre-registered P1/P4 experiments (Section 5.3) are
> pending; placeholders are marked `[P9]`. Everything else is final data.

---

## Abstract

Grokking — delayed generalization long after training-set memorization — is known to
depend strongly on weight decay (wd), with grokking time following an approximate power
law `t_grok ∝ wd^α`. We map how the exponent α depends on task size on modular addition,
training 2-layer transformers across six moduli p ∈ {31, 53, 67, 97, 113, 149} (training
sets K from 480 to 11,100 examples) and five weight decays over ~200 runs on a single
consumer GPU. We find that **|α| shrinks monotonically-in-trend with K**, from -0.93 at
K=480 to -0.58 at K=11,100: the larger the task, the less weight decay accelerates
grokking. We stress-test this map with robustness checks pre-registered before execution:
the effect is not an initialization artifact (Kaiming/Xavier t_grok ratio 1.28); grokking
time increases monotonically with batch size (1067 → 1314 → 1467 for bs 256 → 512 → 1024),
indicating SGD noise *accelerates* grokking; and the power law survives a task-composition
change (depth-2 sums, slope -0.66) even though that regime exhibits no memorization phase
at all — generalization there is ordinary, not grokked. Two negative results bound the
scope: plain SGD-momentum never groks at any tested setting — we identify the mechanism
as weight-norm collapse under coupled L2 + momentum (norm 42 → 0.02 within 1K steps) —
and an earlier theoretical "pre-registration" from our own program is disclosed as
compromised after a forensic reconstruction of its fit; the replacement predictions were
registered publicly before their data collection began. All code, raw runs (including
failures), and pre-registration artifacts are public.

## 1. Introduction

Grokking was reported by Power et al. (2022): small transformers trained on modular
arithmetic memorize the training set quickly, then generalize abruptly many thousands of
steps later, with weight decay strongly modulating the delay. Follow-up work explained
the transition mechanistically (Nanda et al., 2023: Fourier-feature circuits displacing
memorization) and as a phase transition (Liu et al., 2022). What remains unmapped is the
**interaction between regularization and task size**: does weight decay accelerate
grokking equally on small and large tasks?

We answer empirically. Our contributions:

1. **An α(K) map** (Section 3): the exponent of `t_grok ∝ wd^α` measured at six task
   sizes, N=3–8 seeds per cell, with all outliers and DNFs reported. |α| decreases from
   0.93 to 0.58 as K grows 23×.
2. **Pre-registered robustness checks** (Section 4): initialization, batch size,
   optimizer, and task composition, with quantitative acceptance criteria fixed and
   dated before execution.
3. **Two mechanistic negative results** (Sections 4.2, 4.4): SGD-momentum weight-norm
   collapse, and the failure of a gradient-SNR reading of batch-size effects.
4. **A methods contribution** (Section 5.2): we disclose, with the forensic
   reconstruction that exposed it, how a theory "pre-registration" inside our own program
   was contaminated by data it claimed to predate — and how commit-hash-timestamped,
   code-generated predictions prevent the failure mode.

Everything runs on one RTX 3070 (8 GB); total compute for all results is under 150 GPU-hours.

## 2. Setup

**Task.** Modular addition `(a + b) mod p`, all p² pairs enumerated, random 50/50
train/test split. Vocabulary: p digit tokens + operator + equals. Depth-2 variant
(Section 4.3): `((a + b) + c) mod p`, subsampled to 500K examples.

**Model.** 2-layer transformer encoder, d_model=128, 4 heads, d_ff=512, GELU, pre-norm;
~423K parameters. Xavier init unless stated.

**Training.** AdamW (β = 0.9/0.98), lr = 1e-3, 100-step warmup, batch 512, 200K steps,
full-batch evaluation every 100 steps. Weight decay wd ∈ {0.01, 0.1, 0.3, 1.0, 3.0}.

**Definitions (fixed in pre-registration).** `t_grok` = first step with test accuracy
≥ 0.95 sustained for 3 consecutive evaluations. A run is an *outlier* if final test
accuracy < 0.95 or the sustained threshold trails the first crossing by > 1000 steps;
outliers are tabulated but excluded from cell means. DNF = threshold never reached in
budget; cells with DNFs are lower bounds and excluded from slope fits.

## 3. The α(K) map

Master table (mean sustained t_grok; full per-run tables incl. outliers/DNFs in
`results/phase7/REPORT.md`):

| p \ wd | 0.01 | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|---|
| 31 | 198,367 [3/8] | 39,660 [5/6] | 10,267 | 2,767 | 1,200 |
| 53 | 153,700 [2/3] | 32,767 | 9,433 | 2,567 | 1,350 [2/3] |
| 67 | 91,300 | 23,233 | 6,733 | 1,933 | 767 |
| 97 | 30,133 | 9,267 | 4,100 | 1,314 [7/8] | 471 [7/8] |
| 113 | 31,767 | 6,800 | 2,933 | 1,067 | 500 |
| 149 | 13,233 | 5,767 | 2,333 | 833 | 617 [6/8] |

Log–log slope of t_grok vs wd per modulus:

| p | K | α |
|---|---|---|
| 31 | 480 | **-0.926** |
| 53 | 1404 | -0.866 |
| 67 | 2244 | -0.861 |
| 97 | 4704 | -0.731 |
| 113 | 6384 | -0.736 |
| 149 | 11100 | **-0.577** |

Two honest observations. First, the *trend* is clearly decreasing (|α| falls 0.35 over a
23× range in K), but strict monotonicity is not resolvable at N=3/cell: there are two
local plateaus (53→67 and 97→113, both |Δ| = 0.005, far inside noise). Second, the
noisiest cell in the grid (p=31, wd=0.01: 3/8 clean, the rest DNF or unstable) anchors
the steep end; its slope moved from -0.86 to -0.93 when five seeds were added. Both facts
carry into every downstream fit.

Interpretation, informally: weight decay is a *circuit-complexity price*. On small tasks
(few constraints) many circuits fit the data and wd is decisive in forcing the simple,
generalizing one — so t_grok responds strongly. On large tasks the data itself already
excludes most circuits, and wd has less selection left to do.

## 4. Pre-registered robustness (Fase II)

All four hypotheses, acceptance intervals, and analysis rules were fixed and dated
2026-04-27, before execution began 2026-04-28 (`prereg/PRE_REGISTRATION_FASE_II.md`).
Execution completed 2026-07-07 after a crash-and-resume documented in that file's
changelog. Verdicts:

### 4.1 H3 — Initialization (CONFIRMED)
Criterion: Kaiming/Xavier sustained-t_grok ratio ∈ [0.5, 1.5] at the p=97, wd=1.0
baseline, 5 fresh seeds each. Observed: **1.28** (1640 ± 288 vs 1280 ± 84). The Xavier
arm doubles as an independent-seed replication of the baseline (1280 vs 1314 ± 195 from
earlier phases).

### 4.2 H2 — Optimizer (negative result; scope-bounding)
Criterion: SGD-momentum (lr 1e-2, momentum 0.9) preserves the sign of α. Observed:
**0/15 runs grok — all end at chance** (train loss pinned at the uniform-distribution
value). A differential diagnostic isolates the mechanism: with wd=0, the same SGD path
memorizes the training set in ~1K steps (code is healthy); with wd=1.0 the weight norm
collapses 42 → 0.02 within 1K steps and the network emits uniform logits permanently.
Decoupled SGDW at wd=1.0 collapses similarly. AdamW survives because adaptive
normalization keeps update magnitudes O(lr) as weights shrink, while SGD's gradients
shrink *with* the weights, so decay always wins. **The α(K) map is therefore a statement
about AdamW-class optimizers**; "weight decay" is not a single mechanism across
optimizers, echoing the decoupling analysis of Loshchilov & Hutter (2019).

### 4.3 H1 — Task composition (confirmed numerically; regime boundary found)
Criterion: at depth-2 (p=97), slope ∈ [-0.94, -0.54]. Observed: **-0.658** over the
three wd cells that generalize (wd ≥ 1.0: 0/3 reach criterion). But in every clean
depth-2 run, train and test accuracy rise *together* — `t_train_saturated == t_grok`
in all cells — because with 250K training examples and 423K parameters, memorization is
impossible. **There is no grokking at depth-2**; the wd power law survives in a regime
of ordinary generalization. We report this as a boundary of the grokking regime (the
memorization phase requires K small relative to capacity), and explicitly *not* as
"grokking is composition-invariant".

### 4.4 H4 — Batch size (CONFIRMED; kills a naive theory reading)
Criterion: sustained t_grok monotonically increasing with batch size at p=97, wd=1.0.
Observed: **1067 (bs=256) → 1314 (bs=512, reference) → 1467 (bs=1024)** — monotone as
pre-registered. The quantitative sub-prediction (each doubling +20–80%) held for the
first doubling (+23%) and failed for the second (+12%). The direction matters more than
the magnitudes: if minibatch noise limited *signal* (a gradient-SNR reading), larger
batches should grok *faster*. They grok slower. Minibatch noise **accelerates** grokking,
consistent with noise-as-exploration accounts of the memorization-to-generalization
transition.

## 5. Theory: status and method

### 5.1 A descriptive framework
Our working model (developed in `theory/`) treats grokking as circuit selection under a
complexity budget: t_grok ∝ exp(L·μ/D) with wd capping circuit complexity L, μ counting
data-compatible circuits, and D a noise/temperature term. A two-parameter reduction fits
the six-point α(K) curve with residuals 0.03–0.13. **We claim descriptive adequacy only**
(Section 5.2 explains why the framework currently has zero confirmatory standing) —
except that H4's sign (Section 4.4) already forced one revision: the noise term must act
as temperature (T ∝ lr/bs, higher is faster), not as signal-to-noise.

### 5.2 An integrity disclosure
An earlier version of this framework published "pre-registered" predictions for the
intermediate moduli, labeled as written before those data were observed, and they passed
their falsification test (all three within the stated 0.10 line). During preparation of
this paper we reconstructed the fit — no fit script had been saved — and found: (i) the
document's stated parameters reproduce neither its predictions nor its own anchors;
(ii) an honest re-derivation from the anchors it claims to have used *fails* the
falsification line at p=67; (iii) the published predictions instead match, to ≤0.006 on
all three points, a fit using slope values that only exist with the later data; and
(iv) file timestamps show those data existed ~14 hours before the document. We treat the
episode as HARKing, vacate the confirmatory claim, and freeze the original document with
the audit in its changelog (`prereg/theory_extension_v0.1.md`;
reconstruction: `code/fit_alpha_K.py`; figure `figures/alpha_vs_K_pred_vs_obs.png`
shows the published "predictions" sitting on the contaminated fit). We publish the
failure because the alternative — quietly dropping the claim — is how fields accumulate
unreproducible theory.

### 5.3 Replacement predictions, registered properly `[P9]`
Two consequences of the temperature reading were pre-registered in
`prereg/PRE_REGISTRATION_P1_P4.md`, timestamped by public commit `320f7799`
(2026-07-07), **before** their runs began:

- **P1** — at fixed bs, t_grok decreases monotonically with lr. `[P9: pending]`
- **P4** — α is batch-size-invariant: |α(bs=256) − α(bs=1024)| ≤ 0.15. This is the
  separability test: temperature may shift levels but must not touch the wd exponent.
  `[P9: pending]`

The pre-registered decision table (in the registration file) fixes the interpretation of
all four outcomes, including full refutation.

## 6. Related work

Power et al. (2022) established grokking on modular arithmetic and the role of weight
decay. Nanda et al. (2023) identified the Fourier-circuit mechanism at p=113 and
proposed progress measures; our map is consistent with circuit selection being the
rate-limiting process. Liu et al. (2022) frame grokking as a phase transition;
α(K) is naturally read as a K-dependent critical exponent, though we make no formal
claim. Loshchilov & Hutter (2019) motivate why coupled vs decoupled weight decay behave
differently, which Section 4.2 turns into a hard scope boundary. `[TODO: verify and add
citations on SGD-vs-adaptive optimization of transformers and on noise-accelerated
grokking before submission.]`

## 7. Limitations

1. One task family (modular addition), one architecture, p ≤ 149. No universality claim.
2. N=3 for most cells; strict monotonicity of α(K) unresolved; the steep-end anchor
   (p=31, wd=0.01) is fragile (3/8 clean).
3. The theoretical framework is post-hoc (Section 5.2); its first genuine tests are P1/P4.
4. Depth-2 conclusions concern a different (non-grokking) regime; see 4.3.
5. Compute limits: 200K-step budget makes some low-wd cells lower bounds.

## 8. Conclusion

Weight decay's leverage over grokking is not a constant of the phenomenon: it decays
with task size, |α| falling from 0.93 to 0.58 across a 23× range of K. The effect
survives initialization and batch-size checks, is bounded to AdamW-class optimizers by a
mechanistically-explained SGD failure, and persists across a regime boundary where
grokking itself disappears. The accompanying theory is offered as description plus two
publicly pre-registered predictions whose outcomes — pass or fail — will be reported in
v0.2 of this draft.

## Reproducibility

All code, per-run CSVs (including outliers, DNFs, and the truncated run from a mid-phase
crash), auto-generated reports, and dated pre-registration artifacts:
`github.com/DoctorMilesBennettDyson/grokking-alpha-k`. Hardware: single RTX 3070 8 GB;
PyTorch 2.4.1+cu118. Runners are idempotent; re-running resumes.

## References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V. (2022). *Grokking:
  Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177.
- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023). *Progress
  Measures for Grokking via Mechanistic Interpretability.* ICLR 2023. arXiv:2301.05217.
- Liu, Z., Kitouni, O., Nolte, N., Michaud, E., Tegmark, M., Williams, M. (2022).
  *Towards Understanding Grokking: An Effective Theory of Representation Learning.*
  NeurIPS 2022. arXiv:2205.10343.
- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019.
  arXiv:1711.05101.
