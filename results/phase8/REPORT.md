# Phase 8 (Fase II) — REPORT

Generated: 2026-07-07T00:57:13

## Pre-registered hypotheses (from PRE_REGISTRATION_FASE_II.md)

Each section evaluates the observed result against the pre-fixed criteria. No post-hoc adjustment of intervals or thresholds.

## H1 — Depth (task composition) invariance

### Per-wd table (depth=2)

| wd | mean t_grok_sustained | std | N_clean / N_total |
|---|---|---|---|
| 0.01 | 17200 | 3874 | 3/3 |
| 0.1 | 3300 | 283 | 2/3 |
| 0.3 | 1900 | 0 | 1/3 |
| 1.0 | — | — | 0/3 |
| 3.0 | — | — | 0/3 |

**Observed slope: -0.658**
**Pre-registered interval: [-0.94, -0.54]**
**Verdict: H1 CONFIRMED**

- depth=1 reference (Phase 4+5): slope = -0.74
- depth=2 observed: slope = -0.658
- Patrón slope(wd) es robusto a composición de tarea (depth=1 → depth=2).

## H2 — Optimizer dependence (SGD vs AdamW)

### Per-wd table (SGD-momentum)

| wd | mean t_grok_sustained | std | N_clean / N_total | diverged_count |
|---|---|---|---|---|
| 0.01 | — | — | 0/3 | 0 |
| 0.1 | — | — | 0/3 | 0 |
| 0.3 | — | — | 0/3 | 0 |
| 1.0 | — | — | 0/3 | 0 |
| 3.0 | — | — | 0/3 | 0 |

**No data sufficient.**

## H3 — Init invariance (Xavier vs Kaiming)

| init | mean t_grok_sustained | std | N_clean / N_total |
|---|---|---|---|
| xavier | 1280 | 84 | 5/5 |
| kaiming | 1640 | 288 | 5/5 |

**Ratio Kaiming / Xavier: 1.28**
**Pre-registered: ratio ∈ [0.5, 1.5] (within ±50%)**
**Verdict: H3 CONFIRMED**

- Init no es confound mayor para los efectos reportados.

## H4 — batch_size dependence

| bs | mean t_grok_sustained | std | N_clean / N_total |
|---|---|---|---|
| 256 | 1067 | 153 | 3/3 |
| 512 | 1314 (ref) | 195 | 7/8 (Phase 4+5) |
| 1024 | 1467 | 58 | 3/3 |

**Order: bs=256: 1067 → bs=512: 1314 → bs=1024: 1467**
**Pre-registered: t_grok monotonically increasing with bs**
**Verdict: H4 CONFIRMED (monotonic)**

## Meta-analysis: which decision case?

H1: OK
H2: REJECTED
H3: OK
H4: OK

**3/4 hypotheses confirmed.**

→ Caso múltiple del pre-registro. Reformular claims según PRE_REGISTRATION_FASE_II.md sección 'Decisiones de meta-análisis'.
