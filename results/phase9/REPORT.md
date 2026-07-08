# Phase 9 (P1 + P4) — REPORT

Generated: 2026-07-08T19:51:57

Criteria are exactly those of prereg/PRE_REGISTRATION_P1_P4.md. No post-hoc changes.

## P1 — t_grok vs learning rate (bs=512, wd=1.0)

| lr | mean t_sustained | std | N_clean/N |
|---|---|---|---|
| 0.0005 | 2567 | 153 | 3/3 |
| 0.001 | 1333 | 115 | 3/3 |
| 0.002 | 933 | 321 | 3/3 |

**Ordering observed:** 2567 > 1333 > 933 → holds
**Verdict: P1 CONFIRMED** (binding criterion: monotone decrease)
- Exploratory bands: t(2e-3)/t(1e-3) = 0.70 (band [0.65, 0.90]); t(5e-4)/t(1e-3) = 1.93 (band [1.10, 1.55]) — non-binding.

## P4 — wd slope at bs=256 vs bs=1024 (lr=1e-3)

### bs=256

| wd | mean t_sustained | N_clean/N | source |
|---|---|---|---|
| 0.01 | 19333 | 3/3 | phase9 |
| 0.1 | 7367 | 3/3 | phase9 |
| 0.3 | 2467 | 3/3 | phase9 |
| 1.0 | 1067 | 3/3 | phase8 H4 (declared reuse) |
| 3.0 | 600 | 3/3 | phase9 |

slope α(bs=256) = -0.636

### bs=1024

| wd | mean t_sustained | N_clean/N | source |
|---|---|---|---|
| 0.01 | 44633 | 3/3 | phase9 |
| 0.1 | 15033 | 3/3 | phase9 |
| 0.3 | 4667 | 3/3 | phase9 |
| 1.0 | 1467 | 3/3 | phase8 H4 (declared reuse) |
| 3.0 | 500 | 3/3 | phase9 |

slope α(bs=1024) = -0.802

**|α(256) − α(1024)| = 0.166** (binding tolerance 0.15; reference α(512) = -0.731)
**Verdict: P4 REJECTED — wd × bs interaction**

## Pre-registered decision table

P1 pass + P4 fail → T affects level via lr, but separability fails → interaction model needed
