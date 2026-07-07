# Phase 8 DEBUG — REPORT

Generated: 2026-07-06T10:21:33

## Verdicts per hypothesis

| Run | Hypothesis | Result | t_grok | acc | time | Verdict |
|---|---|---|---|---|---|---|
| debugB3_sgd_wd1.0_baseline | SGD-momentum funciona en absoluto con wd=1.0 baseline | NOT GROKKED | — | 0.0079 | 0.2h | **HYPOTHESIS REJECTED** (no grokking, near chance) |
| debugA2_depth2_wd0.3 | depth=2 con wd menor (0.3) sí grokkea | NOT GROKKED | — | 0.4655 | 0.5h | **HYPOTHESIS REJECTED** (no grokking, near chance) |
| debugA3_depth2_wd1.0_4layers | depth=2 wd=1.0 con n_layers=4 (más capacidad) sí grokkea | NOT GROKKED | — | 0.2532 | 7.0h | **HYPOTHESIS REJECTED** (no grokking, near chance) |
| debugA1_depth2_wd1.0_long | depth=2 wd=1.0 simplemente necesita 300K steps (no 200K) | NOT GROKKED | — | 0.1410 | 1.4h | **HYPOTHESIS REJECTED** (no grokking, near chance) |
| debugB1_sgd_wd0.01_lr3e-3 | SGD wd bajo necesita lr menor (3e-3 vs 1e-2) | NOT RUN | — | — | — | — |
| debugB2_sgd_wd0.01_warmup2000 | SGD wd bajo necesita warmup más largo (2000 vs 100) | NOT RUN | — | — | — | — |

## Diagnostic conclusions

### Problem A: H1 depth=2 wd alto no aprende

- A1: depth=2 wd=1.0 NO grokkea ni con 300K steps (acc=0.141) → problema NO es step budget.
- A2: depth=2 wd=0.3 tampoco grokkea (acc=0.466) → problema es estructural.
- A3: depth=2 wd=1.0 con n_layers=4 tampoco grokkea → capacidad no es el cuello de botella.

### Problem B: H2 SGD wd bajo no aprende

- B3 (sanity): SGD wd=1.0 baseline NO grokkea (acc=0.008) → **SGD PATH ESTÁ ROTO en train_v2.py / train_v3.py**. Antes de continuar, debug del código SGD.

### Problem C: SGD ~6× más lento que AdamW

- SGD avg: 5.85s/1K steps
- AdamW avg: 94.90s/1K steps
- Ratio SGD/AdamW: 0.06×

## Recomendación para Phase 8 v2

Basado en los diagnósticos arriba, la siguiente fase Phase 8 corregida debería:

- H2 (SGD): el path está roto. Debugger manual de train_v3.py SGD necesario antes de re-lanzar.
