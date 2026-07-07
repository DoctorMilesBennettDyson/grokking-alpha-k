# Fase I — REPORT (Phase 7 + integración con Fases 4-6)

Generated: 2026-04-28T16:50:34

## Componentes de Fase I

- A) Cierre de celdas frágiles (p=31 wd=0.01/0.1, p=97 wd=3.0, p=149 wd=3.0)
- B) Llenado de primos intermedios (p=53, 67, 113 con sweep completo)

Combinado con datos de Fases 4-6, este reporte produce la curva slope(p) con 6 puntos: p ∈ {31, 53, 67, 97, 113, 149}.

## Tabla maestra: t_grok_sustained por (p, wd) — todas las fases combinadas

Cada celda: mean (std) [N_clean / N_total]

| p \ wd | 0.01 | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|---|
| **p=31** | 198367 (83967) [3/8] | 39660 (12751) [5/6] | 10267 (1882) [3/3] | 2767 (153) [3/3] | 1200 (100) [3/3] |
| **p=53** | 153700 (9899) [2/3] | 32767 (2136) [3/3] | 9433 (1012) [3/3] | 2567 (58) [3/3] | 1350 (71) [2/3] |
| **p=67** | 91300 (13796) [3/3] | 23233 (987) [3/3] | 6733 (306) [3/3] | 1933 (115) [3/3] | 767 (58) [3/3] |
| **p=97** | 30133 (7107) [3/3] | 9267 (1582) [3/3] | 4100 (100) [3/3] | 1314 (195) [7/8] | 471 (76) [7/8] |
| **p=113** | 31767 (8739) [3/3] | 6800 (3460) [3/3] | 2933 (513) [3/3] | 1067 (351) [3/3] | 500 (0) [3/3] |
| **p=149** | 13233 (3371) [3/3] | 5767 (929) [3/3] | 2333 (404) [3/3] | 833 (58) [3/3] | 617 (75) [6/8] |

## Curva slope(p) — Fase I result

| p | n_train | slope (log-log) |
|---|---|---|
| 31 | 480 | -0.926 |
| 53 | 1404 | -0.866 |
| 67 | 2244 | -0.861 |
| 97 | 4704 | -0.731 |
| 113 | 6384 | -0.736 |
| 149 | 11100 | -0.577 |

### Monotonicidad de |slope| con p

|slope| **NO es monotónica** con p. → Caso interesante. La función slope(p) tiene estructura no trivial. Esto puede ser señal de un mecanismo distinto al simple K(X) × wd, o ruido en el estimador con N=3 en algunas celdas.

## Run-level data (todas las celdas, todas las fases)

Útil para revisar outliers y DNFs.

### p=31, wd=0.01

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 144300 | 144300 | 1.0000 | yes |
| 1 | phase6 | DNF | DNF | 0.5343 | NO |
| 2 | phase6 | 151800 | 153000 | 0.8399 | NO |
| 10 | phase7 | DNF | DNF | 0.4823 | NO |
| 11 | phase7 | 295100 | 295100 | 0.9917 | yes |
| 12 | phase7 | DNF | DNF | 0.6133 | NO |
| 13 | phase7 | DNF | DNF | 0.5010 | NO |
| 14 | phase7 | 155700 | 155700 | 0.9501 | yes |

### p=31, wd=0.1

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 25900 | 27200 | 0.9979 | yes |
| 1 | phase6 | 49300 | 50400 | 0.9314 | NO |
| 2 | phase6 | 32700 | 32700 | 0.9730 | yes |
| 10 | phase7 | 32900 | 33100 | 1.0000 | yes |
| 11 | phase7 | 55700 | 58400 | 0.9979 | yes |
| 12 | phase7 | 45800 | 46900 | 0.9667 | yes |

### p=31, wd=0.3

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 7900 | 8100 | 1.0000 | yes |
| 1 | phase6 | 11500 | 11500 | 0.9938 | yes |
| 2 | phase6 | 11200 | 11200 | 0.9896 | yes |

### p=31, wd=1.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 2600 | 2600 | 0.9875 | yes |
| 1 | phase6 | 2900 | 2900 | 0.9979 | yes |
| 2 | phase6 | 2800 | 2800 | 1.0000 | yes |

### p=31, wd=3.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 1000 | 1300 | 0.9979 | yes |
| 1 | phase6 | 1100 | 1100 | 1.0000 | yes |
| 2 | phase6 | 900 | 1200 | 1.0000 | yes |

### p=53, wd=0.01

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 146700 | 146700 | 0.9979 | yes |
| 1 | phase7 | DNF | DNF | 0.8954 | NO |
| 2 | phase7 | 154700 | 160700 | 0.9730 | yes |

### p=53, wd=0.1

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 31900 | 31900 | 1.0000 | yes |
| 1 | phase7 | 31200 | 31200 | 0.9680 | yes |
| 2 | phase7 | 35200 | 35200 | 1.0000 | yes |

### p=53, wd=0.3

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 8900 | 8900 | 1.0000 | yes |
| 1 | phase7 | 8800 | 8800 | 1.0000 | yes |
| 2 | phase7 | 10600 | 10600 | 1.0000 | yes |

### p=53, wd=1.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 2600 | 2600 | 1.0000 | yes |
| 1 | phase7 | 2500 | 2500 | 0.9993 | yes |
| 2 | phase7 | 2600 | 2600 | 1.0000 | yes |

### p=53, wd=3.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 1300 | 1300 | 0.9957 | yes |
| 1 | phase7 | 1100 | 1100 | 0.8342 | NO |
| 2 | phase7 | 1100 | 1400 | 1.0000 | yes |

### p=67, wd=0.01

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 89100 | 90000 | 1.0000 | yes |
| 1 | phase7 | 78200 | 78200 | 0.9902 | yes |
| 2 | phase7 | 105700 | 105700 | 0.9813 | yes |

### p=67, wd=0.1

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 23700 | 23700 | 1.0000 | yes |
| 1 | phase7 | 22100 | 22100 | 1.0000 | yes |
| 2 | phase7 | 23900 | 23900 | 1.0000 | yes |

### p=67, wd=0.3

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 7000 | 7000 | 1.0000 | yes |
| 1 | phase7 | 6400 | 6400 | 1.0000 | yes |
| 2 | phase7 | 6800 | 6800 | 1.0000 | yes |

### p=67, wd=1.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 2000 | 2000 | 1.0000 | yes |
| 1 | phase7 | 1800 | 1800 | 1.0000 | yes |
| 2 | phase7 | 2000 | 2000 | 1.0000 | yes |

### p=67, wd=3.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 800 | 800 | 0.9996 | yes |
| 1 | phase7 | 700 | 700 | 1.0000 | yes |
| 2 | phase7 | 800 | 800 | 1.0000 | yes |

### p=97, wd=0.01

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase5 | 32300 | 32700 | 0.9996 | yes |
| 1 | phase5 | 20800 | 22100 | 0.9979 | yes |
| 2 | phase5 | 35600 | 35600 | 1.0000 | yes |

### p=97, wd=0.1

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase5 | 8900 | 8900 | 1.0000 | yes |
| 1 | phase5 | 7900 | 7900 | 1.0000 | yes |
| 2 | phase5 | 11000 | 11000 | 1.0000 | yes |

### p=97, wd=0.3

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase5 | 4000 | 4000 | 1.0000 | yes |
| 1 | phase5 | 4200 | 4200 | 1.0000 | yes |
| 2 | phase5 | 4100 | 4100 | 0.9998 | yes |

### p=97, wd=1.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase4 | 1200 | 1200 | 1.0000 | yes |
| 1 | phase4 | 1200 | 1400 | 0.7052 | NO |
| 2 | phase4 | 1400 | 1400 | 1.0000 | yes |
| 10 | phase5 | 1600 | 1600 | 1.0000 | yes |
| 11 | phase5 | 1400 | 1400 | 1.0000 | yes |
| 12 | phase5 | 1400 | 1400 | 1.0000 | yes |
| 13 | phase5 | 1200 | 1200 | 1.0000 | yes |
| 14 | phase5 | 1000 | 1000 | 1.0000 | yes |

### p=97, wd=3.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase5 | 400 | 400 | 1.0000 | yes |
| 1 | phase5 | 500 | 500 | 0.4155 | NO |
| 2 | phase5 | 500 | 500 | 1.0000 | yes |
| 10 | phase7 | 600 | 600 | 0.9998 | yes |
| 11 | phase7 | 500 | 500 | 1.0000 | yes |
| 12 | phase7 | 400 | 400 | 1.0000 | yes |
| 13 | phase7 | 500 | 500 | 1.0000 | yes |
| 14 | phase7 | 400 | 400 | 1.0000 | yes |

### p=113, wd=0.01

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 36700 | 37400 | 0.9989 | yes |
| 1 | phase7 | 21700 | 21700 | 0.9994 | yes |
| 2 | phase7 | 36200 | 36200 | 0.9973 | yes |

### p=113, wd=0.1

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 8000 | 8000 | 1.0000 | yes |
| 1 | phase7 | 2900 | 2900 | 0.9998 | yes |
| 2 | phase7 | 9500 | 9500 | 1.0000 | yes |

### p=113, wd=0.3

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 2800 | 2800 | 1.0000 | yes |
| 1 | phase7 | 2500 | 2500 | 1.0000 | yes |
| 2 | phase7 | 3500 | 3500 | 1.0000 | yes |

### p=113, wd=1.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 1100 | 1100 | 0.9998 | yes |
| 1 | phase7 | 700 | 700 | 1.0000 | yes |
| 2 | phase7 | 1400 | 1400 | 1.0000 | yes |

### p=113, wd=3.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase7 | 500 | 500 | 0.9840 | yes |
| 1 | phase7 | 500 | 500 | 0.9984 | yes |
| 2 | phase7 | 500 | 500 | 0.9814 | yes |

### p=149, wd=0.01

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 12200 | 12200 | 0.9995 | yes |
| 1 | phase6 | 17000 | 17000 | 0.9980 | yes |
| 2 | phase6 | 10100 | 10500 | 0.9983 | yes |

### p=149, wd=0.1

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 5500 | 5500 | 0.9984 | yes |
| 1 | phase6 | 6800 | 6800 | 1.0000 | yes |
| 2 | phase6 | 5000 | 5000 | 1.0000 | yes |

### p=149, wd=0.3

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 2400 | 2400 | 1.0000 | yes |
| 1 | phase6 | 2700 | 2700 | 1.0000 | yes |
| 2 | phase6 | 1900 | 1900 | 1.0000 | yes |

### p=149, wd=1.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 800 | 800 | 1.0000 | yes |
| 1 | phase6 | 900 | 900 | 1.0000 | yes |
| 2 | phase6 | 800 | 800 | 1.0000 | yes |

### p=149, wd=3.0

| seed | source | t_grok_first | t_grok_sustained | acc | clean? |
|---|---|---|---|---|---|
| 0 | phase6 | 700 | 700 | 0.9999 | yes |
| 1 | phase6 | 600 | 600 | 0.9845 | yes |
| 2 | phase6 | 800 | 5400 | 0.8739 | NO |
| 10 | phase7 | 500 | 500 | 0.9942 | yes |
| 11 | phase7 | 700 | 700 | 1.0000 | yes |
| 12 | phase7 | 600 | 15800 | 0.8480 | NO |
| 13 | phase7 | 600 | 600 | 0.9865 | yes |
| 14 | phase7 | 600 | 600 | 1.0000 | yes |

## Próximo paso operativo

1. Revisar la tabla maestra para confirmar que cada celda tiene N_clean ≥ 3.
2. Si alguna celda crítica tiene N_clean < 3, considerar Phase 7.5 ad-hoc.
3. Si todas las celdas son robustas, verificar que `PRE_REGISTRATION_FASE_II.md` está commiteado y firmado.
4. Lanzar Fase II (script separado, dependiente de pre-registro).

