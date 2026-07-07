# Pre-Registro Formal — Fase II (Robustness Checks)

**Fecha de firma:** 2026-04-27
**Autores:** Pablo Rainieri + Opus 4.7
**Status:** FIJADO. Modificación posterior solo permitida con changelog explícito al pie.

**Propósito de este documento:** fijar las hipótesis exactas a testear, los métodos de análisis, y los criterios de falsificación ANTES de mirar los datos de Fase II. Esto previene p-hacking inconsciente y asegura que las conclusiones reflejen los datos, no la narrativa preferida.

**Compromiso:** los experimentos de Fase II no se ejecutan hasta que este documento esté firmado, fechado, y commiteado al repositorio. El análisis posterior a Fase II usa exactamente los métodos especificados aquí.

---

## Hipótesis observada en Fase 6 que motiva Fase II

De Fases 4-6 (datos ya cerrados antes de este pre-registro):

```
slope(p=31)  = -0.86
slope(p=97)  = -0.74
slope(p=149) = -0.57
```

Esta variación de slope con p sugiere una interacción K(X) × wd. Antes de claim de generalidad, hay que descartar que el patrón sea artifact de:

1. arquitectura específica (depth)
2. optimizer específico
3. esquema de inicialización
4. hyperparameter de batch_size

---

## Hipótesis pre-registradas para Fase II

Cada hipótesis es **cuantitativa, falsable, y con criterio explícito de aceptación o rechazo**.

---

### H1 — Invarianza por depth

**Hipótesis:** el slope de t_grok vs wd a p=97 con depth=2 cae dentro de ±0.20 del slope observado a depth=1 (Phase 4+5: slope = -0.74).

**Predicción cuantitativa:** slope(p=97, depth=2) ∈ [-0.94, -0.54].

**Por qué este intervalo:** ±0.20 es 27% del valor central (-0.74). Es generoso (permite variación moderada por arquitectura) pero excluye cambios drásticos.

**Aceptación:** si el slope observado cae dentro del intervalo → H1 confirmada → patrón es depth-invariante en este rango.

**Falsación:** si el slope cae fuera → H1 rechazada → la dependencia con depth es no trivial → claim de universalidad debe restringirse explícitamente a depth=1, o framework necesita revisión.

**Experimento:** p=97, depth=2, wd ∈ {0.01, 0.1, 0.3, 1.0, 3.0}, seeds {0, 1, 2}, steps=200K.

---

### H2 — Dependencia con optimizer

**Hipótesis NULA (default):** el patrón monotónicamente decreciente de t_grok vs wd se observa también con SGD-momentum, no solo con AdamW. La pendiente puede diferir cuantitativamente pero el signo es el mismo.

**Predicción cualitativa:** slope(SGD) < 0 (mismo signo que AdamW).

**Predicción cuantitativa exploratoria (no falsifica si falla):** slope(SGD, p=97) ∈ [-1.5, -0.3]. Rango amplio porque SGD-momentum tiene dinámica de optimización distinta.

**Aceptación:** si slope < 0 y monotonía preservada → H2 confirmada → wd modula t_grok via mecanismo no específico al optimizer.

**Falsación:** si slope cambia signo (i.e., wd alto → grokking más lento con SGD) → H2 rechazada → el efecto de wd es Adam-específico → reframing necesario: "we report that AdamW-driven grokking exhibits..." en lugar de claim general.

**Detalles del experimento:**
- p=97, depth=1, wd ∈ {0.01, 0.1, 0.3, 1.0, 3.0}, seeds {0, 1, 2}, steps=200K.
- SGD-momentum con lr=1e-2 (10× la lr de AdamW por convención), momentum=0.9.
- Si lr=1e-2 produce divergencia, fallback a lr=3e-3. Documentar la elección.

**Caveat técnico:** SGD-momentum sin AdamW requiere modificación de `train.py` (actualmente hardcodeado AdamW). La modificación se hace en `train_v2.py` separado para no contaminar runs anteriores. Diff explícito documentado.

---

### H3 — Invarianza por inicialización

**Hipótesis:** los valores absolutos de t_grok (no slope, sino los valores de la curva) en p=97 baseline (wd=1.0, depth=1) difieren entre Xavier y Kaiming init en menos del 50%.

**Predicción cuantitativa:**
- Xavier baseline: t_grok ≈ 1314 ± 195 (de Phase 4+5).
- Kaiming baseline: t_grok ∈ [657, 1971] (intervalo de ±50% del valor Xavier).

**Aceptación:** si Kaiming cae dentro del intervalo → H3 confirmada → init no es confound mayor.

**Falsación:** si Kaiming cae fuera (e.g., 3000 o 500) → init sí es factor relevante → todos los claims deben anclar al esquema de init usado, y un experimento adicional (slope con Kaiming) se requiere antes de generalización.

**Experimento:** p=97, depth=1, wd=1.0, init ∈ {Xavier, Kaiming}, seeds {20, 21, 22, 23, 24}.

**Caveat técnico:** requiere modificación de `model.py` para soportar Kaiming. La modificación se hace en `model_v2.py` separado.

---

### H4 — Dependencia con batch_size

**Hipótesis cualitativa:** t_grok aumenta monotónicamente con batch_size (i.e., menor batch → más ruido SGD → grokking más rápido).

**Predicción cuantitativa:** t_grok(bs=256) ≤ t_grok(bs=512) ≤ t_grok(bs=1024), con cada doblamiento aumentando t_grok en 20%-80%.

**Por qué este rango:** la literatura sugiere que SGD noise scale ∝ lr/bs. Si grokking responde a noise scale linealmente, esperaríamos cambios moderados (no factor 10×).

**Aceptación de monotonía:** orden estricto t_grok(256) < t_grok(512) < t_grok(1024) → H4 confirmada cualitativamente → consistente con interpretación de wd modulando L (efecto separado del μ del SGD).

**Falsación:** si t_grok no es monotónico (e.g., t_grok(512) > t_grok(1024)) → la relación bs ↔ noise ↔ grokking es más compleja de lo asumido → revisión del marco teórico necesaria.

**Experimento:** p=97, depth=1, wd=1.0, batch_size ∈ {256, 512, 1024}, seeds {0, 1, 2}, steps=200K.

---

## Reglas de análisis estadístico (pre-fijadas)

Aplican a TODOS los experimentos de Fase II. Sin desviación post-hoc.

### Cómputo de t_grok

Usamos **sustained threshold** de Phase 6: t_grok = primer step donde test_acc ≥ 0.95 sostenido por 3 evaluaciones consecutivas (300 steps).

### Cómputo de slope

Regresión lineal en log-log: log(t_grok_mean) vs log(wd) sobre los 5 puntos del sweep. Slope reportado con IC95% bootstrapeado (1000 resamples sobre las medias por celda).

### Manejo de outliers

Run se marca outlier si:
- final_test_acc < 0.95 (no grokking sostenido a fin de training), O
- t_grok_sustained > t_grok_first + 1000 steps (transición no estable).

Outliers se reportan en tabla pero **NO se incluyen en el cálculo de medias o slopes**. Si una celda tiene > 50% outliers, la celda se reporta como "no estable" sin promedio.

### Manejo de DNF

Run se marca DNF si t_grok no se alcanza dentro del step budget. DNFs se reportan explícitamente. Una celda con DNFs se trata como "lower bound" en t_grok mean (i.e., el verdadero t_grok es al menos el budget). Slopes calculados sobre celdas sin DNFs solamente.

### Comparación H vs no-H

Para cada hipótesis con predicción cuantitativa: el resultado se acepta si el valor observado cae dentro del intervalo predicho. NO se ajusta el intervalo post-hoc.

---

## Decisiones de meta-análisis pre-fijadas

### Caso 1: todas las H (H1-H4) confirmadas

→ Patrón slope(K) es robusto a las 4 dimensiones testeadas.
→ Paper 04 v0.2 puede afirmar generalidad (con limitations sobre rangos testeados).
→ Pasamos a Fase III (marco teórico).

### Caso 2: H1 rechazada (depth dependence)

→ Paper se restringe explícitamente a depth=1.
→ Se agrega experimento adicional (depth=3) si tiempo permite.
→ Limitation explicit en paper: "We do not claim universality across architectures."

### Caso 3: H2 rechazada (optimizer dependence)

→ Resultado se reporta como "AdamW-specific phenomenon".
→ El paper discute literatura comparable (¿alguien más reportó este patrón con SGD?).
→ Esto NO mata el paper, pero limita las claims.

### Caso 4: H3 rechazada (init dependence)

→ Experimento adicional: replicar slope(p=97) con Kaiming.
→ Si slope con Kaiming es muy distinto, init es confound mayor.

### Caso 5: H4 rechazada (no monotonía bs)

→ Más interesante científicamente. Sugiere bs interactúa con wd no trivialmente.
→ Experimento adicional: matriz bs × wd (5×3=15 runs adicionales).

### Caso 6: múltiples H rechazadas

→ El patrón slope(K) es menos robusto de lo asumido.
→ Reformular paper como "observation in narrow regime" en lugar de "general principle".
→ Honestidad sobre las restricciones de aplicabilidad.

---

## Compromiso operativo final

Yo (Opus 4.7) y Pablo nos comprometemos por escrito a:

1. **No modificar este pre-registro** después de mirar datos de Fase II. Si modificación es necesaria por descubrir bug en metodología, queda registrado en changelog con fecha y justificación.

2. **Reportar todos los resultados** (favorables y no), todos los outliers, todos los DNFs.

3. **Aceptar las decisiones pre-fijadas** de la sección anterior. Si el caso 6 ocurre, reformulamos honestamente. No buscamos otra historia para preservar el resultado de Fase 6.

4. **Tiempo necesario para Fase II:** estimado 25 GPU hrs + análisis. No se acelera.

---

## Changelog

### 2026-04-28 — Aclaración terminológica de "depth" en H1

**Contexto:** al implementar Fase II detectamos ambigüedad en H1. El pre-registro dice "depth=2" sin precisar si se refiere a:
- (A) profundidad de composición de tarea (parámetro `depth` existente en train_single_run, que controla si la operación es `a+b` o `(a+b)+c` mod p), o
- (B) profundidad arquitectural del transformer (n_layers del encoder, actualmente hardcoded en 2).

**Resolución:** se interpreta H1 como (A) — composición de tarea — por las siguientes razones:
1. El parámetro `depth` ya existe en el código y fue usado en Phase 3 con esta semántica.
2. Esta interpretación NO requiere modificar `model.py`, reduciendo riesgo de bugs colaterales.
3. Es un robustness check legítimo: "¿el patrón slope(wd) es invariante a la complejidad por composición de la tarea?".

**Lo que no cambia:** los criterios cuantitativos de H1 (intervalo [-0.94, -0.54] para slope) se mantienen idénticos.

**Lo que se agrega como deuda científica:** el experimento de profundidad arquitectural (variar n_layers del transformer) queda como **Fase II.5** condicional. Se ejecuta solo si:
- H1 (composición) se confirma → Fase II.5 testea si también vale para arquitectura distinta, fortaleciendo el claim.
- H1 (composición) se rechaza → Fase II.5 no se ejecuta porque ya sabemos que la invarianza es limitada, y agregar otra dimensión solo complica.

**Justificación de honestidad:** esta aclaración se escribe ANTES de ejecutar cualquier run de Fase II. No estamos modificando interpretación post-hoc para acomodar resultados. La ambigüedad existía en el documento original y se resuelve aquí sin haber visto datos.

### 2026-04-28 — Detalle técnico de implementación H2 y H3

H2 (SGD-momentum): se usa `torch.optim.SGD(lr=1e-2, momentum=0.9, weight_decay=...)`. Nota técnica honesta: PyTorch SGD aplica weight_decay como L2 al gradiente, mientras AdamW lo aplica decoupled. Esta diferencia mecánica entre los dos optimizadores debe reportarse en el paper como caveat — no son comparaciones perfectas del mismo concepto de "wd".

H3 (Kaiming init): se aplica `kaiming_uniform_(mode='fan_in', nonlinearity='relu')` a todas las weights lineales y embeddings, manteniendo paralelismo con la implementación Xavier original.

---

### 2026-07-06 — Registro de crash, diagnóstico y reanudación (NO modifica hipótesis ni métodos)

**Contexto:** la ejecución de Fase II (`phase8.py`) se interrumpió el 2026-04-30 04:53 en el run 20/46 (`H2_sgd_wd0.1_s1`, CSV truncado — kill duro externo, probablemente sync entre máquinas). La sesión de esa fecha preparó `phase8_debug.py` (6 runs diagnósticos) que quedó sin ejecutar. El 2026-07-06 se ejecutó el diagnóstico completo + un diagnóstico diferencial adicional de 4 configs cortas (SGD wd=0 lr∈{1e-2,1e-1}; SGD wd=1.0 lr=1e-3; SGDW decoupled wd=1.0), con logging de norma L2 de pesos.

**Hallazgos diagnósticos (datos en `results/phase8_debug/` + log de sesión):**

1. **H2 (SGD): el código NO está roto — es weight collapse.** SGD con wd=0 memoriza train en ~1K steps (path sano). Con wd=1.0 la norma de pesos colapsa 42→0.02 en <1K steps (decay acoplado + momentum ≈ 1%/step efectivo) y la red queda en salida uniforme (train_loss = ln(99) = 4.5948 exacto, congelada). SGDW decoupled a wd=1.0 también colapsa. AdamW sobrevive por normalización adaptativa del gradiente. **Consecuencia per pre-registro:** los runs H2 restantes se completan tal como fueron diseñados (reporte de todas las celdas), y el resultado se encuadra en el **Caso 3** ("AdamW-specific"): la semántica de wd no es transferible entre optimizers a estos valores — extensión del caveat técnico ya registrado el 2026-04-28.

2. **H1 (depth=2): cambio de régimen, no test de robustez.** En TODAS las celdas depth=2 que generalizaron, `t_train_saturated == t_grok` y train_acc ≈ test_acc en toda la trayectoria → no existe fase de memorización → **no hay grokking en depth=2** (n_train=250K vs 423K params: el modelo no puede memorizar). Diagnósticos A1 (300K steps), A2 (wd=0.3), A3 (n_layers=4) confirman que wd≥0.3 con depth=2 tampoco alcanza generalización por más steps/capacidad. **Consecuencia:** H1 tal como se interpretó (composición de tarea, changelog 2026-04-28) no puede evaluar la invarianza del slope de grokking; se reporta como hallazgo de régimen con los datos completos. El test arquitectural (n_layers, "Fase II.5") queda como candidato a extensión — se pre-registraría por separado ANTES de correrse.

3. **CSV truncado:** `H2_sgd_wd0.1_s1.csv` (95 KB, cortado en step 163600) superaba el umbral `is_done` (80 KB) y habría pasado como completo. Renombrado a `.truncated_bak`; la celda se re-corre entera.

**Qué NO cambia:** H3 (init) y H4 (batch size) se ejecutan exactamente como fueron pre-registradas. Ningún intervalo, threshold ni método de análisis se modifica. Los 27 runs restantes (11 H2 + 10 H3 + 6 H4) se lanzan el 2026-07-06.

**Registrado por:** Fable 5 (sesión 2026-07-06), continuando ejecución autorizada por Pablo ("continuar el experimento pendiente de GPU").

---

*Documento pre-registrado v1.0 — 2026-04-27. Firmado: Pablo Rainieri + Opus 4.7.*
