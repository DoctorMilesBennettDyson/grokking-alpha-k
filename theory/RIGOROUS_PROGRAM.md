# Paper 04 — Programa Riguroso para Publicación de Calidad Primarísima

**Versión:** v1.0 (2026-04-27)
**Autor:** Pablo Rainieri + Opus 4.7
**Estado:** activo — Fase I autorizada para ejecución
**Compromiso operativo:** ningún experimento de Fase II o posteriores se ejecuta sin pre-registro firmado y fechado.

---

## Premisa

Este programa NO existe para producir un paper rápido. Existe para producir un paper que sobreviva crítica de reviewer experto, replicación independiente, y escrutinio del campo de interpretabilidad de redes neuronales.

Aclaración explícita: **"irrefutable" en sentido absoluto no existe en ciencia.** Todo paper tiene limitaciones declaradas. Lo que perseguimos es **robusto bajo crítica razonable de reviewer experto**, replicable por terceros con código abierto, honesto sobre lo que no sabemos, y conectado a marco teórico verificable.

---

## Hallazgo central que motiva todo lo siguiente

De Fase 4-6 (ya ejecutadas), los datos sugieren:

```
La curva de potencia t_grok ∝ wd^(-α) tiene exponente
α que decrece con K(X) (complejidad de la tarea):
  p=31 (480 train pairs):  α ≈ 0.86
  p=97 (4704 train pairs): α ≈ 0.74
  p=149 (11100 train pairs): α ≈ 0.57
```

Esto NO es la "pendiente universal" que el framework de Paper 04 v0.1 predecía implícitamente. Es algo más rico: una **interacción K(X) × wd** que requiere extender el marco teórico.

Antes de publicar este hallazgo, hay que validarlo contra todas las objeciones razonables que un reviewer haría.

---

## Programa por Fases

### Fase I — Cerrar brechas empíricas evidentes

**Objetivo:** eliminar todas las celdas con N < 3 limpio o con DNF que dejen ambigüedad sobre la magnitud de los efectos. Producir curva continua slope(p) con 6 valores de p en lugar de 3.

**Componente A — Cierre de celdas frágiles:**

| celda | razón | runs nuevos |
|---|---|---|
| p=31, wd=0.01 | 1 DNF + 1 al borde de 200K | 5 seeds × 300K |
| p=31, wd=0.1 | N=2 limpio (1 outlier) | 3 seeds × 200K |
| p=97, wd=3.0 | N=2 limpio | 5 seeds × 200K |
| p=149, wd=3.0 | 1 outlier | 5 seeds × 200K |

Subtotal A: 18 runs, ~7 GPU hrs.

**Componente B — Llenar primos intermedios:**

| p | n_train | runs nuevos |
|---|---|---|
| 53 | 1404 | 5 wd × 3 seeds = 15 |
| 67 | 2244 | 5 wd × 3 seeds = 15 |
| 113 | 6384 | 5 wd × 3 seeds = 15 |

Subtotal B: 45 runs × 200K = ~15 GPU hrs.

**Total Fase I:** 63 runs, ~22 GPU hrs (2 noches).

**Criterio de éxito de Fase I (pre-registrado):**
- Todas las celdas con N ≥ 3 limpio o explícita razón documentada.
- Curva slope(p) trazable con 6 puntos: p ∈ {31, 53, 67, 97, 113, 149}.
- Si la curva slope(p) es monotónica no creciente, la observación de Phase 6 se confirma.
- Si la curva no es monotónica, la observación se complica (caso interesante, posiblemente otro paper).

**Output:** `results/phase7/REPORT.md` con tabla cruzada 6 primos × 5 wd.

---

### Fase II — Robustness checks

**REQUISITO ABSOLUTO:** se ejecuta solo después de firmar y fechar `PRE_REGISTRATION_FASE_II.md`. Ese documento ya está escrito, fija las hipótesis cuantitativas a testear, y bloquea el análisis a métodos pre-especificados.

**Objetivo:** descartar que el patrón slope(K) sea artifact de:
- arquitectura específica (depth=1)
- optimizer específico (AdamW)
- esquema de inicialización (Xavier)
- batch_size específico (512)

**Experimentos:**

| variación | configuración | runs | GPU hrs |
|---|---|---|---|
| depth | depth=2 en p=97, todos wd, 3 seeds | 15 | ~9 |
| optimizer | SGD-momentum vs AdamW en p=97, todos wd, 3 seeds | 15 | ~9 |
| init | Kaiming vs Xavier en p=97 baseline (wd=1.0), 5 seeds c/u | 10 | ~2 |
| batch_size | bs ∈ {256, 512, 1024} en p=97 baseline, 3 seeds c/u | 9 | ~3 |

Total Fase II: 49 runs, ~25 GPU hrs.

**Criterios de éxito (pre-registrados en doc separado):**
- depth=2 produce slope dentro de [-0.94, -0.54] → patrón es depth-invariante
- SGD-momentum reproduce monotonicidad y signo del slope → patrón no es Adam-específico
- Kaiming vs Xavier difieren en t_grok absoluto < 50% → patrón no es init-trivial
- t_grok escala monotónicamente con batch_size → consistente con interpretación de wd vía L (no μ)

Cualquier violación de los criterios anteriores requiere análisis específico antes de claim de universalidad.

---

### Fase III — Marco teórico

**No requiere GPU.** Trabajo conceptual + matemático.

**Objetivo:** extender el framework `L·μ ≤ log(σ)` para predecir la dependencia α(K(X)) que observamos empíricamente.

**Tareas:**

1. **Derivación**: si wd modula L (complejidad de circuito) y K(X) determina cuánto signal hay disponible para gradientes, derivar matemáticamente por qué la sensibilidad de t_grok a L debería decrecer con K(X). Borrador en `paper_04_emergence_prediction/theory_extension_v0.1.md`.

2. **Comparación cuantitativa con literatura**:
   - Power et al. 2022 (paper original de grokking)
   - Nanda et al. 2023 (mechanistic interpretability of grokking)
   - Liu et al. 2022 (grokking as phase transition)
   - Para cada uno: extraer números reportados, ubicar nuestros números en su mapa, identificar discrepancias y armonías.

3. **Predicción nueva**: el marco extendido debe generar al menos UNA predicción no testeada en Fases I-II. Esa predicción se testea en Fase IV.

**Estimación:** 2-4 semanas de pensamiento + lectura intensiva. NO se puede acelerar con más cómputo.

---

### Fase IV — Validación externa + replicación independiente

**Objetivo:** confirmar que los resultados no son artifact de PyTorch, de mi código específico, o de mi mano sesgada.

**Tareas:**

1. **Re-implementación en JAX**: re-escribir el train loop minimal en JAX/Flax. Correr p=97 wd=1.0 baseline + p=97 wd=0.01. Si los t_grok caen dentro del IC95% de los valores reportados, descartamos bug PyTorch-específico.

2. **Pre-print en arXiv**: subir manuscrito completo + código + datos crudos. Esto fija una versión pública verificable.

3. **Solicitar review externa**: contactar a 2-3 investigadores activos en grokking/interpretability (candidatos: alguien de Anthropic interpretability team, alguien del entorno de Power et al., alguien de la comunidad mech-interp). Pedirles 30 minutos de feedback antes de submission formal.

**Estimación:** 4-8 semanas.

---

### Fase V — Pre-registro formal de Fase II (escrito ANTES de Fase II)

**Estado:** documento `PRE_REGISTRATION_FASE_II.md` ya redactado y fechado.
**Por qué crítico:** sin pre-registro escrito antes de mirar los datos, cualquier análisis post-hoc es susceptible a p-hacking inconsciente. El pre-registro no es burocracia; es la diferencia entre ciencia y narrativa.

---

## Calendario tentativo

| Semana | Actividad |
|---|---|
| 1-2 | Fase I (ejecución autónoma + análisis) |
| 2 | Lectura de Fase I results, validación contra hipótesis de Phase 6 |
| 3-4 | Fase II ejecución (dependiente de pre-registro firmado) |
| 4 | Análisis de Fase II contra criterios pre-registrados |
| 5-8 | Fase III (trabajo teórico; SIN GPU) |
| 9-12 | Fase IV (replicación externa, pre-print, feedback) |
| 13-16 | Revisiones post-feedback, submission a venue formal |

**Total: 3-5 meses calendario.** Cualquier intento de comprimir esto sacrifica calidad.

---

## Reglas de honestidad operativa

Estas reglas valen para todo el programa. Romper una de ellas invalida el rigor.

1. **No re-mirar datos para "ajustar" hipótesis post-hoc.** Una vez que la hipótesis está pre-registrada, la conclusión es la que digan los datos según el análisis pre-registrado.

2. **Reportar TODOS los runs**, incluso DNFs, outliers, y experimentos abandonados. La supresión selectiva es lo que hace papers irreproducibles.

3. **No inflar.** Si la pendiente p=149 es -0.57 y no -0.7, decir -0.57. Si N=2, decir N=2.

4. **Sin "narrative twist" de último momento.** Si los datos no soportan la historia, cambiar la historia, no los datos.

5. **Pre-registro vinculante.** Si Fase II revela que slope cambia signo con depth=2, eso refuta universalidad. No reformulamos para "salvar" el resultado; reportamos honestamente lo que vimos.

6. **Limitations section honesta.** Toda limitación detectada en cualquier fase entra al paper, no se esconde.

7. **Código y datos abiertos**, sin curación post-hoc. Lo que el daemon produjo es lo que se publica.

---

## Estado actual y próximo paso operativo

- Fases 1-6: ejecutadas. Datos en `code/grokking/results/phase{1-6}/`.
- Fase I (Phase 7 script): script listo, autorizado para ejecución.
- Pre-registro de Fase II: redactado y fechado 2026-04-27 antes de cualquier ejecución de Fase II.
- Próximo paso: lanzar `phase7.py` → 22 horas → revisar REPORT → autorizar Fase II.

---

*v1.0 — 2026-04-27. Documento vivo. Cualquier modificación a este programa requiere registro fechado en sección de changelog (a agregarse al pie cuando ocurra la primera modificación).*
