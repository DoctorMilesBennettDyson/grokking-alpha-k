"""
phase7.py -- Fase I del programa riguroso (RIGOROUS_PROGRAM.md).

Dos componentes secuenciales:
  A) Cierre de celdas frágiles de Fases 4-6:
     - p=31, wd=0.01: 5 seeds adicionales (10-14), 300K steps (1 DNF previo + 1 al borde)
     - p=31, wd=0.1:  3 seeds adicionales (10-12), 200K steps (N=2 limpio)
     - p=97, wd=3.0:  5 seeds adicionales (10-14), 200K steps (N=2 limpio)
     - p=149, wd=3.0: 5 seeds adicionales (10-14), 200K steps (1 outlier)
  B) Llenado de primos intermedios (curva continua slope(p)):
     - p=53, wd ∈ {0.01, 0.1, 0.3, 1.0, 3.0}, seeds {0,1,2}, 200K
     - p=67, mismo
     - p=113, mismo

Total: 18 + 45 = 63 runs, ~22 horas en RTX 3070.

Idempotente. Crash-safe. Auto-reporta.

Usage:
    python phase7.py
    python phase7.py --analyze    # solo análisis (asume runs hechos)

Monitor (sin Claude):
    python progress.py --phase 7
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train import train_single_run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = _HERE / "results/phase7"
PROGRESS_FILE = OUTPUT_DIR / "PROGRESS.txt"
COMPLETION_FLAG = OUTPUT_DIR / "PHASE7_COMPLETE.flag"
SUMMARY_JSON = OUTPUT_DIR / "phase7_summary.json"
REPORT_MD = OUTPUT_DIR / "REPORT.md"
LOG_FILE = OUTPUT_DIR / "phase7.log"

MIN_CSV_SIZE = 80_000  # bytes (200K-step run ≈ 200KB; 300K ≈ 300KB)

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

COMMON_KWARGS = dict(
    depth=1,
    dropout=0.0,
    label_smoothing=0.0,
    lr=1e-3,
    batch_size=512,
    train_frac=0.5,
    eval_every=100,
    grok_threshold=0.95,
    warmup_steps=100,
)

# Component A: cierre de celdas frágiles
WEAK_CELLS = [
    # (p, wd, [seeds], steps)
    (31, 0.01, [10, 11, 12, 13, 14], 300_000),  # 1 DNF + 1 al borde de 200K
    (31, 0.1,  [10, 11, 12],         200_000),  # N=2 limpio en Phase 6
    (97, 3.0,  [10, 11, 12, 13, 14], 200_000),  # N=2 limpio en Phase 5
    (149, 3.0, [10, 11, 12, 13, 14], 200_000),  # 1 outlier en Phase 6
]

# Component B: primos intermedios
INTERMEDIATE_PRIMES = [53, 67, 113]
INTERMEDIATE_WD_VALUES = [0.01, 0.1, 0.3, 1.0, 3.0]
INTERMEDIATE_SEEDS = [0, 1, 2]
INTERMEDIATE_STEPS = 200_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_run_list():
    """Return list of run kwargs in execution priority order."""
    runs = []
    # Component A first (high priority: close existing weak cells)
    for p, wd, seeds, steps in WEAK_CELLS:
        for s in seeds:
            name = f"p{p}_wd{wd}_s{s}"
            kwargs = dict(COMMON_KWARGS, p=p, weight_decay=wd, seed=s,
                          steps=steps, run_name=name)
            kwargs["__group"] = "A_close_weak"
            runs.append(kwargs)
    # Component B: intermediate primes
    for p in INTERMEDIATE_PRIMES:
        for wd in INTERMEDIATE_WD_VALUES:
            for s in INTERMEDIATE_SEEDS:
                name = f"p{p}_wd{wd}_s{s}"
                kwargs = dict(COMMON_KWARGS, p=p, weight_decay=wd, seed=s,
                              steps=INTERMEDIATE_STEPS, run_name=name)
                kwargs["__group"] = "B_intermediate"
                runs.append(kwargs)
    return runs


def is_done(run_name):
    f = OUTPUT_DIR / f"{run_name}.csv"
    return f.exists() and f.stat().st_size >= MIN_CSV_SIZE


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_progress(done, total, current=None, eta_s=None):
    pct = 100.0 * done / total if total else 0
    parts = [
        f"Phase 7 (Fase I) progress: {done}/{total} ({pct:.0f}%)",
        f"updated: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if current:
        parts.append(f"running: {current}")
    if eta_s and eta_s > 0:
        h = eta_s / 3600
        parts.append(f"eta: {h:.1f}h" if h >= 1 else f"eta: {eta_s/60:.0f}m")
    parts.append("COMPLETE" if done >= total else "IN_PROGRESS")
    PROGRESS_FILE.write_text("\n".join(parts) + "\n", encoding="utf-8")


def load_summaries():
    if SUMMARY_JSON.exists():
        try:
            return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_summaries(summaries):
    seen = {}
    for r in summaries:
        seen[r.get("run_name")] = r
    deduped = list(seen.values())
    SUMMARY_JSON.write_text(json.dumps(deduped, indent=2, default=str),
                            encoding="utf-8")
    return deduped


# ---------------------------------------------------------------------------
# Sustained threshold (same as Phase 6)
# ---------------------------------------------------------------------------

def t_grok_sustained(csv_path, threshold=0.95, n_consecutive=3):
    try:
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((int(row["step"]), float(row["test_acc"])))
    except Exception:
        return None
    window = []
    for step, acc in rows:
        if acc >= threshold:
            window.append(step)
            if len(window) >= n_consecutive:
                return window[0]
        else:
            window = []
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = build_run_list()
    total = len(runs)
    pending_count = sum(1 for r in runs if not is_done(r["run_name"]))
    already = total - pending_count

    # Estimate hours roughly: weighted by steps
    total_steps_pending = sum(r["steps"] for r in runs if not is_done(r["run_name"]))
    est_hours = total_steps_pending * 6 / 1000 / 3600  # ~6s per 1K steps

    log("=" * 60)
    log("PHASE 7 (FASE I) START")
    log(f"  total runs: {total}  |  done: {already}  |  pending: {pending_count}")
    log(f"  estimated time remaining: {est_hours:.1f} hours")
    log("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    summaries = load_summaries()
    wall_times = []
    write_progress(already, total)

    for idx, kw in enumerate(runs):
        name = kw["run_name"]
        if is_done(name):
            continue

        done_so_far = sum(1 for r in runs[:idx+1] if is_done(r["run_name"]))
        remaining = total - done_so_far
        avg_t = sum(wall_times) / len(wall_times) if wall_times else None
        eta = avg_t * remaining if avg_t else None
        write_progress(done_so_far, total, current=name, eta_s=eta)

        group = kw.pop("__group", "?")
        log(f"--- [{done_so_far+1}/{total}] {name}  group={group}  steps={kw['steps']}  wd={kw['weight_decay']} ---")
        t0 = time.time()
        try:
            result = train_single_run(
                output_dir=str(OUTPUT_DIR),
                device=device,
                **kw,
            )
            result["weight_decay"] = kw["weight_decay"]
            result["group"] = group
            csv_path = OUTPUT_DIR / f"{name}.csv"
            result["t_grok_sustained"] = t_grok_sustained(str(csv_path))
            summaries.append(result)
            summaries = save_summaries(summaries)
        except Exception as exc:
            log(f"ERROR {name}: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.time() - t0
        wall_times.append(elapsed)
        log(f"    done {elapsed:.0f}s | t_grok={result.get('t_grok')} | "
            f"sustained={result.get('t_grok_sustained')} | "
            f"acc={result.get('final_test_acc'):.4f}")

    write_progress(total, total)
    log("=" * 60)
    log("PHASE 7 (FASE I) RUNS COMPLETE")
    log("=" * 60)


# ---------------------------------------------------------------------------
# Analysis: combine Phase 4-7 to produce comprehensive Fase I report
# ---------------------------------------------------------------------------

def _load_summary(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _t_grok_sustained_for(csv_path):
    return t_grok_sustained(str(csv_path)) if Path(csv_path).exists() else None


def _collect_wd_data():
    """
    Return dict {p: {wd: [list of (t_grok_first, t_grok_sustained, acc, source)]}}
    Combining Phase 4, 5, 6, 7.
    """
    data = {}

    # Phase 4 (only p=97 baseline + dropout/ls sweep). Only baseline is wd=1.0.
    p4 = _load_summary(_HERE / "results/phase4/phase4_summary.json")
    for r in p4:
        if r.get("dropout") == 0.0 and r.get("label_smoothing") == 0.0:
            csv_path = _HERE / "results/phase4" / f"dropout0.0_ls0.0_s{r.get('seed')}.csv"
            data.setdefault(97, {}).setdefault(1.0, []).append({
                "t_grok_first": r.get("t_grok"),
                "t_grok_sustained": _t_grok_sustained_for(csv_path),
                "acc": r.get("final_test_acc"),
                "source": "phase4",
                "seed": r.get("seed"),
            })

    # Phase 5 (p=97: rebaseline wd=1.0 + wd sweep)
    p5 = _load_summary(_HERE / "results/phase5/phase5_summary.json")
    for r in p5:
        rname = r.get("run_name", "")
        if r.get("group") == "rebaseline":
            wd = 1.0
        else:
            try:
                wd = float(rname.split("_")[0][2:])
            except (ValueError, IndexError):
                continue
        csv_path = _HERE / "results/phase5" / f"{rname}.csv"
        data.setdefault(97, {}).setdefault(wd, []).append({
            "t_grok_first": r.get("t_grok"),
            "t_grok_sustained": r.get("t_grok_sustained") or _t_grok_sustained_for(csv_path),
            "acc": r.get("final_test_acc"),
            "source": "phase5",
            "seed": r.get("seed"),
        })

    # Phase 6 (p=31, p=149)
    p6 = _load_summary(_HERE / "results/phase6/phase6_summary.json")
    for r in p6:
        p = r.get("p")
        wd = r.get("weight_decay")
        rname = r.get("run_name", "")
        if wd is None and "_wd" in rname:
            try:
                wd = float(rname.split("_wd")[1].split("_")[0])
            except (ValueError, IndexError):
                continue
        if p is None or wd is None:
            continue
        data.setdefault(p, {}).setdefault(wd, []).append({
            "t_grok_first": r.get("t_grok"),
            "t_grok_sustained": r.get("t_grok_sustained"),
            "acc": r.get("final_test_acc"),
            "source": "phase6",
            "seed": r.get("seed"),
        })

    # Phase 7 (cell-closing for p=31,97,149 + intermediate p=53,67,113)
    p7 = load_summaries()
    for r in p7:
        p = r.get("p")
        wd = r.get("weight_decay")
        if p is None or wd is None:
            continue
        data.setdefault(p, {}).setdefault(wd, []).append({
            "t_grok_first": r.get("t_grok"),
            "t_grok_sustained": r.get("t_grok_sustained"),
            "acc": r.get("final_test_acc"),
            "source": "phase7",
            "seed": r.get("seed"),
        })

    return data


def _is_clean(run, sustained=True):
    """Run is clean if final_test_acc >= 0.95 and t_grok exists."""
    acc = run.get("acc")
    tg = run.get("t_grok_sustained") if sustained else run.get("t_grok_first")
    return tg is not None and acc is not None and acc >= 0.95


def _stats(values):
    if not values:
        return None, None, None, None
    n = len(values)
    m = sum(values) / n
    s = (sum((x - m) ** 2 for x in values) / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0
    return m, s, min(values), max(values)


def _power_law(wd_means):
    import math
    pts = [(math.log10(wd), math.log10(m)) for wd, m in wd_means if m and m > 0]
    if len(pts) < 2:
        return None, None
    n = len(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x - mx)*(y - my) for x, y in zip(xs, ys))
    den = sum((x - mx)**2 for x in xs)
    if den == 0:
        return None, None
    slope = num / den
    intercept = my - slope * mx
    return slope, intercept


def analyze():
    data = _collect_wd_data()
    if not data:
        log("No data to analyze.")
        return

    primes_sorted = sorted(data.keys())
    wd_values_all = sorted({wd for d in data.values() for wd in d.keys()})

    lines = []
    lines.append("# Fase I — REPORT (Phase 7 + integración con Fases 4-6)\n\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append("## Componentes de Fase I\n\n")
    lines.append("- A) Cierre de celdas frágiles (p=31 wd=0.01/0.1, p=97 wd=3.0, p=149 wd=3.0)\n")
    lines.append("- B) Llenado de primos intermedios (p=53, 67, 113 con sweep completo)\n\n")
    lines.append("Combinado con datos de Fases 4-6, este reporte produce la curva slope(p) ")
    lines.append("con 6 puntos: p ∈ {31, 53, 67, 97, 113, 149}.\n\n")

    # Per-p tables with N counts
    lines.append("## Tabla maestra: t_grok_sustained por (p, wd) — todas las fases combinadas\n\n")
    lines.append("Cada celda: mean (std) [N_clean / N_total]\n\n")
    header = "| p \\ wd | " + " | ".join(str(wd) for wd in wd_values_all) + " |\n"
    sep = "|---" * (len(wd_values_all) + 1) + "|\n"
    lines.append(header)
    lines.append(sep)

    slopes_per_p = {}
    for p in primes_sorted:
        cells = []
        wd_means = []
        for wd in wd_values_all:
            runs = data.get(p, {}).get(wd, [])
            clean_runs = [r for r in runs if _is_clean(r)]
            n_total = len(runs)
            n_clean = len(clean_runs)
            if not clean_runs:
                cells.append(f"DNF [{n_total}]" if n_total else "—")
                continue
            tg_vals = [r["t_grok_sustained"] for r in clean_runs]
            m, s, mn, mx = _stats(tg_vals)
            cells.append(f"{m:.0f} ({s:.0f}) [{n_clean}/{n_total}]")
            wd_means.append((wd, m))
        slope, intercept = _power_law(wd_means)
        slopes_per_p[p] = slope
        lines.append(f"| **p={p}** | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # Slope summary
    lines.append("## Curva slope(p) — Fase I result\n\n")
    lines.append("| p | n_train | slope (log-log) |\n")
    lines.append("|---|---|---|\n")
    n_train_lookup = {31: 480, 53: 1404, 67: 2244, 97: 4704, 113: 6384, 149: 11100}
    for p in primes_sorted:
        slope = slopes_per_p.get(p)
        slope_str = f"{slope:.3f}" if slope is not None else "—"
        nt = n_train_lookup.get(p, "?")
        lines.append(f"| {p} | {nt} | {slope_str} |\n")
    lines.append("\n")

    # Monotonicity check
    valid_slopes = [(p, slopes_per_p[p]) for p in primes_sorted if slopes_per_p.get(p) is not None]
    if len(valid_slopes) >= 3:
        ps_only = [s for _, s in valid_slopes]
        # Check if |slope| is monotonically decreasing with p
        abs_slopes = [abs(s) for s in ps_only]
        is_mono_decreasing = all(abs_slopes[i] >= abs_slopes[i+1] for i in range(len(abs_slopes)-1))
        is_mono_increasing = all(abs_slopes[i] <= abs_slopes[i+1] for i in range(len(abs_slopes)-1))
        lines.append("### Monotonicidad de |slope| con p\n\n")
        if is_mono_decreasing:
            lines.append("|slope| es **monotónicamente NO creciente** con p. ")
            lines.append("→ Patrón observado en Phase 6 se confirma con curva de 6 puntos. ")
            lines.append("La interacción K(X) × wd es real.\n\n")
        elif is_mono_increasing:
            lines.append("|slope| es **monotónicamente NO decreciente** con p. ")
            lines.append("→ Esto contradice el patrón observado en Phase 6. ")
            lines.append("Análisis adicional necesario antes de cualquier claim.\n\n")
        else:
            lines.append("|slope| **NO es monotónica** con p. ")
            lines.append("→ Caso interesante. La función slope(p) tiene estructura no trivial. ")
            lines.append("Esto puede ser señal de un mecanismo distinto al simple K(X) × wd, ")
            lines.append("o ruido en el estimador con N=3 en algunas celdas.\n\n")

    # Detailed per-cell run-level data
    lines.append("## Run-level data (todas las celdas, todas las fases)\n\n")
    lines.append("Útil para revisar outliers y DNFs.\n\n")
    for p in primes_sorted:
        for wd in wd_values_all:
            runs = data.get(p, {}).get(wd, [])
            if not runs:
                continue
            lines.append(f"### p={p}, wd={wd}\n\n")
            lines.append("| seed | source | t_grok_first | t_grok_sustained | acc | clean? |\n")
            lines.append("|---|---|---|---|---|---|\n")
            for r in sorted(runs, key=lambda x: (x.get("source", ""), x.get("seed", -1))):
                clean = "yes" if _is_clean(r) else "NO"
                tgf = r.get("t_grok_first")
                tgs = r.get("t_grok_sustained")
                acc = r.get("acc")
                acc_str = f"{acc:.4f}" if acc is not None else "—"
                lines.append(f"| {r.get('seed')} | {r.get('source')} | "
                             f"{tgf if tgf is not None else 'DNF'} | "
                             f"{tgs if tgs is not None else 'DNF'} | "
                             f"{acc_str} | {clean} |\n")
            lines.append("\n")

    # Recommendations
    lines.append("## Próximo paso operativo\n\n")
    lines.append("1. Revisar la tabla maestra para confirmar que cada celda tiene N_clean ≥ 3.\n")
    lines.append("2. Si alguna celda crítica tiene N_clean < 3, considerar Phase 7.5 ad-hoc.\n")
    lines.append("3. Si todas las celdas son robustas, verificar que `PRE_REGISTRATION_FASE_II.md` ")
    lines.append("está commiteado y firmado.\n")
    lines.append("4. Lanzar Fase II (script separado, dependiente de pre-registro).\n\n")

    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    log(f"REPORT written: {REPORT_MD.absolute()}")
    COMPLETION_FLAG.write_text(
        f"Phase 7 (Fase I) complete at {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()
    if not args.analyze:
        run_all()
    analyze()
    log("Phase 7 (Fase I) done.")


if __name__ == "__main__":
    main()
