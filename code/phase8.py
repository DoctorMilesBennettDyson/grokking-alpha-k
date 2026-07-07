"""
phase8.py -- Fase II: Robustness checks (PRE_REGISTRATION_FASE_II.md).

Cuatro hipótesis pre-registradas:
  H1 — depth (composición de tarea): p=97, depth=2, sweep wd, 3 seeds   = 15 runs
  H2 — optimizer (SGD vs AdamW):     p=97, depth=1, SGD, sweep wd, 3 seeds = 15 runs
  H3 — init (Kaiming vs Xavier):     p=97, depth=1, wd=1.0, 5 seeds c/u   = 10 runs
  H4 — batch_size:                   p=97, depth=1, wd=1.0, bs ∈ {256, 1024}, 3 seeds c/u = 6 runs
                                     (bs=512 ya cubierto por baseline existente)
Total: 15+15+10+6 = 46 runs (no los 49 originales — bs=512 ya está en datos previos).
Estimado: ~9 horas en RTX 3070.

Idempotente. Crash-safe. Auto-reporta contra los criterios cuantitativos pre-registrados.

Usage:
    python phase8.py
    python phase8.py --analyze    # solo análisis

Monitor:
    python progress.py --phase 8
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

from train_v2 import train_single_run_v2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = _HERE / "results/phase8"
PROGRESS_FILE = OUTPUT_DIR / "PROGRESS.txt"
COMPLETION_FLAG = OUTPUT_DIR / "PHASE8_COMPLETE.flag"
SUMMARY_JSON = OUTPUT_DIR / "phase8_summary.json"
REPORT_MD = OUTPUT_DIR / "REPORT.md"
LOG_FILE = OUTPUT_DIR / "phase8.log"

MIN_CSV_SIZE = 80_000

# ---------------------------------------------------------------------------
# Run configurations (each list builds run kwargs)
# ---------------------------------------------------------------------------

WD_VALUES = [0.01, 0.1, 0.3, 1.0, 3.0]
SEEDS_3 = [0, 1, 2]
SEEDS_5_INIT = [20, 21, 22, 23, 24]  # new seeds per pre-reg

COMMON_BASE = dict(
    p=97,
    dropout=0.0,
    label_smoothing=0.0,
    train_frac=0.5,
    eval_every=100,
    grok_threshold=0.95,
    warmup_steps=100,
    steps=200_000,
)


def build_run_list():
    runs = []

    # H1 — depth=2 (task composition), AdamW, Xavier, bs=512
    for wd in WD_VALUES:
        for s in SEEDS_3:
            name = f"H1_depth2_wd{wd}_s{s}"
            kw = dict(COMMON_BASE,
                      depth=2, weight_decay=wd, seed=s, lr=1e-3,
                      batch_size=512, optimizer='adamw', init_scheme='xavier',
                      run_name=name)
            kw['__h'] = 'H1'
            runs.append(kw)

    # H2 — SGD-momentum, depth=1, Xavier, bs=512
    for wd in WD_VALUES:
        for s in SEEDS_3:
            name = f"H2_sgd_wd{wd}_s{s}"
            kw = dict(COMMON_BASE,
                      depth=1, weight_decay=wd, seed=s, lr=1e-2,
                      batch_size=512, optimizer='sgd_momentum', init_scheme='xavier',
                      run_name=name)
            kw['__h'] = 'H2'
            runs.append(kw)

    # H3 — Init comparison: Xavier vs Kaiming at baseline (wd=1.0, depth=1, AdamW)
    for init in ['xavier', 'kaiming']:
        for s in SEEDS_5_INIT:
            name = f"H3_init{init}_s{s}"
            kw = dict(COMMON_BASE,
                      depth=1, weight_decay=1.0, seed=s, lr=1e-3,
                      batch_size=512, optimizer='adamw', init_scheme=init,
                      run_name=name)
            kw['__h'] = 'H3'
            runs.append(kw)

    # H4 — batch_size sweep at baseline (wd=1.0, depth=1, AdamW, Xavier)
    # bs=512 NOT included because already covered by Phase 4+5 baseline (N=7)
    for bs in [256, 1024]:
        for s in SEEDS_3:
            name = f"H4_bs{bs}_s{s}"
            kw = dict(COMMON_BASE,
                      depth=1, weight_decay=1.0, seed=s, lr=1e-3,
                      batch_size=bs, optimizer='adamw', init_scheme='xavier',
                      run_name=name)
            kw['__h'] = 'H4'
            runs.append(kw)

    return runs


# ---------------------------------------------------------------------------
# Helpers (mirror phase7 patterns)
# ---------------------------------------------------------------------------

def is_done(name):
    f = OUTPUT_DIR / f"{name}.csv"
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
        f"Phase 8 (Fase II) progress: {done}/{total} ({pct:.0f}%)",
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
    SUMMARY_JSON.write_text(json.dumps(deduped, indent=2, default=str), encoding="utf-8")
    return deduped


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

    log("=" * 60)
    log("PHASE 8 (FASE II) START")
    log(f"  total: {total}  done: {already}  pending: {pending_count}")
    breakdown = {}
    for r in runs:
        breakdown[r['__h']] = breakdown.get(r['__h'], 0) + 1
    for h, n in sorted(breakdown.items()):
        log(f"    {h}: {n} runs")
    est_h = pending_count * 700 / 3600
    log(f"  estimated time remaining: ~{est_h:.1f}h")
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

        h_tag = kw.pop('__h')
        log(f"--- [{done_so_far+1}/{total}] {name} ({h_tag}) ---")
        t0 = time.time()
        try:
            result = train_single_run_v2(
                output_dir=str(OUTPUT_DIR),
                device=device,
                **kw,
            )
            result["__h"] = h_tag
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
            f"acc={result.get('final_test_acc'):.4f} | "
            f"diverged={result.get('diverged', False)}")

    write_progress(total, total)
    log("PHASE 8 RUNS COMPLETE")


# ---------------------------------------------------------------------------
# Analysis: evaluate against pre-registered hypotheses
# ---------------------------------------------------------------------------

def _stats(values):
    if not values:
        return None, None
    n = len(values)
    m = sum(values) / n
    s = (sum((x - m) ** 2 for x in values) / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0
    return m, s


def _power_law_slope(wd_means):
    import math
    pts = [(math.log10(wd), math.log10(m)) for wd, m in wd_means if m and m > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x - mx)*(y - my) for x, y in zip(xs, ys))
    den = sum((x - mx)**2 for x in xs)
    return num / den if den else None


def _is_clean(r):
    tg = r.get("t_grok_sustained")
    acc = r.get("final_test_acc")
    return tg is not None and acc is not None and acc >= 0.95


def analyze():
    summaries = load_summaries()
    if not summaries:
        log("No data.")
        return

    by_h = {}
    for r in summaries:
        by_h.setdefault(r.get("__h", "?"), []).append(r)

    lines = []
    lines.append("# Phase 8 (Fase II) — REPORT\n\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append("## Pre-registered hypotheses (from PRE_REGISTRATION_FASE_II.md)\n\n")
    lines.append("Each section evaluates the observed result against the pre-fixed criteria. ")
    lines.append("No post-hoc adjustment of intervals or thresholds.\n\n")

    # --- H1: depth=2, slope should fall in [-0.94, -0.54] ---
    lines.append("## H1 — Depth (task composition) invariance\n\n")
    h1_runs = by_h.get("H1", [])
    h1_groups = {}
    for r in h1_runs:
        h1_groups.setdefault(r["weight_decay"], []).append(r)
    lines.append("### Per-wd table (depth=2)\n\n")
    lines.append("| wd | mean t_grok_sustained | std | N_clean / N_total |\n")
    lines.append("|---|---|---|---|\n")
    h1_wd_means = []
    for wd in sorted(h1_groups.keys()):
        runs = h1_groups[wd]
        clean = [r for r in runs if _is_clean(r)]
        if clean:
            tg = [r["t_grok_sustained"] for r in clean]
            m, s = _stats(tg)
            lines.append(f"| {wd} | {m:.0f} | {s:.0f} | {len(clean)}/{len(runs)} |\n")
            h1_wd_means.append((wd, m))
        else:
            lines.append(f"| {wd} | — | — | 0/{len(runs)} |\n")
    h1_slope = _power_law_slope(h1_wd_means)
    if h1_slope is not None:
        in_interval = -0.94 <= h1_slope <= -0.54
        verdict = "CONFIRMED" if in_interval else "REJECTED"
        lines.append(f"\n**Observed slope: {h1_slope:.3f}**\n")
        lines.append(f"**Pre-registered interval: [-0.94, -0.54]**\n")
        lines.append(f"**Verdict: H1 {verdict}**\n\n")
        lines.append(f"- depth=1 reference (Phase 4+5): slope = -0.74\n")
        lines.append(f"- depth=2 observed: slope = {h1_slope:.3f}\n")
        if in_interval:
            lines.append("- Patrón slope(wd) es robusto a composición de tarea (depth=1 → depth=2).\n\n")
        else:
            lines.append("- La dependencia con depth de composición es no trivial. ")
            lines.append("Limitar claims al depth=1 hasta análisis adicional.\n\n")
    else:
        lines.append("\n**No data sufficient to estimate slope.**\n\n")

    # --- H2: SGD slope should be < 0 (sign preservation) ---
    lines.append("## H2 — Optimizer dependence (SGD vs AdamW)\n\n")
    h2_runs = by_h.get("H2", [])
    h2_groups = {}
    for r in h2_runs:
        h2_groups.setdefault(r["weight_decay"], []).append(r)
    lines.append("### Per-wd table (SGD-momentum)\n\n")
    lines.append("| wd | mean t_grok_sustained | std | N_clean / N_total | diverged_count |\n")
    lines.append("|---|---|---|---|---|\n")
    h2_wd_means = []
    for wd in sorted(h2_groups.keys()):
        runs = h2_groups[wd]
        clean = [r for r in runs if _is_clean(r)]
        n_div = sum(1 for r in runs if r.get("diverged", False))
        if clean:
            tg = [r["t_grok_sustained"] for r in clean]
            m, s = _stats(tg)
            lines.append(f"| {wd} | {m:.0f} | {s:.0f} | {len(clean)}/{len(runs)} | {n_div} |\n")
            h2_wd_means.append((wd, m))
        else:
            lines.append(f"| {wd} | — | — | 0/{len(runs)} | {n_div} |\n")
    h2_slope = _power_law_slope(h2_wd_means)
    if h2_slope is not None:
        sign_ok = h2_slope < 0
        in_interval = -1.5 <= h2_slope <= -0.3
        verdict = "CONFIRMED (sign and range)" if (sign_ok and in_interval) else (
            "PARTIAL (sign OK, range outside)" if sign_ok else "REJECTED (sign change)")
        lines.append(f"\n**Observed slope (SGD): {h2_slope:.3f}**\n")
        lines.append(f"**Pre-registered: slope < 0 (qualitative); slope ∈ [-1.5, -0.3] (exploratory)**\n")
        lines.append(f"**Verdict: H2 {verdict}**\n\n")
        if not sign_ok:
            lines.append("- Sign change implica que el efecto de wd en grokking es Adam-específico. ")
            lines.append("Reformular paper como 'AdamW-driven phenomenon'.\n\n")
        elif not in_interval:
            lines.append("- Sign correcto pero magnitud fuera del rango exploratorio. ")
            lines.append("Reportar honestamente; SGD y AdamW tienen mecánicas distintas de wd.\n\n")
        else:
            lines.append("- Patrón cualitativo robusto al cambio de optimizer.\n\n")
    else:
        lines.append("\n**No data sufficient.**\n\n")

    # --- H3: Init comparison ---
    lines.append("## H3 — Init invariance (Xavier vs Kaiming)\n\n")
    h3_runs = by_h.get("H3", [])
    by_init = {"xavier": [], "kaiming": []}
    for r in h3_runs:
        if r.get("init_scheme") in by_init:
            by_init[r["init_scheme"]].append(r)
    lines.append("| init | mean t_grok_sustained | std | N_clean / N_total |\n")
    lines.append("|---|---|---|---|\n")
    means_by_init = {}
    for init in ["xavier", "kaiming"]:
        runs = by_init[init]
        clean = [r for r in runs if _is_clean(r)]
        if clean:
            tg = [r["t_grok_sustained"] for r in clean]
            m, s = _stats(tg)
            means_by_init[init] = m
            lines.append(f"| {init} | {m:.0f} | {s:.0f} | {len(clean)}/{len(runs)} |\n")
        else:
            lines.append(f"| {init} | — | — | 0/{len(runs)} |\n")
    if "xavier" in means_by_init and "kaiming" in means_by_init:
        ratio = means_by_init["kaiming"] / means_by_init["xavier"]
        within_50pct = 0.5 <= ratio <= 1.5
        verdict = "CONFIRMED" if within_50pct else "REJECTED"
        lines.append(f"\n**Ratio Kaiming / Xavier: {ratio:.2f}**\n")
        lines.append(f"**Pre-registered: ratio ∈ [0.5, 1.5] (within ±50%)**\n")
        lines.append(f"**Verdict: H3 {verdict}**\n\n")
        if not within_50pct:
            lines.append("- Init es confound mayor. Todos los claims deben anclar al esquema usado. ")
            lines.append("Considerar replicar slope sweep con Kaiming.\n\n")
        else:
            lines.append("- Init no es confound mayor para los efectos reportados.\n\n")

    # --- H4: batch_size monotonicity ---
    lines.append("## H4 — batch_size dependence\n\n")
    h4_runs = by_h.get("H4", [])
    by_bs = {}
    for r in h4_runs:
        by_bs.setdefault(r["batch_size"], []).append(r)
    # Add Phase 4/5 baseline as bs=512 reference (mean=1314 from prior analysis)
    by_bs[512] = "BASELINE_REFERENCE"  # marker
    lines.append("| bs | mean t_grok_sustained | std | N_clean / N_total |\n")
    lines.append("|---|---|---|---|\n")
    bs_means = {}
    for bs in sorted([k for k in by_bs.keys() if isinstance(k, int)]):
        if by_bs[bs] == "BASELINE_REFERENCE":
            # Use Phase 4+5 reference value
            lines.append(f"| {bs} | 1314 (ref) | 195 | 7/8 (Phase 4+5) |\n")
            bs_means[bs] = 1314
        else:
            runs = by_bs[bs]
            clean = [r for r in runs if _is_clean(r)]
            if clean:
                tg = [r["t_grok_sustained"] for r in clean]
                m, s = _stats(tg)
                bs_means[bs] = m
                lines.append(f"| {bs} | {m:.0f} | {s:.0f} | {len(clean)}/{len(runs)} |\n")
            else:
                lines.append(f"| {bs} | — | — | 0/{len(runs)} |\n")
    if len(bs_means) >= 2:
        ordered = sorted(bs_means.items())
        is_monotonic = all(ordered[i][1] <= ordered[i+1][1] for i in range(len(ordered)-1))
        verdict = "CONFIRMED (monotonic)" if is_monotonic else "REJECTED (non-monotonic)"
        lines.append(f"\n**Order: " + " → ".join(f"bs={k}: {v:.0f}" for k, v in ordered) + "**\n")
        lines.append(f"**Pre-registered: t_grok monotonically increasing with bs**\n")
        lines.append(f"**Verdict: H4 {verdict}**\n\n")

    # --- Meta: which case ---
    lines.append("## Meta-analysis: which decision case?\n\n")
    h1_ok = h1_slope is not None and -0.94 <= h1_slope <= -0.54
    h2_ok = h2_slope is not None and h2_slope < 0
    h3_ok = ("xavier" in means_by_init and "kaiming" in means_by_init
             and 0.5 <= means_by_init["kaiming"] / means_by_init["xavier"] <= 1.5)
    h4_ok = (len(bs_means) >= 2 and
             all(sorted(bs_means.items())[i][1] <= sorted(bs_means.items())[i+1][1]
                 for i in range(len(bs_means)-1)))
    n_ok = sum([h1_ok, h2_ok, h3_ok, h4_ok])
    lines.append(f"H1: {'OK' if h1_ok else 'REJECTED'}\n")
    lines.append(f"H2: {'OK' if h2_ok else 'REJECTED'}\n")
    lines.append(f"H3: {'OK' if h3_ok else 'REJECTED'}\n")
    lines.append(f"H4: {'OK' if h4_ok else 'REJECTED'}\n\n")
    lines.append(f"**{n_ok}/4 hypotheses confirmed.**\n\n")
    if n_ok == 4:
        lines.append("→ Caso 1 del pre-registro: todas confirmadas. ")
        lines.append("Pasamos a Fase III (marco teórico).\n")
    else:
        lines.append("→ Caso múltiple del pre-registro. Reformular claims según ")
        lines.append("PRE_REGISTRATION_FASE_II.md sección 'Decisiones de meta-análisis'.\n")

    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    log(f"REPORT written: {REPORT_MD.absolute()}")
    COMPLETION_FLAG.write_text(
        f"Phase 8 (Fase II) complete at {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()
    if not args.analyze:
        run_all()
    analyze()
    log("Phase 8 (Fase II) done.")


if __name__ == "__main__":
    main()
