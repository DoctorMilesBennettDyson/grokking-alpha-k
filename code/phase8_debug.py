"""
phase8_debug.py -- Diagnostic runs for Phase 8 failures.

Phase 8 revealed THREE problems:
  A. H1 con depth=2 y wd alto (1.0, 3.0): acc ≈ 0.01 (chance random, no aprende).
  B. H2 SGD-momentum con wd bajo (0.01, 0.1): acc ≈ 0.01 (chance random).
  C. H2 SGD ~6× más lento que AdamW.

Hypotheses to test (one run per hypothesis, single seed=42 for diagnostic):

  Problem A — Why H1 depth=2 wd alto fails:
    A1. depth=2 simplemente necesita más steps. Test: depth=2, wd=1.0, 300K steps.
    A2. wd=1.0 es demasiada regularización para depth=2. Test: depth=2, wd=0.3, 100K.
    A3. n_layers=2 insuficiente para depth=2. Test: depth=2, wd=1.0, n_layers=4, 100K.

  Problem B — Why H2 SGD wd bajo fails (B3 first as sanity check):
    B3. SANITY CHECK: SGD wd=1.0 baseline grokkea? Test: SGD, wd=1.0, lr=1e-2, 100K.
    B1. lr=1e-2 demasiado alta para SGD con wd bajo. Test: SGD, wd=0.01, lr=3e-3, 100K.
    B2. SGD necesita warmup más largo. Test: SGD, wd=0.01, lr=1e-2, warmup=2000, 100K.

Run order optimized: B3 first (sanity check), then A2/A3 (fast AdamW), then A1 (long), then B1/B2.
This way if B3 fails (SGD bug), we abort B1/B2 saving ~6h.

Total estimated time: ~12h on RTX 3070.

Usage:
    python phase8_debug.py
    python phase8_debug.py --analyze    # only re-run analysis

Monitor:
    python progress.py --phase debug
    (or just look at results/phase8_debug/PROGRESS.txt)
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

from train_v3 import train_single_run_v3

OUTPUT_DIR = _HERE / "results/phase8_debug"
PROGRESS_FILE = OUTPUT_DIR / "PROGRESS.txt"
COMPLETION_FLAG = OUTPUT_DIR / "PHASE8_DEBUG_COMPLETE.flag"
SUMMARY_JSON = OUTPUT_DIR / "phase8_debug_summary.json"
REPORT_MD = OUTPUT_DIR / "REPORT.md"
LOG_FILE = OUTPUT_DIR / "phase8_debug.log"

MIN_CSV_SIZE = 30_000  # smaller threshold (some runs are 50-100K steps)

COMMON_BASE = dict(
    p=97,
    seed=42,
    dropout=0.0,
    label_smoothing=0.0,
    train_frac=0.5,
    eval_every=100,
    grok_threshold=0.95,
    batch_size=512,
    init_scheme='xavier',
)

DEBUG_RUNS = [
    # Sanity check first: SGD with baseline wd should grokkea
    {
        "name": "debugB3_sgd_wd1.0_baseline",
        "hypothesis": "SGD-momentum funciona en absoluto con wd=1.0 baseline",
        "kw": dict(COMMON_BASE,
                   depth=1, weight_decay=1.0, steps=100_000,
                   optimizer='sgd_momentum', lr=1e-2, warmup_steps=100,
                   n_layers=2),
    },
    # Problem A diagnostics (AdamW, fast)
    {
        "name": "debugA2_depth2_wd0.3",
        "hypothesis": "depth=2 con wd menor (0.3) sí grokkea",
        "kw": dict(COMMON_BASE,
                   depth=2, weight_decay=0.3, steps=100_000,
                   optimizer='adamw', lr=1e-3, warmup_steps=100,
                   n_layers=2),
    },
    {
        "name": "debugA3_depth2_wd1.0_4layers",
        "hypothesis": "depth=2 wd=1.0 con n_layers=4 (más capacidad) sí grokkea",
        "kw": dict(COMMON_BASE,
                   depth=2, weight_decay=1.0, steps=100_000,
                   optimizer='adamw', lr=1e-3, warmup_steps=100,
                   n_layers=4),
    },
    {
        "name": "debugA1_depth2_wd1.0_long",
        "hypothesis": "depth=2 wd=1.0 simplemente necesita 300K steps (no 200K)",
        "kw": dict(COMMON_BASE,
                   depth=2, weight_decay=1.0, steps=300_000,
                   optimizer='adamw', lr=1e-3, warmup_steps=100,
                   n_layers=2),
    },
    # Problem B diagnostics (SGD, slow — only run if B3 succeeds)
    {
        "name": "debugB1_sgd_wd0.01_lr3e-3",
        "hypothesis": "SGD wd bajo necesita lr menor (3e-3 vs 1e-2)",
        "kw": dict(COMMON_BASE,
                   depth=1, weight_decay=0.01, steps=100_000,
                   optimizer='sgd_momentum', lr=3e-3, warmup_steps=100,
                   n_layers=2),
    },
    {
        "name": "debugB2_sgd_wd0.01_warmup2000",
        "hypothesis": "SGD wd bajo necesita warmup más largo (2000 vs 100)",
        "kw": dict(COMMON_BASE,
                   depth=1, weight_decay=0.01, steps=100_000,
                   optimizer='sgd_momentum', lr=1e-2, warmup_steps=2000,
                   n_layers=2),
    },
]


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


def write_progress(done, total, current=None, eta_s=None, note=""):
    pct = 100.0 * done / total if total else 0
    parts = [
        f"Phase 8 DEBUG progress: {done}/{total} ({pct:.0f}%)",
        f"updated: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if current:
        parts.append(f"running: {current}")
    if eta_s and eta_s > 0:
        h = eta_s / 3600
        parts.append(f"eta: {h:.1f}h" if h >= 1 else f"eta: {eta_s/60:.0f}m")
    if note:
        parts.append(f"note: {note}")
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


def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(DEBUG_RUNS)
    pending = sum(1 for r in DEBUG_RUNS if not is_done(r["name"]))
    already = total - pending

    log("=" * 60)
    log("PHASE 8 DEBUG START")
    log(f"  total: {total}  done: {already}  pending: {pending}")
    log("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    summaries = load_summaries()
    wall_times = []
    write_progress(already, total)

    sgd_baseline_works = None  # tracks B3 result for early-abort logic

    for idx, item in enumerate(DEBUG_RUNS):
        name = item["name"]
        if is_done(name):
            # Re-load result from JSON if available, to track sanity check verdict
            for r in summaries:
                if r.get("run_name") == name and name == "debugB3_sgd_wd1.0_baseline":
                    sgd_baseline_works = r.get("t_grok") is not None
            continue

        # Early-abort logic: if B3 (SGD sanity) failed, skip B1 and B2
        if name in ("debugB1_sgd_wd0.01_lr3e-3", "debugB2_sgd_wd0.01_warmup2000"):
            if sgd_baseline_works is False:
                log(f"SKIPPING {name}: SGD baseline (B3) did not grokkea, "
                    f"so SGD path is broken. Diagnostic complete for B-runs.")
                write_progress(idx, total, note="SGD broken; B1/B2 skipped")
                continue

        done_so_far = sum(1 for r in DEBUG_RUNS[:idx+1] if is_done(r["name"]))
        avg_t = sum(wall_times) / len(wall_times) if wall_times else None
        eta = avg_t * (total - done_so_far) if avg_t else None
        write_progress(done_so_far, total, current=name, eta_s=eta)

        log(f"--- [{done_so_far+1}/{total}] {name} ---")
        log(f"    Hypothesis: {item['hypothesis']}")
        t0 = time.time()
        try:
            result = train_single_run_v3(
                output_dir=str(OUTPUT_DIR),
                device=device,
                run_name=name,
                **item["kw"],
            )
            result["__hypothesis"] = item["hypothesis"]
            csv_path = OUTPUT_DIR / f"{name}.csv"
            result["t_grok_sustained"] = t_grok_sustained(str(csv_path))
            summaries.append(result)
            summaries = save_summaries(summaries)

            # Track B3 result
            if name == "debugB3_sgd_wd1.0_baseline":
                sgd_baseline_works = result.get("t_grok") is not None
                log(f"    SANITY CHECK B3 verdict: SGD baseline "
                    f"{'GROKKED' if sgd_baseline_works else 'FAILED'}")

        except Exception as exc:
            log(f"ERROR {name}: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.time() - t0
        wall_times.append(elapsed)
        log(f"    done {elapsed:.0f}s | t_grok={result.get('t_grok')} | "
            f"acc={result.get('final_test_acc'):.4f}")

    write_progress(total, total)
    log("PHASE 8 DEBUG COMPLETE")


def analyze():
    summaries = load_summaries()
    if not summaries:
        log("No data.")
        return

    by_name = {r.get("run_name"): r for r in summaries}

    lines = []
    lines.append("# Phase 8 DEBUG — REPORT\n\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append("## Verdicts per hypothesis\n\n")
    lines.append("| Run | Hypothesis | Result | t_grok | acc | time | Verdict |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    for item in DEBUG_RUNS:
        name = item["name"]
        r = by_name.get(name)
        if r is None:
            lines.append(f"| {name} | {item['hypothesis']} | NOT RUN | — | — | — | — |\n")
            continue
        tg = r.get("t_grok")
        acc = r.get("final_test_acc", 0)
        time_h = r.get("total_time_sec", 0) / 3600
        if tg is not None:
            verdict = "**HYPOTHESIS CONFIRMED**"
        elif acc < 0.5:
            verdict = "**HYPOTHESIS REJECTED** (no grokking, near chance)"
        else:
            verdict = "PARTIAL (some learning but no grokking)"
        lines.append(f"| {name} | {item['hypothesis']} | "
                     f"{'GROKKED' if tg else 'NOT GROKKED'} | "
                     f"{tg if tg else '—'} | {acc:.4f} | {time_h:.1f}h | {verdict} |\n")

    lines.append("\n## Diagnostic conclusions\n\n")

    # Problem A diagnosis
    lines.append("### Problem A: H1 depth=2 wd alto no aprende\n\n")
    a1 = by_name.get("debugA1_depth2_wd1.0_long")
    a2 = by_name.get("debugA2_depth2_wd0.3")
    a3 = by_name.get("debugA3_depth2_wd1.0_4layers")
    a_summary = []
    if a1:
        if a1.get("t_grok"):
            a_summary.append(f"- A1: depth=2 wd=1.0 SÍ grokkea con 300K steps (vs 200K original) → "
                             f"problema era step budget. Fix: aumentar steps a 300K para depth=2.")
        else:
            a_summary.append(f"- A1: depth=2 wd=1.0 NO grokkea ni con 300K steps (acc={a1.get('final_test_acc', 0):.3f}) → "
                             f"problema NO es step budget.")
    if a2:
        if a2.get("t_grok"):
            a_summary.append(f"- A2: depth=2 wd=0.3 SÍ grokkea (acc={a2.get('final_test_acc', 0):.3f}) → "
                             f"problema es wd alto + depth=2. Fix: usar wd menor para depth=2.")
        else:
            a_summary.append(f"- A2: depth=2 wd=0.3 tampoco grokkea (acc={a2.get('final_test_acc', 0):.3f}) → "
                             f"problema es estructural.")
    if a3:
        if a3.get("t_grok"):
            a_summary.append(f"- A3: depth=2 wd=1.0 con n_layers=4 SÍ grokkea → "
                             f"problema es capacidad arquitectural. Fix: n_layers=4 para depth=2.")
        else:
            a_summary.append(f"- A3: depth=2 wd=1.0 con n_layers=4 tampoco grokkea → "
                             f"capacidad no es el cuello de botella.")
    lines.extend(s + "\n" for s in a_summary)
    if not a_summary:
        lines.append("- No A diagnostics completed yet.\n")
    lines.append("\n")

    # Problem B diagnosis
    lines.append("### Problem B: H2 SGD wd bajo no aprende\n\n")
    b3 = by_name.get("debugB3_sgd_wd1.0_baseline")
    b1 = by_name.get("debugB1_sgd_wd0.01_lr3e-3")
    b2 = by_name.get("debugB2_sgd_wd0.01_warmup2000")
    b_summary = []
    if b3:
        if b3.get("t_grok"):
            b_summary.append(f"- B3 (sanity): SGD wd=1.0 baseline SÍ grokkea (t_grok={b3.get('t_grok')}) → "
                             f"path SGD funciona en general.")
        else:
            b_summary.append(f"- B3 (sanity): SGD wd=1.0 baseline NO grokkea (acc={b3.get('final_test_acc', 0):.3f}) → "
                             f"**SGD PATH ESTÁ ROTO en train_v2.py / train_v3.py**. "
                             f"Antes de continuar, debug del código SGD.")
    if b1:
        if b1.get("t_grok"):
            b_summary.append(f"- B1: SGD wd=0.01 con lr=3e-3 SÍ grokkea → "
                             f"problema era lr alta (1e-2 vs 3e-3). Fix: usar lr=3e-3 para SGD wd bajo.")
        else:
            b_summary.append(f"- B1: SGD wd=0.01 con lr=3e-3 tampoco grokkea (acc={b1.get('final_test_acc', 0):.3f}).")
    if b2:
        if b2.get("t_grok"):
            b_summary.append(f"- B2: SGD wd=0.01 con warmup=2000 SÍ grokkea → "
                             f"problema era warmup corto. Fix: usar warmup=2000 para SGD wd bajo.")
        else:
            b_summary.append(f"- B2: SGD wd=0.01 con warmup=2000 tampoco grokkea (acc={b2.get('final_test_acc', 0):.3f}).")
    lines.extend(s + "\n" for s in b_summary)
    if not b_summary:
        lines.append("- No B diagnostics completed yet.\n")
    lines.append("\n")

    # Problem C: timing
    lines.append("### Problem C: SGD ~6× más lento que AdamW\n\n")
    sgd_runs = [by_name[r["name"]] for r in DEBUG_RUNS
                if r["kw"]["optimizer"] == "sgd_momentum" and r["name"] in by_name]
    adamw_runs = [by_name[r["name"]] for r in DEBUG_RUNS
                  if r["kw"]["optimizer"] == "adamw" and r["name"] in by_name]
    if sgd_runs and adamw_runs:
        sgd_avg = sum(r.get("total_time_sec", 0) / max(r.get("steps", 1), 1) for r in sgd_runs) / len(sgd_runs)
        adamw_avg = sum(r.get("total_time_sec", 0) / max(r.get("steps", 1), 1) for r in adamw_runs) / len(adamw_runs)
        ratio = sgd_avg / adamw_avg if adamw_avg > 0 else 0
        lines.append(f"- SGD avg: {sgd_avg*1000:.2f}s/1K steps\n")
        lines.append(f"- AdamW avg: {adamw_avg*1000:.2f}s/1K steps\n")
        lines.append(f"- Ratio SGD/AdamW: {ratio:.2f}×\n\n")
        if ratio > 3:
            lines.append("- Confirmado: SGD es significativamente más lento. ")
            lines.append("Posibles causas: gradient clipping overhead (en cada step), ")
            lines.append("CSV writes overhead, o algún sync de GPU específico.\n\n")

    # Recommendation
    lines.append("## Recomendación para Phase 8 v2\n\n")
    lines.append("Basado en los diagnósticos arriba, la siguiente fase Phase 8 corregida debería:\n\n")
    rec = []
    if a1 and a1.get("t_grok"):
        rec.append("- H1 (depth=2): usar steps=300K en lugar de 200K")
    if a2 and a2.get("t_grok") and not (a1 and a1.get("t_grok")):
        rec.append("- H1 (depth=2): solo evaluar wd ≤ 0.3 (wd alto es estructuralmente incompatible)")
    if a3 and a3.get("t_grok"):
        rec.append("- H1 (depth=2): usar n_layers=4 en lugar de n_layers=2")
    if b3 and b3.get("t_grok"):
        if b1 and b1.get("t_grok"):
            rec.append("- H2 (SGD): usar lr=3e-3 (no 1e-2)")
        if b2 and b2.get("t_grok"):
            rec.append("- H2 (SGD): usar warmup_steps=2000 (no 100)")
    elif b3 and not b3.get("t_grok"):
        rec.append("- H2 (SGD): el path está roto. Debugger manual de train_v3.py SGD necesario antes de re-lanzar.")
    if not rec:
        rec.append("- Esperar a que terminen más diagnósticos para recomendar.")
    lines.extend(r + "\n" for r in rec)

    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    log(f"REPORT written: {REPORT_MD.absolute()}")
    COMPLETION_FLAG.write_text(
        f"Phase 8 DEBUG complete at {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()
    if not args.analyze:
        run_all()
    analyze()
    log("Phase 8 DEBUG done.")


if __name__ == "__main__":
    main()
