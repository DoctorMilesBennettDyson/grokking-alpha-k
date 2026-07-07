"""
phase9.py -- Experiments P1 + P4 per prereg/PRE_REGISTRATION_P1_P4.md.

MUST NOT be launched before the pre-registration file is committed and pushed
publicly (see that document's header). Analysis evaluates exactly the
pre-registered criteria; no post-hoc adjustment.

  P1 (9 runs):  p=97 d=1 bs=512 wd=1.0, lr in {5e-4, 1e-3, 2e-3} x seeds {0,1,2}
                acceptance: mean t_sustained monotonically DECREASING in lr.
  P4 (24 runs): p=97 d=1 lr=1e-3, bs in {256,1024} x wd in {0.01,0.1,0.3,3.0}
                x seeds {0,1,2}; wd=1.0 cells REUSED from phase8 H4 (declared).
                acceptance: |alpha(256) - alpha(1024)| <= 0.15.

Usage:
    python phase9.py             # run + analyze
    python phase9.py --analyze   # analysis only
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_v3 import train_single_run_v3

OUTPUT_DIR = _HERE / "results/phase9"
PHASE8_DIR = _HERE / "results/phase8"
PROGRESS_FILE = OUTPUT_DIR / "PROGRESS.txt"
COMPLETION_FLAG = OUTPUT_DIR / "PHASE9_COMPLETE.flag"
SUMMARY_JSON = OUTPUT_DIR / "phase9_summary.json"
REPORT_MD = OUTPUT_DIR / "REPORT.md"
LOG_FILE = OUTPUT_DIR / "phase9.log"
MIN_CSV_SIZE = 80_000

COMMON = dict(p=97, depth=1, dropout=0.0, label_smoothing=0.0, train_frac=0.5,
              eval_every=100, grok_threshold=0.95, warmup_steps=100,
              steps=200_000, optimizer='adamw', init_scheme='xavier', n_layers=2)

SEEDS = [0, 1, 2]
P1_LRS = [5e-4, 1e-3, 2e-3]
P4_BSS = [256, 1024]
P4_WDS = [0.01, 0.1, 0.3, 3.0]   # wd=1.0 reused from phase8 H4 (pre-registered reuse)


def build_runs():
    runs = []
    for lr in P1_LRS:                      # P1 first: cheap, early signal
        for s in SEEDS:
            runs.append(dict(COMMON, weight_decay=1.0, batch_size=512, lr=lr,
                             seed=s, run_name=f"P1_lr{lr}_s{s}", __h='P1'))
    for bs in P4_BSS:                      # P4: bs=256 block, then bs=1024
        for wd in P4_WDS:
            for s in SEEDS:
                runs.append(dict(COMMON, weight_decay=wd, batch_size=bs, lr=1e-3,
                                 seed=s, run_name=f"P4_bs{bs}_wd{wd}_s{s}", __h='P4'))
    return runs


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
    parts = [f"Phase 9 (P1+P4) progress: {done}/{total} ({100.0*done/total:.0f}%)",
             f"updated: {datetime.now().isoformat(timespec='seconds')}"]
    if current:
        parts.append(f"running: {current}")
    if eta_s and eta_s > 0:
        parts.append(f"eta: {eta_s/3600:.1f}h" if eta_s >= 3600 else f"eta: {eta_s/60:.0f}m")
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
    seen = {r.get("run_name"): r for r in summaries}
    SUMMARY_JSON.write_text(json.dumps(list(seen.values()), indent=2, default=str),
                            encoding="utf-8")
    return list(seen.values())


def t_grok_sustained(csv_path, threshold=0.95, n_consecutive=3):
    try:
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
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
    runs = build_runs()
    total = len(runs)
    pending = sum(1 for r in runs if not is_done(r["run_name"]))
    log("=" * 60)
    log(f"PHASE 9 (P1+P4) START — total {total}, done {total-pending}, pending {pending}")
    log("Pre-registration: prereg/PRE_REGISTRATION_P1_P4.md (must be pushed publicly first)")
    log("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    summaries = load_summaries()
    wall = []
    for idx, kw in enumerate(runs):
        name = kw["run_name"]
        if is_done(name):
            continue
        done_now = sum(1 for r in runs[:idx+1] if is_done(r["run_name"]))
        eta = (sum(wall)/len(wall)) * (total - done_now) if wall else None
        write_progress(done_now, total, current=name, eta_s=eta)
        h = kw.pop('__h')
        log(f"--- [{done_now+1}/{total}] {name} ({h}) ---")
        t0 = time.time()
        try:
            res = train_single_run_v3(output_dir=str(OUTPUT_DIR), device=device, **kw)
            res["__h"] = h
            res["t_grok_sustained"] = t_grok_sustained(str(OUTPUT_DIR / f"{name}.csv"))
            summaries.append(res)
            summaries = save_summaries(summaries)
        except Exception as exc:
            log(f"ERROR {name}: {type(exc).__name__}: {exc}")
            continue
        wall.append(time.time() - t0)
        log(f"    done {wall[-1]:.0f}s | t_grok={res.get('t_grok')} | "
            f"sustained={res.get('t_grok_sustained')} | acc={res.get('final_test_acc'):.4f}")
    write_progress(total, total)
    log("PHASE 9 RUNS COMPLETE")


# --------------------------------------------------------------------------
# Analysis — exactly the pre-registered criteria
# --------------------------------------------------------------------------

def _is_clean(r):
    tg, acc = r.get("t_grok_sustained"), r.get("final_test_acc")
    if tg is None or acc is None or acc < 0.95:
        return False
    t_first = r.get("t_grok")
    return not (t_first is not None and tg > t_first + 1000)


def _mean_std(v):
    if not v:
        return None, None
    m = sum(v) / len(v)
    s = (sum((x-m)**2 for x in v)/max(len(v)-1, 1))**0.5 if len(v) > 1 else 0.0
    return m, s


def _slope(pairs):  # pairs: [(wd, mean_t), ...]
    pts = [(math.log10(w), math.log10(t)) for w, t in pairs if t and t > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts)/n
    my = sum(p[1] for p in pts)/n
    num = sum((x-mx)*(y-my) for x, y in pts)
    den = sum((x-mx)**2 for x, _ in pts)
    return num/den if den else None


def _phase8_h4_cell(bs):
    """Reuse wd=1.0 cells from phase8 H4 (pre-registered reuse)."""
    f = PHASE8_DIR / "phase8_summary.json"
    out = []
    if f.exists():
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("__h") == "H4" and r.get("batch_size") == bs and _is_clean(r):
                out.append(r["t_grok_sustained"])
    return out


def analyze():
    summaries = load_summaries()
    by_h = {}
    for r in summaries:
        by_h.setdefault(r.get("__h", "?"), []).append(r)
    L = []
    L.append("# Phase 9 (P1 + P4) — REPORT\n\n")
    L.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    L.append("Criteria are exactly those of prereg/PRE_REGISTRATION_P1_P4.md. No post-hoc changes.\n\n")

    # ---- P1 ----
    L.append("## P1 — t_grok vs learning rate (bs=512, wd=1.0)\n\n")
    L.append("| lr | mean t_sustained | std | N_clean/N |\n|---|---|---|---|\n")
    means = {}
    for lr in P1_LRS:
        cell = [r for r in by_h.get("P1", []) if abs(r["lr"] - lr) < 1e-12]
        clean = [r["t_grok_sustained"] for r in cell if _is_clean(r)]
        m, s = _mean_std(clean)
        means[lr] = m
        L.append(f"| {lr} | {f'{m:.0f}' if m else '—'} | {f'{s:.0f}' if s is not None and m else '—'} | {len(clean)}/{len(cell)} |\n")
    p1_ok = None
    if all(means.get(lr) for lr in P1_LRS):
        p1_ok = means[5e-4] > means[1e-3] > means[2e-3]
        r_hi = means[2e-3]/means[1e-3]
        r_lo = means[5e-4]/means[1e-3]
        L.append(f"\n**Ordering observed:** {means[5e-4]:.0f} > {means[1e-3]:.0f} > {means[2e-3]:.0f} "
                 f"→ {'holds' if p1_ok else 'VIOLATED'}\n")
        L.append(f"**Verdict: P1 {'CONFIRMED' if p1_ok else 'REJECTED'}** "
                 f"(binding criterion: monotone decrease)\n")
        L.append(f"- Exploratory bands: t(2e-3)/t(1e-3) = {r_hi:.2f} (band [0.65, 0.90]); "
                 f"t(5e-4)/t(1e-3) = {r_lo:.2f} (band [1.10, 1.55]) — non-binding.\n\n")
    else:
        L.append("\n**Insufficient clean data — check DNF rule in prereg (inconclusive, not rescued).**\n\n")

    # ---- P4 ----
    L.append("## P4 — wd slope at bs=256 vs bs=1024 (lr=1e-3)\n\n")
    alphas = {}
    for bs in P4_BSS:
        L.append(f"### bs={bs}\n\n| wd | mean t_sustained | N_clean/N | source |\n|---|---|---|---|\n")
        pairs = []
        for wd in sorted(P4_WDS + [1.0]):
            if wd == 1.0:
                clean = _phase8_h4_cell(bs)
                src = "phase8 H4 (declared reuse)"
                ncell = 3
            else:
                cell = [r for r in by_h.get("P4", [])
                        if r["batch_size"] == bs and abs(r["weight_decay"] - wd) < 1e-12]
                clean = [r["t_grok_sustained"] for r in cell if _is_clean(r)]
                src = "phase9"
                ncell = len(cell)
            m, _ = _mean_std(clean)
            if m:
                pairs.append((wd, m))
            L.append(f"| {wd} | {f'{m:.0f}' if m else '—'} | {len(clean)}/{ncell} | {src} |\n")
        alphas[bs] = _slope(pairs)
        L.append(f"\nslope α(bs={bs}) = {alphas[bs]:.3f}\n\n" if alphas[bs] is not None
                 else "\nslope not estimable\n\n")
    p4_ok = None
    if all(alphas.get(bs) is not None for bs in P4_BSS):
        d = abs(alphas[256] - alphas[1024])
        p4_ok = d <= 0.15
        L.append(f"**|α(256) − α(1024)| = {d:.3f}** (binding tolerance 0.15; "
                 f"reference α(512) = -0.731)\n")
        L.append(f"**Verdict: P4 {'CONFIRMED — separability holds' if p4_ok else 'REJECTED — wd × bs interaction'}**\n\n")

    # ---- decision table ----
    L.append("## Pre-registered decision table\n\n")
    if p1_ok is not None and p4_ok is not None:
        cell = {(True, True): "Temperature reading survives first two tests → eq. (10) 'supported working model'",
                (True, False): "T affects level via lr, but separability fails → interaction model needed",
                (False, True): "bs effect is not temperature (likely epoch coverage) → drop σ/T term",
                (False, False): "v0.2 refinement refuted → paper reports empirical maps only"}[(p1_ok, p4_ok)]
        L.append(f"P1 {'pass' if p1_ok else 'fail'} + P4 {'pass' if p4_ok else 'fail'} → {cell}\n")
    else:
        L.append("Incomplete data; no decision issued.\n")

    REPORT_MD.write_text("".join(L), encoding="utf-8")
    log(f"REPORT written: {REPORT_MD}")
    COMPLETION_FLAG.write_text(
        f"Phase 9 complete at {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if not args.analyze:
        run_all()
    analyze()
    log("Phase 9 done.")


if __name__ == "__main__":
    main()
