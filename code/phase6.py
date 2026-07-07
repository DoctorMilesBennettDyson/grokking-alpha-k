"""
phase6.py -- Replication of weight_decay sweep at p=31 and p=149.

Addresses the final reviewer concern: does the power law
    t_grok ∝ wd^(-0.73)
hold across different moduli (task complexities), or is it p=97-specific?

Also introduces sustained threshold: t_grok_sustained = first step where
test_acc ≥ 0.95 for 3 consecutive evaluations (300 steps), preventing
false-positive threshold crossings (the outlier pattern seen in P4+P5).

Configuration:
  p         ∈ {31, 149}
  wd        ∈ {0.01, 0.1, 0.3, 1.0, 3.0}
  seeds     ∈ {0, 1, 2}
  steps     = 200_000  (safe budget: wd=0.01 at p=31 estimated ~62K)
  Total     = 30 runs, ~10 hours on RTX 3070

Usage:
    python phase6.py              # run everything pending, then analyze
    python phase6.py --analyze    # analyze only (assumes runs complete)

Monitor progress (no Claude needed):
    python progress.py --phase 6
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

# Ensure imports resolve regardless of launch directory
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train import train_single_run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = _HERE / "results/phase6"
PROGRESS_FILE = OUTPUT_DIR / "PROGRESS.txt"
COMPLETION_FLAG = OUTPUT_DIR / "PHASE6_COMPLETE.flag"
SUMMARY_JSON = OUTPUT_DIR / "phase6_summary.json"
REPORT_MD = OUTPUT_DIR / "REPORT.md"
LOG_FILE = OUTPUT_DIR / "phase6.log"

MIN_CSV_SIZE = 80_000  # bytes; 200K-step run ≈ 200KB

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

PRIMES = [31, 149]
WD_VALUES = [0.01, 0.1, 0.3, 1.0, 3.0]
SEEDS = [0, 1, 2]
STEPS = 200_000

COMMON_KWARGS = dict(
    depth=1,
    steps=STEPS,
    dropout=0.0,
    label_smoothing=0.0,
    lr=1e-3,
    batch_size=512,
    train_frac=0.5,
    eval_every=100,
    grok_threshold=0.95,
    warmup_steps=100,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_run_list():
    runs = []
    for p in PRIMES:
        for wd in WD_VALUES:
            for seed in SEEDS:
                name = f"p{p}_wd{wd}_s{seed}"
                kwargs = dict(COMMON_KWARGS, p=p, weight_decay=wd, seed=seed,
                              run_name=name)
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
        f"Phase 6 progress: {done}/{total} ({pct:.0f}%)",
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
# Sustained threshold computation (from CSV)
# ---------------------------------------------------------------------------

def t_grok_sustained(csv_path, threshold=0.95, n_consecutive=3):
    """
    First step where test_acc >= threshold for n_consecutive evaluations in a row.
    eval_every=100, so n_consecutive=3 means sustained for 300 steps.
    Returns None if never sustained.
    """
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
                return window[0]  # first step of the sustained run
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
    pending = [r for r in runs if not is_done(r["run_name"])]
    already_done = total - len(pending)

    log("=" * 60)
    log("PHASE 6 START")
    log(f"  primes: {PRIMES}")
    log(f"  wd values: {WD_VALUES}")
    log(f"  seeds: {SEEDS}")
    log(f"  steps per run: {STEPS:,}")
    log(f"  total runs: {total}  |  already done: {already_done}  |  pending: {len(pending)}")
    log(f"  estimated time: {len(pending) * 1200 / 3600:.1f}h")
    log("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    summaries = load_summaries()
    wall_times = []
    write_progress(already_done, total)

    for idx, kwargs in enumerate(runs):
        name = kwargs["run_name"]
        if is_done(name):
            continue

        done_so_far = sum(1 for r in runs[:idx+1] if is_done(r["run_name"]))
        remaining = total - done_so_far
        avg_t = sum(wall_times) / len(wall_times) if wall_times else None
        eta = avg_t * remaining if avg_t else None
        write_progress(done_so_far, total, current=name, eta_s=eta)

        log(f"--- [{done_so_far+1}/{total}] {name}  p={kwargs['p']}  wd={kwargs['weight_decay']}  seed={kwargs['seed']} ---")
        t0 = time.time()
        try:
            result = train_single_run(
                output_dir=str(OUTPUT_DIR),
                device=device,
                **kwargs,
            )
            result["weight_decay"] = kwargs["weight_decay"]
            result["group"] = f"p{kwargs['p']}"
            # Compute sustained t_grok from CSV
            csv_path = OUTPUT_DIR / f"{name}.csv"
            result["t_grok_sustained"] = t_grok_sustained(str(csv_path))
            summaries.append(result)
            summaries = save_summaries(summaries)
        except Exception as exc:
            log(f"ERROR {name}: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.time() - t0
        wall_times.append(elapsed)
        tg = result.get("t_grok")
        tgs = result.get("t_grok_sustained")
        log(f"    done {elapsed:.0f}s | t_grok_first={tg} | t_grok_sustained={tgs} | acc={result.get('final_test_acc'):.4f}")

    write_progress(total, total)
    log("=" * 60)
    log("PHASE 6 RUNS COMPLETE")
    log("=" * 60)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _load_p97_reference():
    """Pull p=97 wd sweep from Phase 5 + Phase 4 for the combined table."""
    ref = {}  # wd -> list of t_grok_sustained (or t_grok_first fallback)

    # Phase 4: baseline (wd=1.0) clean runs and wd sweep not available -- only wd=1.0 exists
    # Phase 5: all wd sweep data
    p5 = _HERE / "results/phase5/phase5_summary.json"
    p4_baseline_tg = {0: 1200, 2: 1400}  # from verified data (s1 excluded, acc=0.705)
    p4_baseline_tgs = {}  # we'd need to compute from CSVs; skip for now

    if p5.exists():
        try:
            data = json.loads(p5.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
        for r in data:
            if r.get("group") != "wd_sweep":
                continue
            rname = r.get("run_name", "")
            wd = r.get("weight_decay")
            if wd is None and rname.startswith("wd"):
                try:
                    wd = float(rname.split("_")[0][2:])
                except (ValueError, IndexError):
                    continue
            if wd is None:
                continue
            # Compute sustained from CSV if not stored
            tgs = r.get("t_grok_sustained")
            if tgs is None:
                csv_path = _HERE / "results/phase5" / f"{rname}.csv"
                tgs = t_grok_sustained(str(csv_path))
            ref.setdefault(wd, []).append({
                "t_grok_first": r.get("t_grok"),
                "t_grok_sustained": tgs,
                "final_test_acc": r.get("final_test_acc"),
                "seed": r.get("seed"),
            })

    # Inject wd=1.0 from Phase 4+5 combined baseline
    # Phase 5 rebaseline (wd=1.0, group=rebaseline)
    rebaseline_tg = []
    if p5.exists():
        try:
            data = json.loads(p5.read_text(encoding="utf-8"))
            for r in data:
                if r.get("group") == "rebaseline" and r.get("final_test_acc", 0) >= 0.95:
                    rname = r.get("run_name", "")
                    tgs = r.get("t_grok_sustained")
                    if tgs is None:
                        csv_path = _HERE / "results/phase5" / f"{rname}.csv"
                        tgs = t_grok_sustained(str(csv_path))
                    rebaseline_tg.append({
                        "t_grok_first": r.get("t_grok"),
                        "t_grok_sustained": tgs,
                        "final_test_acc": r.get("final_test_acc"),
                    })
        except Exception:
            pass
    # Phase 4 baseline clean seeds
    for seed, tg in p4_baseline_tg.items():
        csv_path = _HERE / "results/phase4" / f"dropout0.0_ls0.0_s{seed}.csv"
        tgs = t_grok_sustained(str(csv_path))
        rebaseline_tg.append({"t_grok_first": tg, "t_grok_sustained": tgs,
                               "final_test_acc": 1.0})

    if rebaseline_tg:
        ref[1.0] = rebaseline_tg

    return ref


def _group_by_wd(summaries, p):
    groups = {}
    for r in summaries:
        if r.get("p") != p:
            continue
        wd = r.get("weight_decay")
        if wd is None:
            rname = r.get("run_name", "")
            if "_wd" in rname:
                try:
                    wd = float(rname.split("_wd")[1].split("_")[0])
                except (ValueError, IndexError):
                    continue
        if wd is not None:
            groups.setdefault(wd, []).append(r)
    return groups


def _mean_std(vals):
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0
    s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5
    return m, s


def _wd_table_row(wd, runs, use_sustained=True):
    key = "t_grok_sustained" if use_sustained else "t_grok"
    valid = [r for r in runs
             if r.get(key) is not None and r.get("final_test_acc", 0) >= 0.95]
    dnf = len(runs) - len(valid)
    if not valid:
        return wd, None, None, None, None, f"DNF({len(runs)})"
    tg = [r[key] for r in valid]
    m, s = _mean_std(tg)
    return wd, m, s, min(tg), max(tg), f"N={len(valid)}" + (f" DNF={dnf}" if dnf else "")


def _power_law_slope(wd_means):
    """log-log slope. Returns slope, or None if < 2 points."""
    import math
    pts = [(math.log10(wd), math.log10(m)) for wd, m in wd_means if m and m > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x - mx)*(y - my) for x, y in zip(xs, ys))
    den = sum((x - mx)**2 for x in xs)
    return num / den if den else None


def analyze():
    summaries = load_summaries()
    if not summaries:
        log("No Phase 6 summaries. Nothing to analyze.")
        return

    p97_ref = _load_p97_reference()

    lines = []
    lines.append("# Phase 6 — REPORT\n\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append("## Objective\n\n")
    lines.append("Replicate the weight_decay power law `t_grok ∝ wd^(-0.73)` ")
    lines.append("(from Phase 4+5 at p=97) at two additional moduli:\n")
    lines.append("- **p=31** (small task, fewer training examples: 480 vs 4704)\n")
    lines.append("- **p=149** (larger task, more examples: 11100)\n\n")
    lines.append("Also introduces **sustained threshold**: t_grok_sustained = first ")
    lines.append("step where test_acc ≥ 0.95 for 3 consecutive evaluations (300 steps), ")
    lines.append("eliminating false-positive threshold crossings.\n\n")

    # === Per-p tables ===
    lines.append("## Weight-decay sweep results\n\n")
    lines.append("All using sustained threshold (t_grok_sustained). ")
    lines.append("'first' columns show original threshold for comparison.\n\n")

    for p in PRIMES:
        groups = _group_by_wd(summaries, p)
        lines.append(f"### p = {p}\n\n")
        lines.append("| wd | mean (sustained) | std | min | max | mean (first) | note |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        wd_means_sust = []
        wd_means_first = []
        for wd in WD_VALUES:
            runs = groups.get(wd, [])
            if not runs:
                lines.append(f"| {wd} | — | — | — | — | — | NOT RUN |\n")
                continue
            _, ms, ss, mn_s, mx_s, note_s = _wd_table_row(wd, runs, use_sustained=True)
            _, mf, _, _, _, _ = _wd_table_row(wd, runs, use_sustained=False)
            ms_str = f"{ms:.0f}" if ms else "DNF"
            ss_str = f"{ss:.0f}" if ss else "—"
            mn_str = str(mn_s) if mn_s else "—"
            mx_str = str(mx_s) if mx_s else "—"
            mf_str = f"{mf:.0f}" if mf else "DNF"
            lines.append(f"| {wd} | {ms_str} | {ss_str} | {mn_str} | {mx_str} | {mf_str} | {note_s} |\n")
            if ms:
                wd_means_sust.append((wd, ms))
            if mf:
                wd_means_first.append((wd, mf))

        slope_s = _power_law_slope(wd_means_sust)
        slope_f = _power_law_slope(wd_means_first)
        if slope_s:
            lines.append(f"\n**Power law slope (sustained): t_grok ∝ wd^({slope_s:.2f})**\n")
        if slope_f:
            lines.append(f"Power law slope (first-hit): t_grok ∝ wd^({slope_f:.2f})\n")
        lines.append("\n")

    # === p=97 reference from Phase 4+5 ===
    lines.append("### p = 97 (reference from Phase 4 + Phase 5)\n\n")
    lines.append("| wd | mean (sustained) | std | note |\n")
    lines.append("|---|---|---|---|\n")
    wd_means_97 = []
    for wd in WD_VALUES:
        runs97 = p97_ref.get(wd, [])
        if not runs97:
            lines.append(f"| {wd} | — | — | no data |\n")
            continue
        valid = [r for r in runs97
                 if (r.get("t_grok_sustained") or r.get("t_grok_first")) is not None
                 and r.get("final_test_acc", 0) >= 0.95]
        if not valid:
            lines.append(f"| {wd} | DNF | — | all outliers |\n")
            continue
        vals = [r.get("t_grok_sustained") or r.get("t_grok_first") for r in valid]
        m, s = _mean_std(vals)
        lines.append(f"| {wd} | {m:.0f} | {s:.0f} | N={len(valid)} |\n")
        wd_means_97.append((wd, m))
    slope_97 = _power_law_slope(wd_means_97)
    if slope_97:
        lines.append(f"\n**Power law slope p=97: t_grok ∝ wd^({slope_97:.2f})**\n\n")

    # === Cross-p comparison ===
    lines.append("## Cross-p comparison: does the power law hold?\n\n")
    lines.append("| p | wd=0.01 | wd=0.1 | wd=0.3 | wd=1.0 | wd=3.0 | slope |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    for p, label in [(31, "p=31"), (97, "p=97 (ref)"), (149, "p=149")]:
        if p == 97:
            groups = p97_ref
        else:
            groups = _group_by_wd(summaries, p)

        row_vals = []
        wd_means = []
        for wd in WD_VALUES:
            runs_here = groups.get(wd, [])
            if not runs_here:
                row_vals.append("—")
                continue
            if p == 97:
                valid = [r for r in runs_here
                         if (r.get("t_grok_sustained") or r.get("t_grok_first")) is not None
                         and r.get("final_test_acc", 0) >= 0.95]
                vals = [r.get("t_grok_sustained") or r.get("t_grok_first") for r in valid]
            else:
                _, ms, _, _, _, _ = _wd_table_row(wd, runs_here, use_sustained=True)
                if ms is None:
                    row_vals.append("DNF")
                    continue
                vals = [ms]
            if not vals:
                row_vals.append("DNF")
                continue
            m, _ = _mean_std(vals)
            row_vals.append(f"{m:.0f}")
            wd_means.append((wd, m))

        slope = _power_law_slope(wd_means)
        slope_str = f"{slope:.2f}" if slope else "—"
        lines.append(f"| {label} | " + " | ".join(row_vals) + f" | {slope_str} |\n")

    lines.append("\n")
    lines.append("If slopes are consistent across p (all around -0.7±0.15), ")
    lines.append("the power law is universal and not p-specific.\n\n")

    # === Sustained vs first-hit comparison ===
    lines.append("## Sustained vs first-hit threshold comparison\n\n")
    lines.append("Shows how many runs had false-positive t_grok under the original criterion.\n\n")
    lines.append("| run | t_grok_first | t_grok_sustained | delta | outlier? |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in sorted(summaries, key=lambda x: x.get("run_name", "")):
        tf = r.get("t_grok")
        ts = r.get("t_grok_sustained")
        acc = r.get("final_test_acc", 1.0)
        if tf is None:
            lines.append(f"| {r.get('run_name')} | DNF | — | — | yes |\n")
            continue
        if ts is None:
            lines.append(f"| {r.get('run_name')} | {tf} | DNF | — | yes |\n")
            continue
        delta = ts - tf
        outlier = "YES" if (acc < 0.95 or ts > tf + 500) else "no"
        lines.append(f"| {r.get('run_name')} | {tf} | {ts} | +{delta} | {outlier} |\n")
    lines.append("\n")

    # === Interpretation ===
    lines.append("## Interpretation for Paper 04 v0.2\n\n")
    lines.append("### On universality of the power law\n\n")
    lines.append("If the cross-p comparison shows similar slopes (~-0.7) across p=31, 97, 149:\n\n")
    lines.append("- The power law `t_grok ∝ wd^(-0.7)` is a property of the **learning ")
    lines.append("dynamics**, not of the specific task structure.\n")
    lines.append("- This supports the framework claim that weight_decay modulates L ")
    lines.append("(circuit complexity) in the Eigen inequality, independent of K(X) (task complexity).\n")
    lines.append("- Combined with Phase 4 dropout and label smoothing results: the three ")
    lines.append("regularization axes (wd→L, dropout→μ, ls→σ) are orthogonal and universal.\n\n")
    lines.append("### On sustained threshold\n\n")
    lines.append("If t_grok_sustained ≈ t_grok_first for most runs: the original threshold ")
    lines.append("is reliable and outliers were genuinely rare edge cases.\n")
    lines.append("If t_grok_sustained >> t_grok_first for many runs: the original results ")
    lines.append("were noisier than reported and the sustained threshold is the correct measure.\n\n")
    lines.append("Either conclusion is informative and publishable.\n\n")
    lines.append("### Recommended Paper 04 v0.2 additions\n\n")
    lines.append("1. **Main table** (Table 1): wd sweep at p=31, 97, 149 side by side.\n")
    lines.append("2. **Figure**: log-log plot of t_grok vs wd for the 3 primes, ")
    lines.append("all on one axes. Overlapping power law lines = universality claim.\n")
    lines.append("3. **Sustained threshold**: report both metrics, argue sustained is ")
    lines.append("the robust measure and is consistent with first-hit within noise.\n")
    lines.append("4. **Combined framework table**: wd (→L), dropout (→μ), ls (→σ) in ")
    lines.append("a 3-row table showing direction, magnitude, and mechanism.\n\n")

    lines.append("---\n\n")
    lines.append("*Phase 6 complete. Ready for Paper 04 v0.2 write-up.*\n")

    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    log(f"REPORT written: {REPORT_MD.absolute()}")
    COMPLETION_FLAG.write_text(
        f"Phase 6 complete at {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    write_progress(len(build_run_list()), len(build_run_list()))


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
    log("Phase 6 done.")


if __name__ == "__main__":
    main()
