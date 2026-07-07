"""
phase5.py -- Self-contained Phase 5 runner.

Goal: address the two weaknesses Opus 4.7 flagged in Phase 4 review:
  (A) Re-validate baseline (1 of 3 baseline runs in Phase 4 had final_test_acc=0.7,
      meaning the Phase 4 mean baseline of 1267 may be biased).
      Solution: 5 new seeds (s10-s14) on baseline config, raising N=8 total.
  (B) Sweep weight_decay -- the canonical regularizer in grokking literature
      (Power et al. 2022, Nanda et al. 2023). NOT varied in any prior phase.
      Solution: wd in {0.01, 0.1, 0.3, 3.0} x seeds {0,1,2} = 12 runs.

Total: 17 runs ~= 3 hours on RTX 3070.

Properties:
  - IDEMPOTENT: re-running skips completed runs (checked by csv file presence + size)
  - PROGRESS-VISIBLE: writes results/phase5/PROGRESS.txt after each run
  - SELF-CONTAINED: no external orchestrator needed; just `python phase5.py`
  - AUTO-REPORT: generates results/phase5/REPORT.md at the end
  - CRASH-SAFE: PROGRESS.txt and individual CSVs persist across restarts

Usage:
    python phase5.py            # run everything pending
    python phase5.py --analyze  # only run the analysis (assumes runs done)

To check progress without bothering Claude/Opus:
    python progress.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

from train import train_single_run

# --- Configuration ---------------------------------------------------------

OUTPUT_DIR = Path("results/phase5")
PROGRESS_FILE = OUTPUT_DIR / "PROGRESS.txt"
COMPLETION_FLAG = OUTPUT_DIR / "PHASE5_COMPLETE.flag"
SUMMARY_JSON = OUTPUT_DIR / "phase5_summary.json"
REPORT_MD = OUTPUT_DIR / "REPORT.md"
LOG_FILE = OUTPUT_DIR / "phase5.log"

# Minimum CSV size for "complete" detection (typical 100k-step run is ~100KB)
MIN_CSV_SIZE = 80_000

# Common kwargs (mirror Phase 4 baseline EXCEPT where varied)
COMMON_KWARGS = dict(
    p=97,
    depth=1,
    steps=100_000,
    dropout=0.0,
    label_smoothing=0.0,
    lr=1e-3,
    batch_size=512,
    train_frac=0.5,
    eval_every=100,
    grok_threshold=0.95,
    warmup_steps=100,
)

# (A) Re-baseline runs: 5 new seeds with weight_decay=1.0 (Phase 4 baseline config)
REBASELINE_SEEDS = [10, 11, 12, 13, 14]
REBASELINE_WD = 1.0

# (B) Weight-decay sweep: 4 wd values x 3 seeds
WD_SWEEP_VALUES = [0.01, 0.1, 0.3, 3.0]
WD_SWEEP_SEEDS = [0, 1, 2]


def build_run_list():
    """Build the full ordered list of (run_name, kwargs) for Phase 5."""
    runs = []
    # (A) re-baseline
    for seed in REBASELINE_SEEDS:
        kwargs = dict(COMMON_KWARGS, seed=seed, weight_decay=REBASELINE_WD,
                      run_name=f"rebaseline_s{seed}")
        runs.append(("rebaseline", kwargs))
    # (B) wd sweep
    for wd in WD_SWEEP_VALUES:
        for seed in WD_SWEEP_SEEDS:
            kwargs = dict(COMMON_KWARGS, seed=seed, weight_decay=wd,
                          run_name=f"wd{wd}_s{seed}")
            runs.append(("wd_sweep", kwargs))
    return runs


def is_run_complete(run_name):
    """Check if a run already has a complete CSV."""
    csv_path = OUTPUT_DIR / f"{run_name}.csv"
    return csv_path.exists() and csv_path.stat().st_size >= MIN_CSV_SIZE


def log(msg):
    """Append timestamped message to log + stdout."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_progress(done, total, current=None, eta_seconds=None):
    """Write a single human-readable progress line to PROGRESS.txt."""
    pct = 100.0 * done / total if total else 0.0
    parts = [
        f"Phase 5 progress: {done}/{total} runs ({pct:.0f}%)",
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if current:
        parts.append(f"running_now: {current}")
    if eta_seconds and eta_seconds > 0:
        eta_min = eta_seconds / 60
        eta_h = eta_min / 60
        if eta_h >= 1:
            parts.append(f"eta_remaining: {eta_h:.1f} hours")
        else:
            parts.append(f"eta_remaining: {eta_min:.0f} minutes")
    if done >= total:
        parts.append("STATUS: COMPLETE")
    else:
        parts.append("STATUS: IN_PROGRESS")
    PROGRESS_FILE.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run_all():
    """Run all pending Phase 5 experiments. Idempotent."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = build_run_list()
    total = len(runs)
    pending = [r for r in runs if not is_run_complete(r[1]["run_name"])]
    already_done = total - len(pending)

    log("=" * 60)
    log("PHASE 5 STARTING")
    log("=" * 60)
    log(f"Total planned runs: {total}")
    log(f"Already complete (skipped): {already_done}")
    log(f"To execute: {len(pending)}")
    log(f"Output dir: {OUTPUT_DIR.absolute()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    if device == "cpu":
        log("WARNING: no CUDA detected. Runs will be very slow.")

    # initial progress write
    write_progress(already_done, total)

    times = []  # for ETA estimation
    summaries = _load_existing_summaries()

    for i, (group, kwargs) in enumerate(runs):
        run_name = kwargs["run_name"]
        if is_run_complete(run_name):
            continue

        # Update progress -- we are about to start this one
        elapsed_avg = sum(times) / len(times) if times else None
        remaining_count = sum(1 for r in runs if not is_run_complete(r[1]["run_name"]))
        eta = elapsed_avg * remaining_count if elapsed_avg else None
        write_progress(total - remaining_count, total, current=run_name, eta_seconds=eta)

        log(f"--- Run {i+1}/{total}: {run_name} (group={group}) ---")
        log(f"    weight_decay={kwargs['weight_decay']}, seed={kwargs['seed']}")
        t0 = time.time()
        try:
            result = train_single_run(
                output_dir=str(OUTPUT_DIR),
                device=device,
                **kwargs,
            )
            result["group"] = group
            result["weight_decay"] = kwargs.get("weight_decay", 1.0)
            summaries.append(result)
            summaries = _save_summaries(summaries)
        except Exception as exc:
            log(f"ERROR in {run_name}: {type(exc).__name__}: {exc}")
            log("Continuing with next run.")
            continue

        elapsed = time.time() - t0
        times.append(elapsed)
        t_grok_str = str(result.get("t_grok"))
        log(f"    DONE in {elapsed:.0f}s. t_grok={t_grok_str}, "
            f"final_test_acc={result.get('final_test_acc'):.4f}")

    # final progress write
    write_progress(total, total)

    log("=" * 60)
    log("PHASE 5 RUNS COMPLETE")
    log("=" * 60)


def _load_existing_summaries():
    if SUMMARY_JSON.exists():
        try:
            return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def _save_summaries(summaries):
    # Deduplicate by run_name, keeping the last entry (most recent overwrite wins)
    seen = {}
    for r in summaries:
        seen[r.get("run_name")] = r
    deduped = list(seen.values())
    SUMMARY_JSON.write_text(
        json.dumps(deduped, indent=2, default=str),
        encoding="utf-8",
    )
    return deduped


# --- Analysis --------------------------------------------------------------

def _read_phase4_baseline_t_groks():
    """Pull baseline t_grok values from Phase 4 summary for combined statistics."""
    p4 = Path("results/phase4/phase4_summary.json")
    if not p4.exists():
        return []
    try:
        data = json.loads(p4.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    out = []
    for r in data:
        if r.get("dropout") == 0.0 and r.get("label_smoothing") == 0.0:
            out.append({
                "seed": r.get("seed"),
                "t_grok": r.get("t_grok"),
                "final_test_acc": r.get("final_test_acc"),
            })
    return out


def analyze():
    """Generate REPORT.md from Phase 5 + Phase 4 baseline data."""
    summaries = _load_existing_summaries()
    if not summaries:
        log("No summaries to analyze. Aborting analyze().")
        return

    p4_baseline = _read_phase4_baseline_t_groks()

    # Group new data
    rebaseline = [r for r in summaries if r.get("group") == "rebaseline"]
    wd_runs = [r for r in summaries if r.get("group") == "wd_sweep"]

    # Combined baseline (Phase 4 baseline + Phase 5 re-baseline)
    combined_baseline = []
    for r in p4_baseline:
        combined_baseline.append({
            "source": "phase4",
            "seed": r["seed"],
            "t_grok": r["t_grok"],
            "final_test_acc": r["final_test_acc"],
        })
    for r in rebaseline:
        combined_baseline.append({
            "source": "phase5",
            "seed": r["seed"],
            "t_grok": r["t_grok"],
            "final_test_acc": r["final_test_acc"],
        })

    # WD groups
    wd_groups = {}
    for r in wd_runs:
        # weight_decay may be missing from entries written before the fix.
        # Parse from run_name (format: "wd{value}_s{seed}") as fallback.
        wd = r.get("weight_decay")
        if wd is None:
            rname = r.get("run_name", "")
            if rname.startswith("wd"):
                try:
                    wd = float(rname.split("_")[0][2:])
                except (ValueError, IndexError):
                    wd = None
        if wd is not None:
            wd_groups.setdefault(wd, []).append(r)

    # --- Build report ---
    lines = []
    lines.append("# Phase 5 -- REPORT\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append("## What this phase tested\n\n")
    lines.append("Two specific weaknesses flagged in the Phase 4 review:\n\n")
    lines.append("- **(A) Re-baseline robustness**: 5 new seeds (10-14) added to ")
    lines.append("the original 3 Phase 4 baseline seeds, raising N=8.\n")
    lines.append("- **(B) Weight-decay sweep**: missing in Phase 4 despite being ")
    lines.append("the canonical grokking regularizer (Power et al. 2022).\n\n")

    # --- Baseline analysis ---
    lines.append("## (A) Baseline robustness check\n\n")
    valid_baseline = [r for r in combined_baseline
                      if r["t_grok"] is not None and r["final_test_acc"] >= 0.95]
    invalid_baseline = [r for r in combined_baseline
                        if r["final_test_acc"] is None or r["final_test_acc"] < 0.95]

    lines.append(f"- Total baseline runs (Phase 4 + Phase 5): {len(combined_baseline)}\n")
    lines.append(f"- Runs with final_test_acc >= 0.95 (valid groks): {len(valid_baseline)}\n")
    lines.append(f"- Runs flagged as not-grokked-cleanly (final_test_acc < 0.95): {len(invalid_baseline)}\n\n")

    if valid_baseline:
        t_groks = [r["t_grok"] for r in valid_baseline]
        mean = sum(t_groks) / len(t_groks)
        std = (sum((x - mean) ** 2 for x in t_groks) / max(len(t_groks) - 1, 1)) ** 0.5
        lines.append(f"**Baseline mean t_grok (clean grokks only): {mean:.0f} steps "
                     f"(std={std:.0f}, N={len(t_groks)})**\n\n")
        # Compare with the Phase 4 reported mean of 1267
        old_mean = 1267
        delta_pct = 100 * (mean - old_mean) / old_mean
        lines.append(f"Phase 4 reported baseline mean was 1267 (N=3, including 1 ")
        lines.append(f"flagged outlier). Updated mean is {mean:.0f}, ")
        lines.append(f"a change of {delta_pct:+.1f}%.\n\n")

    lines.append("### Per-run baseline table\n\n")
    lines.append("| source | seed | t_grok | final_test_acc | clean? |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in sorted(combined_baseline, key=lambda x: (x["source"], x["seed"])):
        clean = "yes" if (r["final_test_acc"] is not None and r["final_test_acc"] >= 0.95) else "NO"
        ta = f"{r['final_test_acc']:.4f}" if r["final_test_acc"] is not None else "N/A"
        lines.append(f"| {r['source']} | {r['seed']} | {r['t_grok']} | {ta} | {clean} |\n")
    lines.append("\n")

    # --- WD sweep analysis ---
    lines.append("## (B) Weight-decay sweep\n\n")
    if wd_groups:
        lines.append("| weight_decay | N | mean t_grok | std | min | max | min_acc |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        # Include the combined baseline as wd=1.0 row for reference
        if valid_baseline:
            tg = [r["t_grok"] for r in valid_baseline]
            min_acc = min(r["final_test_acc"] for r in valid_baseline)
            mean = sum(tg) / len(tg)
            std = (sum((x - mean) ** 2 for x in tg) / max(len(tg) - 1, 1)) ** 0.5
            lines.append(f"| 1.0 (baseline) | {len(tg)} | {mean:.0f} | {std:.0f} | "
                         f"{min(tg)} | {max(tg)} | {min_acc:.3f} |\n")
        for wd in sorted(wd_groups.keys()):
            runs = wd_groups[wd]
            valid = [r for r in runs if r["t_grok"] is not None]
            if not valid:
                lines.append(f"| {wd} | 0 | NO_GROKKING | - | - | - | - |\n")
                continue
            tg = [r["t_grok"] for r in valid]
            mean = sum(tg) / len(tg)
            std = (sum((x - mean) ** 2 for x in tg) / max(len(tg) - 1, 1)) ** 0.5
            min_acc = min(r["final_test_acc"] for r in valid)
            lines.append(f"| {wd} | {len(valid)} | {mean:.0f} | {std:.0f} | "
                         f"{min(tg)} | {max(tg)} | {min_acc:.3f} |\n")
        lines.append("\n")

    # --- Diagnostic interpretation ---
    lines.append("## Diagnostic interpretation (auto-generated)\n\n")
    if wd_groups:
        wd_means = {}
        for wd, runs in wd_groups.items():
            valid = [r["t_grok"] for r in runs if r["t_grok"] is not None]
            if valid:
                wd_means[wd] = sum(valid) / len(valid)
        if valid_baseline:
            wd_means[1.0] = sum(r["t_grok"] for r in valid_baseline) / len(valid_baseline)

        if wd_means:
            ordered = sorted(wd_means.items())
            lines.append("- weight_decay -> mean t_grok:\n")
            for wd, m in ordered:
                lines.append(f"    * wd={wd}: {m:.0f}\n")
            lines.append("\n")

            # Detect monotonicity
            xs = [w for w, _ in ordered]
            ys = [m for _, m in ordered]
            if all(ys[i] >= ys[i-1] for i in range(1, len(ys))):
                shape = "monotonically non-decreasing"
            elif all(ys[i] <= ys[i-1] for i in range(1, len(ys))):
                shape = "monotonically non-increasing"
            else:
                # Find min
                min_idx = ys.index(min(ys))
                shape = f"U-shaped (min at wd={xs[min_idx]})"
            lines.append(f"- Shape of weight_decay -> t_grok curve: **{shape}**\n\n")

    lines.append("## Notes for Paper 04 v0.2 update\n\n")
    lines.append("1. Replace the Phase 4 baseline mean (1267, N=3) with the combined ")
    lines.append("baseline above. Drop or flag the s1 outlier explicitly.\n")
    lines.append("2. Use the weight_decay sweep to position the framework's claim about ")
    lines.append("noise (mu) vs. structural pressure (sigma). If wd shows U-shape, the ")
    lines.append("framework's argument that 'too much regularization slows things' is ")
    lines.append("supported. If monotonic, it needs revision.\n")
    lines.append("3. Compare wd-curve direction with dropout-curve direction (Phase 4). ")
    lines.append("Two regularizers acting through different mechanisms with different ")
    lines.append("curve shapes is publishable evidence on its own.\n")

    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    log(f"REPORT written: {REPORT_MD.absolute()}")

    # Write completion flag
    COMPLETION_FLAG.write_text(
        f"Phase 5 complete at {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    log(f"Completion flag: {COMPLETION_FLAG.absolute()}")


# --- Entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5 runner")
    parser.add_argument("--analyze", action="store_true",
                        help="Skip running; only analyze existing results.")
    args = parser.parse_args()

    if not args.analyze:
        run_all()
    analyze()
    log("Phase 5 fully done. Open REPORT.md.")


if __name__ == "__main__":
    main()
