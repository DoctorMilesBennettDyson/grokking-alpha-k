"""
progress.py -- Show progress for Phase 5 and/or Phase 6.

Usage:
    python progress.py          # show both phases
    python progress.py --phase 5
    python progress.py --phase 6
"""

import argparse
import sys
from pathlib import Path

MIN_CSV_SIZE = 80_000
BASE = Path(__file__).parent  # always relative to this script, not cwd

PHASES = {
    5: {
        "dir": BASE / "results/phase5",
        "flag": BASE / "results/phase5/PHASE5_COMPLETE.flag",
        "progress": BASE / "results/phase5/PROGRESS.txt",
        "log": BASE / "results/phase5/phase5.log",
        "report": BASE / "results/phase5/REPORT.md",
        "expected": (
            [f"rebaseline_s{s}" for s in [10, 11, 12, 13, 14]]
            + [f"wd{w}_s{s}" for w in [0.01, 0.1, 0.3, 3.0] for s in [0, 1, 2]]
        ),
    },
    6: {
        "dir": BASE / "results/phase6",
        "flag": BASE / "results/phase6/PHASE6_COMPLETE.flag",
        "progress": BASE / "results/phase6/PROGRESS.txt",
        "log": BASE / "results/phase6/phase6.log",
        "report": BASE / "results/phase6/REPORT.md",
        "expected": (
            [f"p{p}_wd{w}_s{s}"
             for p in [31, 149]
             for w in [0.01, 0.1, 0.3, 1.0, 3.0]
             for s in [0, 1, 2]]
        ),
    },
    7: {
        "dir": BASE / "results/phase7",
        "flag": BASE / "results/phase7/PHASE7_COMPLETE.flag",
        "progress": BASE / "results/phase7/PROGRESS.txt",
        "log": BASE / "results/phase7/phase7.log",
        "report": BASE / "results/phase7/REPORT.md",
        "expected": (
            # Component A: cell closing
            [f"p31_wd0.01_s{s}" for s in [10, 11, 12, 13, 14]]
            + [f"p31_wd0.1_s{s}" for s in [10, 11, 12]]
            + [f"p97_wd3.0_s{s}" for s in [10, 11, 12, 13, 14]]
            + [f"p149_wd3.0_s{s}" for s in [10, 11, 12, 13, 14]]
            # Component B: intermediate primes
            + [f"p{p}_wd{w}_s{s}"
               for p in [53, 67, 113]
               for w in [0.01, 0.1, 0.3, 1.0, 3.0]
               for s in [0, 1, 2]]
        ),
    },
    8: {
        "dir": BASE / "results/phase8",
        "flag": BASE / "results/phase8/PHASE8_COMPLETE.flag",
        "progress": BASE / "results/phase8/PROGRESS.txt",
        "log": BASE / "results/phase8/phase8.log",
        "report": BASE / "results/phase8/REPORT.md",
        "expected": (
            # H1: depth=2 sweep
            [f"H1_depth2_wd{w}_s{s}"
             for w in [0.01, 0.1, 0.3, 1.0, 3.0] for s in [0, 1, 2]]
            # H2: SGD sweep
            + [f"H2_sgd_wd{w}_s{s}"
               for w in [0.01, 0.1, 0.3, 1.0, 3.0] for s in [0, 1, 2]]
            # H3: init comparison
            + [f"H3_init{i}_s{s}"
               for i in ["xavier", "kaiming"] for s in [20, 21, 22, 23, 24]]
            # H4: batch_size (only 256 and 1024; 512 = baseline reference)
            + [f"H4_bs{bs}_s{s}" for bs in [256, 1024] for s in [0, 1, 2]]
        ),
    },
}


def show_phase(phase_num):
    cfg = PHASES[phase_num]
    print(f"\n{'='*50}")
    print(f"PHASE {phase_num}")
    print(f"{'='*50}")

    if cfg["flag"].exists():
        print("STATUS: COMPLETE")
        print(cfg["flag"].read_text(encoding="utf-8").strip())
        if cfg["report"].exists():
            print(f"Report: {cfg['report'].absolute()}")
        return

    if not cfg["dir"].exists():
        print("STATUS: NOT STARTED")
        print(f"  Run: python phase{phase_num}.py")
        return

    expected = cfg["expected"]
    total = len(expected)
    done = sum(
        1 for n in expected
        if (cfg["dir"] / f"{n}.csv").exists()
        and (cfg["dir"] / f"{n}.csv").stat().st_size >= MIN_CSV_SIZE
    )
    pending = [n for n in expected
               if not (cfg["dir"] / f"{n}.csv").exists()
               or (cfg["dir"] / f"{n}.csv").stat().st_size < MIN_CSV_SIZE]

    pct = 100 * done / total if total else 0
    print(f"Runs: {done}/{total} ({pct:.0f}%)")

    if cfg["progress"].exists():
        print()
        for line in cfg["progress"].read_text(encoding="utf-8").strip().splitlines():
            print(f"  {line}")

    if pending:
        print(f"\nPending ({len(pending)}):")
        for n in pending[:6]:
            print(f"  - {n}")
        if len(pending) > 6:
            print(f"  ... and {len(pending)-6} more")

    if cfg["log"].exists():
        print("\nLast 4 log lines:")
        try:
            tail = cfg["log"].read_text(encoding="utf-8").splitlines()[-4:]
            for line in tail:
                print(f"  {line}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[5, 6, 7, 8], default=None)
    args = parser.parse_args()

    phases = [args.phase] if args.phase else [5, 6, 7, 8]
    for ph in phases:
        show_phase(ph)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
