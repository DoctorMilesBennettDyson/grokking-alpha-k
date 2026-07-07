"""
orchestrate.py -- Autonomous experiment orchestrator.

Monitors Phase 2 completion, then chains Phase 3, Phase 4, and final analysis.
Designed to run unattended in the background.

Usage:
    python orchestrate.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime


RESULTS_DIR = Path("results")
PHASE2_DIR = RESULTS_DIR / "phase2"
PHASE3_DIR = RESULTS_DIR / "phase3"
PHASE4_DIR = RESULTS_DIR / "phase4"

# Expected Phase 2 files (6 primes x 3 seeds = 18)
PHASE2_PRIMES = [31, 53, 67, 97, 113, 149]
SEEDS = [0, 1, 2]
PHASE2_EXPECTED = [f"p{p}_s{s}.csv" for p in PHASE2_PRIMES for s in SEEDS]

# Phase 3 expected files (3 depths x 3 seeds = 9)
PHASE3_DEPTHS = [1, 2, 3]
PHASE3_EXPECTED = [f"d{d}_p97_s{s}.csv" for d in PHASE3_DEPTHS for s in SEEDS]

# Minimum file size to consider "complete" (full 200k step run ~ 117KB)
MIN_FILE_SIZE = 100_000  # bytes

LOG_FILE = RESULTS_DIR / "orchestrator.log"


def log(msg):
    """Print and log a message with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_phase_complete(directory, expected_files, min_size=MIN_FILE_SIZE):
    """Check if all expected files exist and are large enough."""
    directory = Path(directory)
    if not directory.exists():
        return False, []
    
    missing = []
    incomplete = []
    for fname in expected_files:
        fpath = directory / fname
        if not fpath.exists():
            missing.append(fname)
        elif fpath.stat().st_size < min_size:
            incomplete.append(f"{fname} ({fpath.stat().st_size / 1024:.0f}KB)")
    
    if missing or incomplete:
        return False, missing + incomplete
    return True, []


def wait_for_phase(phase_name, directory, expected_files, poll_interval=60):
    """Poll until a phase is complete."""
    log(f"Waiting for {phase_name} to complete...")
    log(f"  Monitoring: {directory}")
    log(f"  Expected files: {len(expected_files)}")
    
    while True:
        done, remaining = check_phase_complete(directory, expected_files)
        if done:
            log(f"{phase_name} COMPLETE -- all {len(expected_files)} files present and valid.")
            return True
        
        # Count progress
        directory_path = Path(directory)
        existing = 0
        if directory_path.exists():
            for f in expected_files:
                fp = directory_path / f
                if fp.exists() and fp.stat().st_size >= MIN_FILE_SIZE:
                    existing += 1
        
        log(f"  {phase_name}: {existing}/{len(expected_files)} complete. "
            f"Remaining: {remaining[:3]}{'...' if len(remaining) > 3 else ''}. "
            f"Polling again in {poll_interval}s...")
        time.sleep(poll_interval)


def run_phase(phase_num):
    """Run a specific phase using run_experiment.py."""
    log(f"=" * 60)
    log(f"LAUNCHING PHASE {phase_num}")
    log(f"=" * 60)
    
    cmd = [sys.executable, "run_experiment.py", "--phase", str(phase_num)]
    log(f"  Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).parent),
    )
    
    # Stream output in real-time
    for line in process.stdout:
        line = line.rstrip()
        if line:
            # Only log important lines to avoid flooding
            if any(kw in line for kw in [
                "PHASE", "GROKKED", "FAILED", "t_grok", "Running",
                "SUMMARY", "COMPLETE", "===", "---", "WARNING",
                "Step", "ENVIRONMENT", "Using device"
            ]):
                log(f"  [P{phase_num}] {line}")
    
    process.wait()
    log(f"Phase {phase_num} process exited with code {process.returncode}")
    return process.returncode


def run_analysis():
    """Run analyze.py for all completed phases."""
    log(f"=" * 60)
    log(f"RUNNING FINAL ANALYSIS")
    log(f"=" * 60)
    
    for phase in [2, 3, 4]:
        phase_dir = RESULTS_DIR / f"phase{phase}"
        if phase_dir.exists() and any(phase_dir.iterdir()):
            log(f"Analyzing Phase {phase}...")
            cmd = [sys.executable, "analyze.py", "--phase", str(phase)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent),
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    log(f"  [analyze P{phase}] {line}")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    log(f"  [analyze P{phase} ERR] {line}")
        else:
            log(f"Phase {phase} directory not found or empty, skipping analysis.")
    
    # Also run the combined analysis if available
    log("Running combined analysis across all phases...")
    cmd = [sys.executable, "analyze.py", "--phase", "all"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log(f"  [analyze ALL] {line}")
    if result.returncode != 0 and result.stderr:
        log(f"  [analyze ALL] Note: combined analysis returned code {result.returncode}")
        for line in result.stderr.strip().split("\n")[-5:]:
            log(f"  [analyze ALL ERR] {line}")


def extract_final_results():
    """Extract and log the final t_grok table from all phases."""
    import csv
    
    log("=" * 60)
    log("FINAL RESULTS EXTRACTION")
    log("=" * 60)
    
    # Phase 2: t_grok by modulus
    log("\n--- Phase 2: t_grok vs Modulus p ---")
    log(f"{'p':>6} | {'s0':>6} | {'s1':>6} | {'s2':>6} | {'mean':>8} | {'std':>6}")
    log("-" * 50)
    
    for p in PHASE2_PRIMES:
        t_groks = []
        for s in SEEDS:
            fpath = PHASE2_DIR / f"p{p}_s{s}.csv"
            t_grok = _extract_t_grok(fpath)
            t_groks.append(t_grok)
        
        vals = [t for t in t_groks if t is not None]
        mean = sum(vals) / len(vals) if vals else 0
        if len(vals) > 1:
            std = (sum((v - mean)**2 for v in vals) / (len(vals) - 1)) ** 0.5
        else:
            std = 0
        
        t_strs = [str(t) if t else "N/A" for t in t_groks]
        log(f"{p:>6} | {t_strs[0]:>6} | {t_strs[1]:>6} | {t_strs[2]:>6} | "
            f"{mean:>8.0f} | {std:>6.0f}")
    
    # Phase 3: t_grok by depth
    if PHASE3_DIR.exists():
        log("\n--- Phase 3: t_grok vs Depth ---")
        log(f"{'depth':>6} | {'s0':>6} | {'s1':>6} | {'s2':>6} | {'mean':>8} | {'std':>6}")
        log("-" * 50)
        
        for d in PHASE3_DEPTHS:
            t_groks = []
            for s in SEEDS:
                fpath = PHASE3_DIR / f"d{d}_p97_s{s}.csv"
                t_grok = _extract_t_grok(fpath)
                t_groks.append(t_grok)
            
            vals = [t for t in t_groks if t is not None]
            mean = sum(vals) / len(vals) if vals else 0
            if len(vals) > 1:
                std = (sum((v - mean)**2 for v in vals) / (len(vals) - 1)) ** 0.5
            else:
                std = 0
            
            t_strs = [str(t) if t else "N/A" for t in t_groks]
            log(f"{d:>6} | {t_strs[0]:>6} | {t_strs[1]:>6} | {t_strs[2]:>6} | "
                f"{mean:>8.0f} | {std:>6.0f}")


def _extract_t_grok(csv_path, threshold=0.95):
    """Extract t_grok from a CSV file."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['test_acc']) >= threshold:
                    return int(row['step'])
    except Exception:
        pass
    return None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    log("=" * 60)
    log("GROKKING EXPERIMENT ORCHESTRATOR STARTED")
    log("=" * 60)
    log(f"Working directory: {Path.cwd()}")
    log(f"Python: {sys.executable}")
    log("")
    
    # -- Step 1: Wait for Phase 2 --
    phase2_done, _ = check_phase_complete(PHASE2_DIR, PHASE2_EXPECTED)
    if phase2_done:
        log("Phase 2 already complete! Skipping wait.")
    else:
        wait_for_phase("Phase 2", PHASE2_DIR, PHASE2_EXPECTED, poll_interval=60)
    
    # Small delay to ensure files are fully flushed
    time.sleep(5)
    
    # -- Step 2: Run Phase 3 --
    phase3_done, _ = check_phase_complete(PHASE3_DIR, PHASE3_EXPECTED)
    if phase3_done:
        log("Phase 3 already complete! Skipping.")
    else:
        ret = run_phase(3)
        if ret != 0:
            log(f"WARNING: Phase 3 exited with code {ret}, continuing anyway...")
    
    # -- Step 3: Run Phase 4 --
    # Phase 4 has many configs, check if directory has enough files
    phase4_has_data = PHASE4_DIR.exists() and len(list(PHASE4_DIR.glob("*.csv"))) >= 18
    if phase4_has_data:
        log("Phase 4 already has data! Skipping.")
    else:
        ret = run_phase(4)
        if ret != 0:
            log(f"WARNING: Phase 4 exited with code {ret}, continuing anyway...")
    
    # -- Step 4: Extract results and run analysis --
    extract_final_results()
    run_analysis()
    
    # -- Done --
    log("")
    log("=" * 60)
    log("ALL EXPERIMENTS AND ANALYSIS COMPLETE")
    log("=" * 60)
    log(f"Results in: {RESULTS_DIR.absolute()}")
    log(f"Log file: {LOG_FILE.absolute()}")
    
    # Write completion flag
    flag = RESULTS_DIR / "EXPERIMENT_COMPLETE.flag"
    with open(flag, "w") as f:
        f.write(f"Completed at {datetime.now().isoformat()}\n")
    log(f"Completion flag written to: {flag}")


if __name__ == "__main__":
    main()
