"""
run_experiment.py -- Main entry point for the Grokking Experiment.

Phase-Transition Framework Falsification Test (Paper 04, S6.3).
Author: Pablo Luciano Rainieri
Hardware target: RTX 3070, 8 GB VRAM

Usage:
    python run_experiment.py --phase 1              # Baseline only
    python run_experiment.py --phase 2              # Vary K via p
    python run_experiment.py --phase 3              # Vary depth
    python run_experiment.py --phase 4              # Vary noise
    python run_experiment.py --phase all             # Full experiment
    python run_experiment.py --phase 1 --device cpu  # CPU fallback
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

from train import train_single_run


# --- Configuration ----------------------------------------------------------

RESULTS_DIR = Path("results")
SEEDS = [0, 1, 2]

# Phase 1: Baseline
BASELINE_P = 97
BASELINE_STEPS = 100_000

# Phase 2: Vary K via modulus p
PHASE2_PRIMES = [31, 53, 67, 97, 113, 149]
PHASE2_STEPS = 200_000

# Phase 3: Vary depth
PHASE3_P = 97
PHASE3_DEPTHS = [1, 2, 3]
PHASE3_STEPS = 300_000

# Phase 4: Vary noise
PHASE4_P = 97
PHASE4_DROPOUT_VALUES = [0.0, 0.05, 0.1, 0.2]
PHASE4_LABEL_SMOOTHING_VALUES = [0.0, 0.05, 0.1, 0.2]
PHASE4_STEPS = 100_000

# Common hyperparameters (matches Nanda et al. 2023 / Power et al. 2022)
COMMON_KWARGS = dict(
    weight_decay=1.0,
    lr=1e-3,
    batch_size=512,
    train_frac=0.5,
    eval_every=100,
    grok_threshold=0.95,
    warmup_steps=100,
)


# --- Phase Runners ----------------------------------------------------------

def run_phase1(device):
    """Phase 1: Baseline grokking reproduction with p=97."""
    print("=" * 80)
    print("PHASE 1 -- Baseline Grokking Reproduction")
    print(f"  Task: (a + b) mod {BASELINE_P}")
    print(f"  Steps: {BASELINE_STEPS}")
    print("=" * 80)
    
    output_dir = RESULTS_DIR / "phase1"
    result = train_single_run(
        p=BASELINE_P,
        depth=1,
        seed=0,
        steps=BASELINE_STEPS,
        output_dir=str(output_dir),
        run_name="baseline_p97_s0",
        device=device,
        **COMMON_KWARGS,
    )
    
    # Validate Phase 1
    if result['t_grok'] is None:
        print("\n" + "!" * 80)
        print("PHASE 1 FAILED -- Grokking not observed within step budget.")
        print("Possible causes:")
        print("  1. Weight decay too low (current: 1.0)")
        print("  2. Train fraction too high (current: 0.5, try 0.3)")
        print("  3. Insufficient steps")
        print("  4. Data pipeline issue")
        print("Attempting retry with train_frac=0.3...")
        print("!" * 80)
        
        # Retry with Nanda et al. hyperparameters
        retry_kwargs = {**COMMON_KWARGS, 'train_frac': 0.3}
        result = train_single_run(
            p=BASELINE_P,
            depth=1,
            seed=0,
            steps=BASELINE_STEPS,
            output_dir=str(output_dir),
            run_name="baseline_p97_s0_retry_frac03",
            device=device,
            **retry_kwargs,
        )
        
        if result['t_grok'] is None:
            print("\n" + "!" * 80)
            print("PHASE 1 FAILED ON RETRY -- STOPPING EXPERIMENT")
            print("Cannot proceed without baseline grokking reproduction.")
            print("!" * 80)
            return None
    
    grok_ratio = result['t_grok'] / max(result['t_train_saturated'], 1)
    print(f"\nPHASE 1 RESULT:")
    print(f"  t_grok = {result['t_grok']} steps")
    print(f"  t_train_saturated = {result['t_train_saturated']} steps")
    print(f"  Grokking ratio = {grok_ratio:.1f}x (expected: 10-100x)")
    
    if grok_ratio < 3:
        print("  WARNING: Grokking ratio is low -- grokking may not be clean.")
    
    _save_summary(result, output_dir / "phase1_summary.json")
    return result


def run_phase2(device):
    """Phase 2: Vary K(X) via modulus p."""
    print("\n" + "=" * 80)
    print("PHASE 2 -- Vary K(X) via Modulus p")
    print(f"  Primes: {PHASE2_PRIMES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Max steps: {PHASE2_STEPS}")
    print("=" * 80)
    
    output_dir = RESULTS_DIR / "phase2"
    results = []
    
    for p in PHASE2_PRIMES:
        for seed in SEEDS:
            run_name = f"p{p}_s{seed}"
            print(f"\n--- Running p={p}, seed={seed} ---")
            
            result = train_single_run(
                p=p,
                depth=1,
                seed=seed,
                steps=PHASE2_STEPS,
                output_dir=str(output_dir),
                run_name=run_name,
                device=device,
                **COMMON_KWARGS,
            )
            results.append(result)
    
    _save_summary(results, output_dir / "phase2_summary.json")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY")
    print(f"{'p':>6} | {'seed':>4} | {'t_grok':>10} | {'final_test_acc':>14} | {'time':>8}")
    print("-" * 55)
    for r in results:
        t_grok_str = str(r['t_grok']) if r['t_grok'] is not None else "N/A"
        print(f"{r['p']:>6} | {r['seed']:>4} | {t_grok_str:>10} | "
              f"{r['final_test_acc']:>14.4f} | {r['total_time_sec']:>7.0f}s")
    print("=" * 80)
    
    return results


def run_phase3(device):
    """Phase 3: Vary depth via composition."""
    print("\n" + "=" * 80)
    print("PHASE 3 -- Vary Depth via Composition")
    print(f"  p = {PHASE3_P}")
    print(f"  Depths: {PHASE3_DEPTHS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Max steps: {PHASE3_STEPS}")
    print("=" * 80)
    
    output_dir = RESULTS_DIR / "phase3"
    results = []
    
    for depth in PHASE3_DEPTHS:
        for seed in SEEDS:
            run_name = f"d{depth}_p{PHASE3_P}_s{seed}"
            print(f"\n--- Running depth={depth}, seed={seed} ---")
            
            result = train_single_run(
                p=PHASE3_P,
                depth=depth,
                seed=seed,
                steps=PHASE3_STEPS,
                output_dir=str(output_dir),
                run_name=run_name,
                device=device,
                **COMMON_KWARGS,
            )
            results.append(result)
    
    _save_summary(results, output_dir / "phase3_summary.json")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 3 SUMMARY")
    print(f"{'depth':>5} | {'seed':>4} | {'t_grok':>10} | {'final_test_acc':>14} | {'time':>8}")
    print("-" * 55)
    for r in results:
        t_grok_str = str(r['t_grok']) if r['t_grok'] is not None else "N/A"
        print(f"{r['depth']:>5} | {r['seed']:>4} | {t_grok_str:>10} | "
              f"{r['final_test_acc']:>14.4f} | {r['total_time_sec']:>7.0f}s")
    print("=" * 80)
    
    return results


def run_phase4(device):
    """Phase 4: Vary noise via dropout and label smoothing."""
    print("\n" + "=" * 80)
    print("PHASE 4 -- Vary eps via Noise Injection")
    print(f"  p = {PHASE4_P}, depth = 1")
    print(f"  Dropout values: {PHASE4_DROPOUT_VALUES}")
    print(f"  Label smoothing values: {PHASE4_LABEL_SMOOTHING_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Steps: {PHASE4_STEPS}")
    print("=" * 80)
    
    output_dir = RESULTS_DIR / "phase4"
    results = []
    
    # Test dropout (with label_smoothing=0)
    for dropout in PHASE4_DROPOUT_VALUES:
        for seed in SEEDS:
            run_name = f"dropout{dropout}_ls0.0_s{seed}"
            print(f"\n--- Running dropout={dropout}, ls=0.0, seed={seed} ---")
            
            result = train_single_run(
                p=PHASE4_P,
                depth=1,
                seed=seed,
                steps=PHASE4_STEPS,
                dropout=dropout,
                label_smoothing=0.0,
                output_dir=str(output_dir),
                run_name=run_name,
                device=device,
                **COMMON_KWARGS,
            )
            results.append(result)
    
    # Test label smoothing (with dropout=0) -- skip ls=0 since already covered
    for ls in PHASE4_LABEL_SMOOTHING_VALUES:
        if ls == 0.0:
            continue  # already covered in dropout sweep with dropout=0
        for seed in SEEDS:
            run_name = f"dropout0.0_ls{ls}_s{seed}"
            print(f"\n--- Running dropout=0.0, ls={ls}, seed={seed} ---")
            
            result = train_single_run(
                p=PHASE4_P,
                depth=1,
                seed=seed,
                steps=PHASE4_STEPS,
                dropout=0.0,
                label_smoothing=ls,
                output_dir=str(output_dir),
                run_name=run_name,
                device=device,
                **COMMON_KWARGS,
            )
            results.append(result)
    
    _save_summary(results, output_dir / "phase4_summary.json")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 4 SUMMARY")
    print(f"{'dropout':>8} | {'ls':>6} | {'seed':>4} | {'t_grok':>10} | "
          f"{'final_test_acc':>14} | {'time':>8}")
    print("-" * 65)
    for r in results:
        t_grok_str = str(r['t_grok']) if r['t_grok'] is not None else "N/A"
        print(f"{r['dropout']:>8.2f} | {r['label_smoothing']:>6.2f} | {r['seed']:>4} | "
              f"{t_grok_str:>10} | {r['final_test_acc']:>14.4f} | "
              f"{r['total_time_sec']:>7.0f}s")
    print("=" * 80)
    
    return results


# --- Utilities --------------------------------------------------------------

def _save_summary(data, path):
    """Save results summary as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Summary saved to {path}")


def check_environment():
    """Verify compute environment."""
    print("=" * 80)
    print("ENVIRONMENT CHECK")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No CUDA device found. Training will be slow on CPU.")
    print("=" * 80)


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grokking Experiment -- Phase-Transition Framework Test"
    )
    parser.add_argument('--phase', type=str, default='1',
                       choices=['1', '2', '3', '4', 'all'],
                       help='Which phase to run (1-4 or all)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    check_environment()
    print(f"\nUsing device: {device}")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    
    start = time.time()
    
    if args.phase in ['1', 'all']:
        p1_result = run_phase1(device)
        if p1_result is None and args.phase == 'all':
            print("Aborting remaining phases due to Phase 1 failure.")
            return
    
    if args.phase in ['2', 'all']:
        run_phase2(device)
    
    if args.phase in ['3', 'all']:
        run_phase3(device)
    
    if args.phase in ['4', 'all']:
        run_phase4(device)
    
    total = time.time() - start
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT COMPLETE -- Total time: {total/3600:.1f} hours")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
