"""
Analysis script for the Grokking Experiment.
Reads all phase results, produces plots and statistical fits.
Run after all phases complete, or incrementally for available data.

Usage:
    python analyze.py                # Analyze all available phases
    python analyze.py --phase 2      # Analyze Phase 2 only
"""

import csv
import glob
import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.optimize import curve_fit
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("WARNING: matplotlib/scipy not available. Text-only analysis.")


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def load_run_csv(csv_path):
    """Load a single run's CSV and extract key metrics."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'step': int(row['step']),
                'train_loss': float(row['train_loss']),
                'train_acc': float(row['train_acc']),
                'test_loss': float(row['test_loss']),
                'test_acc': float(row['test_acc']),
            })
    return rows


def find_t_grok(rows, threshold=0.95):
    """Find first step where test_acc exceeds threshold."""
    for row in rows:
        if row['test_acc'] > threshold:
            return row['step']
    return None


def find_t_train(rows, threshold=0.95):
    """Find first step where train_acc exceeds threshold."""
    for row in rows:
        if row['train_acc'] > threshold:
            return row['step']
    return None


# --- Phase 1 Analysis -------------------------------------------------------

def analyze_phase1():
    """Analyze baseline grokking reproduction."""
    print("\n" + "=" * 70)
    print("PHASE 1 ANALYSIS -- Baseline Grokking")
    print("=" * 70)

    csv_files = glob.glob(str(RESULTS_DIR / "phase1" / "baseline*.csv"))
    if not csv_files:
        print("  No Phase 1 data found.")
        return

    for f in csv_files:
        rows = load_run_csv(f)
        t_grok = find_t_grok(rows)
        t_train = find_t_train(rows)
        final_te = rows[-1]['test_acc'] if rows else 0

        name = os.path.basename(f)
        ratio = t_grok / max(t_train, 1) if t_grok and t_train else None
        print(f"  {name}: t_train={t_train}, t_grok={t_grok}, "
              f"ratio={ratio:.1f}x, final_test={final_te:.4f}")

        if HAS_PLOTTING:
            steps = [r['step'] for r in rows]
            tr_acc = [r['train_acc'] for r in rows]
            te_acc = [r['test_acc'] for r in rows]

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(steps, tr_acc, label='Train Acc', alpha=0.8)
            ax.plot(steps, te_acc, label='Test Acc', alpha=0.8)
            ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='Grok threshold')
            if t_grok:
                ax.axvline(t_grok, color='red', linestyle=':', alpha=0.7, label=f't_grok={t_grok}')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Phase 1: Baseline Grokking (p=97)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            FIGURES_DIR.mkdir(exist_ok=True)
            fig.savefig(FIGURES_DIR / 'fig_baseline_grokking_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: figures/fig_baseline_grokking_curve.png")


# --- Phase 2 Analysis -------------------------------------------------------

def analyze_phase2():
    """Analyze K(X) variation via modulus p."""
    print("\n" + "=" * 70)
    print("PHASE 2 ANALYSIS -- t_grok vs p (K complexity)")
    print("=" * 70)

    csv_files = sorted(glob.glob(str(RESULTS_DIR / "phase2" / "p*.csv")))
    if not csv_files:
        print("  No Phase 2 data found.")
        return

    data = []
    for f in csv_files:
        name = os.path.basename(f).replace('.csv', '')
        parts = name.split('_')
        p = int(parts[0][1:])
        seed = int(parts[1][1:])
        rows = load_run_csv(f)
        t_grok = find_t_grok(rows)
        t_train = find_t_train(rows)
        last_step = rows[-1]['step'] if rows else 0
        final_te = rows[-1]['test_acc'] if rows else 0
        complete = last_step >= 199000
        data.append({
            'p': p, 'seed': seed, 't_grok': t_grok, 't_train': t_train,
            'final_te': final_te, 'complete': complete, 'last_step': last_step
        })

    # Print table
    print(f"  {'p':>5} {'seed':>4} {'t_grok':>8} {'t_train':>8} {'final_te':>9} {'status':>12}")
    print("  " + "-" * 50)
    for d in data:
        tg = str(d['t_grok']) if d['t_grok'] else 'N/A'
        tt = str(d['t_train']) if d['t_train'] else 'N/A'
        st = 'DONE' if d['complete'] else 'running(%d)' % d['last_step']
        print(f"  {d['p']:>5} {d['seed']:>4} {tg:>8} {tt:>8} {d['final_te']:>9.4f} {st:>12}")

    # Group by p, compute mean/std of t_grok
    from collections import defaultdict
    by_p = defaultdict(list)
    for d in data:
        if d['t_grok'] is not None:
            by_p[d['p']].append(d['t_grok'])

    if len(by_p) < 2:
        print("\n  Insufficient completed data for fitting. Need >=2 p values with grokking.")
        return

    print(f"\n  {'p':>5} {'n':>3} {'mean_t_grok':>12} {'std':>8}")
    print("  " + "-" * 35)
    ps, means, stds = [], [], []
    for p in sorted(by_p.keys()):
        vals = by_p[p]
        m = np.mean(vals)
        s = np.std(vals) if len(vals) > 1 else 0
        ps.append(p)
        means.append(m)
        stds.append(s)
        print(f"  {p:>5} {len(vals):>3} {m:>12.0f} {s:>8.0f}")

    ps = np.array(ps, dtype=float)
    means = np.array(means)
    stds = np.array(stds)

    # Log-log fit: t_grok = c * p^alpha
    log_p = np.log(ps)
    log_t = np.log(means)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_p, log_t)
    alpha = slope
    c = np.exp(intercept)

    print(f"\n  Log-log fit: t_grok = {c:.2f} * p^{alpha:.3f}")
    print(f"  R2 = {r_value**2:.4f}, p-value = {p_value:.4e}")
    print(f"  alpha = {alpha:.3f} +/- {std_err:.3f}")

    if alpha > 0:
        print(f"\n  P1 VERDICT: alpha = {alpha:.3f} > 0 -> SUPPORTED (super-constant scaling)")
    else:
        print(f"\n  P1 VERDICT: alpha = {alpha:.3f} <= 0 -> REFUTED")

    if HAS_PLOTTING:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Linear plot
        ax1.errorbar(ps, means, yerr=stds, fmt='o-', capsize=4, markersize=8)
        ax1.set_xlabel('Modulus p')
        ax1.set_ylabel('t_grok (steps)')
        ax1.set_title('Phase 2: t_grok vs p')
        ax1.grid(True, alpha=0.3)

        # Log-log plot
        ax2.errorbar(ps, means, yerr=stds, fmt='o', capsize=4, markersize=8)
        p_fit = np.linspace(ps.min() * 0.8, ps.max() * 1.2, 100)
        ax2.plot(p_fit, c * p_fit**alpha, 'r--',
                 label=f'Fit: {c:.1f}·p^{alpha:.2f} (R2={r_value**2:.3f})')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Modulus p (log)')
        ax2.set_ylabel('t_grok (log)')
        ax2.set_title('Phase 2: Log-log fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')

        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / 'fig_tgrok_vs_p_loglog.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/fig_tgrok_vs_p_loglog.png")

    return {'alpha': alpha, 'std_err': std_err, 'r2': r_value**2, 'by_p': {int(p): float(m) for p, m in zip(ps, means)}}


# -- Phase 3 Analysis -------------------------------------------------------

def analyze_phase3():
    """Analyze depth variation."""
    print("\n" + "=" * 70)
    print("PHASE 3 ANALYSIS -- t_grok vs Depth")
    print("=" * 70)

    csv_files = sorted(glob.glob(str(RESULTS_DIR / "phase3" / "d*.csv")))
    if not csv_files:
        print("  No Phase 3 data found.")
        return

    data = []
    for f in csv_files:
        name = os.path.basename(f).replace('.csv', '')
        parts = name.split('_')
        depth = int(parts[0][1:])
        seed = int(parts[2][1:])
        rows = load_run_csv(f)
        t_grok = find_t_grok(rows)
        last_step = rows[-1]['step'] if rows else 0
        final_te = rows[-1]['test_acc'] if rows else 0
        complete = last_step >= 299000
        data.append({'depth': depth, 'seed': seed, 't_grok': t_grok,
                     'final_te': final_te, 'complete': complete})

    print(f"  {'depth':>5} {'seed':>4} {'t_grok':>10} {'final_te':>10} {'status':>10}")
    print("  " + "-" * 45)
    for d in data:
        tg = str(d['t_grok']) if d['t_grok'] else 'N/A'
        st = 'DONE' if d['complete'] else 'running'
        print(f"  {d['depth']:>5} {d['seed']:>4} {tg:>10} {d['final_te']:>10.4f} {st:>10}")

    # Group by depth
    from collections import defaultdict
    by_depth = defaultdict(list)
    for d in data:
        if d['t_grok'] is not None:
            by_depth[d['depth']].append(d['t_grok'])

    if len(by_depth) < 2:
        print("\n  Insufficient data for depth analysis.")
        return

    print(f"\n  {'depth':>5} {'n':>3} {'mean_t_grok':>12} {'std':>8}")
    print("  " + "-" * 35)
    depths, means = [], []
    for depth in sorted(by_depth.keys()):
        vals = by_depth[depth]
        m = np.mean(vals)
        s = np.std(vals) if len(vals) > 1 else 0
        depths.append(depth)
        means.append(m)
        print(f"  {depth:>5} {len(vals):>3} {m:>12.0f} {s:>8.0f}")

    depths = np.array(depths, dtype=float)
    means = np.array(means)

    # Fit polynomial: t = c * depth^gamma
    if len(depths) >= 2:
        log_d = np.log(depths)
        log_t = np.log(means)
        slope, intercept, r_poly, _, _ = stats.linregress(log_d, log_t)
        print(f"\n  Polynomial fit: t_grok ~ depth^{slope:.2f} (R2={r_poly**2:.4f})")

    # Fit exponential: t = c * beta^depth
    if len(depths) >= 2:
        log_t = np.log(means)
        slope_e, intercept_e, r_exp, _, _ = stats.linregress(depths, log_t)
        print(f"  Exponential fit: t_grok ~ {np.exp(slope_e):.2f}^depth (R2={r_exp**2:.4f})")

        if r_exp**2 > r_poly**2:
            print(f"\n  P2 VERDICT: Exponential fit better -> STRONG SUPPORT")
        elif slope > 1:
            print(f"\n  P2 VERDICT: Super-linear polynomial -> SUPPORTED")
        else:
            print(f"\n  P2 VERDICT: Sub-linear or linear -> REFUTED")

    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(depths, means, width=0.5, alpha=0.7)
        ax.set_xlabel('Composition Depth')
        ax.set_ylabel('t_grok (steps)')
        ax.set_title('Phase 3: t_grok vs Depth')
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3, axis='y')
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / 'fig_tgrok_vs_depth.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/fig_tgrok_vs_depth.png")


# --- Phase 4 Analysis -------------------------------------------------------

def analyze_phase4():
    """Analyze noise/dropout variation."""
    print("\n" + "=" * 70)
    print("PHASE 4 ANALYSIS -- t_grok vs Noise")
    print("=" * 70)

    csv_files = sorted(glob.glob(str(RESULTS_DIR / "phase4" / "*.csv")))
    if not csv_files:
        print("  No Phase 4 data found.")
        return

    data = []
    for f in csv_files:
        name = os.path.basename(f).replace('.csv', '')
        # Parse: dropout0.1_ls0.0_s0
        parts = name.split('_')
        dropout = float(parts[0].replace('dropout', ''))
        ls = float(parts[1].replace('ls', ''))
        seed = int(parts[2].replace('s', ''))
        rows = load_run_csv(f)
        t_grok = find_t_grok(rows)
        final_te = rows[-1]['test_acc'] if rows else 0
        data.append({'dropout': dropout, 'ls': ls, 'seed': seed,
                     't_grok': t_grok, 'final_te': final_te})

    print(f"  {'dropout':>8} {'ls':>6} {'seed':>4} {'t_grok':>8} {'final_te':>10}")
    print("  " + "-" * 45)
    for d in data:
        tg = str(d['t_grok']) if d['t_grok'] else 'N/A'
        print(f"  {d['dropout']:>8.2f} {d['ls']:>6.2f} {d['seed']:>4} {tg:>8} {d['final_te']:>10.4f}")

    # Group by noise type
    from collections import defaultdict

    # Dropout sweep
    do_groups = defaultdict(list)
    for d in data:
        if d['ls'] == 0.0 and d['t_grok'] is not None:
            do_groups[d['dropout']].append(d['t_grok'])

    if do_groups:
        print(f"\n  Dropout sweep (ls=0.0):")
        print(f"  {'dropout':>8} {'mean_t_grok':>12}")
        for do in sorted(do_groups.keys()):
            print(f"  {do:>8.2f} {np.mean(do_groups[do]):>12.0f}")

    # Label smoothing sweep
    ls_groups = defaultdict(list)
    for d in data:
        if d['dropout'] == 0.0 and d['t_grok'] is not None:
            ls_groups[d['ls']].append(d['t_grok'])

    if ls_groups:
        print(f"\n  Label smoothing sweep (dropout=0.0):")
        print(f"  {'ls':>8} {'mean_t_grok':>12}")
        for ls in sorted(ls_groups.keys()):
            print(f"  {ls:>8.2f} {np.mean(ls_groups[ls]):>12.0f}")

    if HAS_PLOTTING and (do_groups or ls_groups):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if do_groups:
            dos = sorted(do_groups.keys())
            do_means = [np.mean(do_groups[d]) for d in dos]
            ax1.bar(range(len(dos)), do_means, tick_label=[str(d) for d in dos], alpha=0.7)
            ax1.set_xlabel('Dropout')
            ax1.set_ylabel('t_grok')
            ax1.set_title('t_grok vs Dropout')
            ax1.grid(True, alpha=0.3, axis='y')

        if ls_groups:
            lss = sorted(ls_groups.keys())
            ls_means = [np.mean(ls_groups[l]) for l in lss]
            ax2.bar(range(len(lss)), ls_means, tick_label=[str(l) for l in lss], alpha=0.7, color='orange')
            ax2.set_xlabel('Label Smoothing')
            ax2.set_ylabel('t_grok')
            ax2.set_title('t_grok vs Label Smoothing')
            ax2.grid(True, alpha=0.3, axis='y')

        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / 'fig_tgrok_vs_noise.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/fig_tgrok_vs_noise.png")


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all',
                       choices=['1', '2', '3', '4', 'all'])
    args = parser.parse_args()

    if args.phase in ['1', 'all']:
        analyze_phase1()
    if args.phase in ['2', 'all']:
        analyze_phase2()
    if args.phase in ['3', 'all']:
        analyze_phase3()
    if args.phase in ['4', 'all']:
        analyze_phase4()

    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
