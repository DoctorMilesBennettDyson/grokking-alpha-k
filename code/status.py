"""Quick status check -- run anytime to see current experiment progress."""
import csv, glob, os
from collections import defaultdict

def check():
    for phase in ['phase1', 'phase2', 'phase3', 'phase4']:
        d = os.path.join('results', phase)
        if not os.path.exists(d):
            continue
        files = sorted(glob.glob(os.path.join(d, '*.csv')))
        if not files:
            continue
        print('\n=== %s ===' % phase.upper())
        for f in files:
            name = os.path.basename(f).replace('.csv','')
            rows = list(csv.DictReader(open(f)))
            if not rows:
                print('  %s: empty' % name)
                continue
            last = rows[-1]
            last_step = int(last['step'])
            t_grok = None
            t_train = None
            for r in rows:
                if t_train is None and float(r['train_acc']) > 0.95:
                    t_train = int(r['step'])
                if t_grok is None and float(r['test_acc']) > 0.95:
                    t_grok = int(r['step'])
            
            total_bytes = os.path.getsize(f)
            status = 'GROK@%d' % t_grok if t_grok else 'step=%d' % last_step
            print('  %-30s %s  final_te=%.4f  (%dKB)' % (
                name, status, float(last['test_acc']), total_bytes//1024))
    
    # Summary for phase2
    p2_dir = os.path.join('results', 'phase2')
    if os.path.exists(p2_dir):
        by_p = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(p2_dir, 'p*.csv'))):
            name = os.path.basename(f).replace('.csv','')
            p = int(name.split('_')[0][1:])
            rows = list(csv.DictReader(open(f)))
            t_grok = None
            for r in rows:
                if t_grok is None and float(r['test_acc']) > 0.95:
                    t_grok = int(r['step'])
            if t_grok:
                by_p[p].append(t_grok)
        
        if by_p:
            print('\n--- Phase 2 Summary: mean t_grok by p ---')
            import numpy as np
            for p in sorted(by_p):
                vals = by_p[p]
                print('  p=%3d: n=%d  mean=%.0f  std=%.0f  vals=%s' % (
                    p, len(vals), np.mean(vals), np.std(vals) if len(vals)>1 else 0, vals))

if __name__ == '__main__':
    check()
