"""
train_v2.py -- Training engine for Fase II.

Extensions vs train.py (kept identical otherwise):
  - optimizer parameter: 'adamw' (default) | 'sgd_momentum'
  - init_scheme parameter: 'xavier' (default) | 'kaiming' (passes to model_v2)

Returns the same result dict, plus 'optimizer' and 'init_scheme' fields for tracking.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import csv
import os

from model_v2 import GrokkingTransformerV2, count_parameters
from data import make_dataset, prepare_tensors, get_vocab_size, get_seq_len


def train_single_run_v2(
    p,
    depth=1,
    seed=0,
    steps=200_000,
    eval_every=100,
    weight_decay=1.0,
    lr=1e-3,
    batch_size=512,
    dropout=0.0,
    label_smoothing=0.0,
    train_frac=0.5,
    output_dir='results',
    run_name=None,
    device='cuda',
    grok_threshold=0.95,
    warmup_steps=100,
    optimizer='adamw',
    init_scheme='xavier',
):
    """Same semantics as train_single_run, with optimizer + init_scheme parameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if run_name is None:
        run_name = (f"p{p}_d{depth}_s{seed}_do{dropout}_ls{label_smoothing}"
                    f"_opt{optimizer}_init{init_scheme}_bs{batch_size}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{run_name}.csv")

    print(f"[{run_name}] Generating dataset: p={p}, depth={depth}...")
    data = make_dataset(p, depth=depth, seed=seed)
    X_tr, y_tr, X_te, y_te = prepare_tensors(data, train_frac=train_frac, seed=seed)
    print(f"  Train samples: {len(X_tr)}, Test samples: {len(X_te)}")

    vocab_size = get_vocab_size(p)
    seq_len = get_seq_len(depth)
    model = GrokkingTransformerV2(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=seq_len,
        dropout=dropout,
        init_scheme=init_scheme,
    ).to(device)

    n_params = count_parameters(model)
    print(f"  Model parameters: {n_params:,}  |  optimizer={optimizer}  init={init_scheme}")

    # Optimizer selection
    if optimizer == 'adamw':
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )
    elif optimizer == 'sgd_momentum':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Use 'adamw' or 'sgd_momentum'.")

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)

    t_grok = None
    t_train_saturated = None
    train_saturated = False
    start_time = time.time()
    diverged = False

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc',
                         'elapsed_sec', 'lr'])

    print(f"  Training for {steps} steps...")

    for step in range(steps):
        model.train()
        idx = torch.randint(0, len(X_tr), (min(batch_size, len(X_tr)),), device=device)
        logits = model(X_tr[idx])
        loss = criterion(logits, y_tr[idx])

        opt.zero_grad()
        loss.backward()
        # Soft divergence guard for SGD with high lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()
        scheduler.step()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                tr_logits = _eval_chunked(model, X_tr, batch_size=2048)
                tr_loss = nn.functional.cross_entropy(tr_logits, y_tr).item()
                tr_acc = (tr_logits.argmax(-1) == y_tr).float().mean().item()
                te_logits = _eval_chunked(model, X_te, batch_size=2048)
                te_loss = nn.functional.cross_entropy(te_logits, y_te).item()
                te_acc = (te_logits.argmax(-1) == y_te).float().mean().item()

            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                                 f"{te_loss:.6f}", f"{te_acc:.6f}",
                                 f"{elapsed:.1f}", f"{current_lr:.6f}"])

            if not train_saturated and tr_acc > grok_threshold:
                t_train_saturated = step
                train_saturated = True
            if t_grok is None and te_acc > grok_threshold:
                t_grok = step

            if step % (eval_every * 50) == 0:
                grok_str = f"t_grok={t_grok}" if t_grok else "not yet"
                print(f"  Step {step:>7d} | tr_acc={tr_acc:.4f} te_acc={te_acc:.4f} "
                      f"loss={tr_loss:.4f} | {grok_str} | {elapsed:.0f}s")

            # Divergence check (loss explodes or NaN)
            if not (tr_loss == tr_loss) or tr_loss > 100:  # NaN check + explosion
                print(f"  DIVERGED at step {step}: tr_loss={tr_loss}")
                diverged = True
                break

    model.eval()
    with torch.no_grad():
        te_logits = _eval_chunked(model, X_te, batch_size=2048)
        final_test_acc = (te_logits.argmax(-1) == y_te).float().mean().item()

    total_time = time.time() - start_time

    result = {
        'run_name': run_name,
        'p': p,
        'depth': depth,
        'seed': seed,
        'dropout': dropout,
        'label_smoothing': label_smoothing,
        'weight_decay': weight_decay,
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'init_scheme': init_scheme,
        'steps': steps,
        'n_params': n_params,
        'n_train': len(X_tr),
        'n_test': len(X_te),
        't_grok': t_grok,
        't_train_saturated': t_train_saturated,
        'final_test_acc': final_test_acc,
        'total_time_sec': total_time,
        'csv_path': csv_path,
        'diverged': diverged,
    }

    status = "GROKKED" if t_grok is not None else ("DIVERGED" if diverged else "NOT GROKKED")
    print(f"  [{run_name}] {status}  t_grok={t_grok}  acc={final_test_acc:.4f}  time={total_time:.0f}s\n")
    return result


def _eval_chunked(model, X, batch_size=2048):
    outputs = []
    for i in range(0, len(X), batch_size):
        outputs.append(model(X[i:i+batch_size]))
    return torch.cat(outputs, dim=0)
