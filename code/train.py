"""
Training engine for grokking experiments.

Handles:
  - Single-run training with periodic evaluation
  - Logging to CSV
  - Grokking detection (t_grok)
  - Learning rate scheduling (optional warmup)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import csv
import os
from pathlib import Path

from model import GrokkingTransformer, count_parameters
from data import make_dataset, prepare_tensors, get_vocab_size, get_seq_len


def train_single_run(
    p,
    depth=1,
    seed=0,
    steps=100_000,
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
):
    """
    Train a single grokking run and log all metrics.
    
    Args:
        p: modulus
        depth: composition depth (1, 2, or 3)
        seed: random seed
        steps: total training steps
        eval_every: evaluation frequency
        weight_decay: AdamW weight decay
        lr: learning rate
        batch_size: training batch size
        dropout: dropout rate
        label_smoothing: label smoothing factor
        train_frac: fraction of data for training
        output_dir: directory for output files
        run_name: name for this run (auto-generated if None)
        device: 'cuda' or 'cpu'
        grok_threshold: test accuracy threshold for grokking detection
        warmup_steps: linear warmup steps
    
    Returns:
        dict with run metadata and results
    """
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if run_name is None:
        run_name = f"p{p}_d{depth}_s{seed}_do{dropout}_ls{label_smoothing}"
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{run_name}.csv")
    
    # Data
    print(f"[{run_name}] Generating dataset: p={p}, depth={depth}...")
    data = make_dataset(p, depth=depth, seed=seed)
    X_tr, y_tr, X_te, y_te = prepare_tensors(data, train_frac=train_frac, seed=seed)
    
    print(f"  Train samples: {len(X_tr)}, Test samples: {len(X_te)}")
    
    # Model
    vocab_size = get_vocab_size(p)
    seq_len = get_seq_len(depth)
    model = GrokkingTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=seq_len,
        dropout=dropout,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"  Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    
    # Learning rate scheduler: linear warmup then constant
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Move data to device
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)
    
    # Training loop
    logs = []
    t_grok = None
    t_train_saturated = None
    train_saturated = False
    start_time = time.time()
    
    # CSV header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc',
                         'elapsed_sec', 'lr'])
    
    print(f"  Training for {steps} steps...")
    
    for step in range(steps):
        model.train()
        
        # Sample batch
        idx = torch.randint(0, len(X_tr), (min(batch_size, len(X_tr)),), device=device)
        logits = model(X_tr[idx])
        loss = criterion(logits, y_tr[idx])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluate
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Train metrics -- evaluate on full train set in chunks
                tr_logits = _eval_chunked(model, X_tr, batch_size=2048)
                tr_loss = nn.functional.cross_entropy(tr_logits, y_tr).item()
                tr_acc = (tr_logits.argmax(-1) == y_tr).float().mean().item()
                
                # Test metrics
                te_logits = _eval_chunked(model, X_te, batch_size=2048)
                te_loss = nn.functional.cross_entropy(te_logits, y_te).item()
                te_acc = (te_logits.argmax(-1) == y_te).float().mean().item()
            
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            
            row = {
                'step': step,
                'train_loss': tr_loss,
                'train_acc': tr_acc,
                'test_loss': te_loss,
                'test_acc': te_acc,
                'elapsed_sec': elapsed,
                'lr': current_lr,
            }
            logs.append(row)
            
            # Write to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                                f"{te_loss:.6f}", f"{te_acc:.6f}",
                                f"{elapsed:.1f}", f"{current_lr:.6f}"])
            
            # Detect train saturation
            if not train_saturated and tr_acc > grok_threshold:
                t_train_saturated = step
                train_saturated = True
            
            # Detect grokking
            if t_grok is None and te_acc > grok_threshold:
                t_grok = step
            
            # Progress print
            if step % (eval_every * 50) == 0:
                grok_str = f"t_grok={t_grok}" if t_grok else "not yet"
                print(f"  Step {step:>7d} | train_acc={tr_acc:.4f} test_acc={te_acc:.4f} "
                      f"loss={tr_loss:.4f} | {grok_str} | {elapsed:.0f}s")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        te_logits = _eval_chunked(model, X_te, batch_size=2048)
        final_test_acc = (te_logits.argmax(-1) == y_te).float().mean().item()
    
    total_time = time.time() - start_time
    
    # Result summary
    result = {
        'run_name': run_name,
        'p': p,
        'depth': depth,
        'seed': seed,
        'dropout': dropout,
        'label_smoothing': label_smoothing,
        'steps': steps,
        'n_params': n_params,
        'n_train': len(X_tr),
        'n_test': len(X_te),
        't_grok': t_grok,
        't_train_saturated': t_train_saturated,
        'final_test_acc': final_test_acc,
        'total_time_sec': total_time,
        'csv_path': csv_path,
    }
    
    status = "GROKKED" if t_grok is not None else "NOT GROKKED"
    print(f"\n  [{run_name}] {status}")
    if t_grok is not None:
        print(f"    t_grok = {t_grok} steps")
        print(f"    t_train_saturated = {t_train_saturated} steps")
        print(f"    grokking_ratio = {t_grok / max(t_train_saturated, 1):.1f}x")
    print(f"    final_test_acc = {final_test_acc:.4f}")
    print(f"    total_time = {total_time:.1f}s")
    print()
    
    return result


def _eval_chunked(model, X, batch_size=2048):
    """Evaluate model on data in chunks to avoid OOM."""
    outputs = []
    for i in range(0, len(X), batch_size):
        chunk = X[i:i+batch_size]
        outputs.append(model(chunk))
    return torch.cat(outputs, dim=0)
