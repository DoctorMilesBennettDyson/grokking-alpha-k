"""
Data generation for modular arithmetic grokking experiments.

Supports:
  - Depth-1: (a + b) mod p
  - Depth-2: ((a + b) mod p + c) mod p  
  - Depth-3: (((a + b) mod p + c) mod p + d) mod p

Each sample is a sequence of tokens: [operand, op, operand, eq] for depth-1,
extended with additional [op, operand] pairs for deeper compositions.
"""

import numpy as np
import torch


# Special token offsets (relative to p)
OP_OFFSET = 0   # op_token = p + 0
EQ_OFFSET = 1   # eq_token = p + 1


def make_dataset(p, depth=1, op='add', seed=None):
    """
    Generate all samples for modular arithmetic at given depth.
    
    Args:
        p: modulus (prime recommended)
        depth: 1, 2, or 3 -- number of composed additions
        op: 'add' or 'mul'
        seed: random seed for reproducibility (used for subsampling at depth>1)
    
    Returns:
        list of (input_sequence, target_value) tuples
    """
    op_fn = {
        'add': lambda a, b: (a + b) % p,
        'mul': lambda a, b: (a * b) % p,
    }[op]
    
    op_token = p + OP_OFFSET
    eq_token = p + EQ_OFFSET
    
    if depth == 1:
        # Full enumeration: p² samples
        data = []
        for a in range(p):
            for b in range(p):
                result = op_fn(a, b)
                seq = [a, op_token, b, eq_token]
                data.append((seq, result))
        return data
    
    elif depth == 2:
        # Full enumeration: p³ samples -- feasible for p ≤ 97
        # For p=97: 97³ = 912,673 samples
        data = []
        for a in range(p):
            for b in range(p):
                intermediate = op_fn(a, b)
                for c in range(p):
                    result = op_fn(intermediate, c)
                    seq = [a, op_token, b, op_token, c, eq_token]
                    data.append((seq, result))
        
        # If dataset is too large, subsample
        if len(data) > 500_000:
            rng = np.random.RandomState(seed if seed is not None else 42)
            indices = rng.choice(len(data), 500_000, replace=False)
            data = [data[i] for i in indices]
        
        return data
    
    elif depth == 3:
        # p⁴ is too large for p=97. Subsample.
        # Strategy: generate p³ samples by fixing enumeration pattern
        rng = np.random.RandomState(seed if seed is not None else 42)
        
        # Generate a manageable number of samples
        n_samples = min(p ** 3, 500_000)
        data = []
        seen = set()
        
        while len(data) < n_samples:
            a = rng.randint(0, p)
            b = rng.randint(0, p)
            c = rng.randint(0, p)
            d = rng.randint(0, p)
            
            key = (a, b, c, d)
            if key in seen:
                continue
            seen.add(key)
            
            r1 = op_fn(a, b)
            r2 = op_fn(r1, c)
            result = op_fn(r2, d)
            
            seq = [a, op_token, b, op_token, c, op_token, d, eq_token]
            data.append((seq, result))
        
        return data
    
    else:
        raise ValueError(f"Unsupported depth: {depth}")


def prepare_tensors(data, train_frac=0.5, seed=0):
    """
    Split data into train/test and convert to tensors.
    
    Args:
        data: list of (sequence, target) tuples
        train_frac: fraction of data for training
        seed: random seed for split reproducibility
    
    Returns:
        X_train, y_train, X_test, y_test as tensors
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(data))
    n_train = int(len(data) * train_frac)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = torch.tensor([data[i][0] for i in train_idx], dtype=torch.long)
    y_train = torch.tensor([data[i][1] for i in train_idx], dtype=torch.long)
    X_test = torch.tensor([data[i][0] for i in test_idx], dtype=torch.long)
    y_test = torch.tensor([data[i][1] for i in test_idx], dtype=torch.long)
    
    return X_train, y_train, X_test, y_test


def get_vocab_size(p):
    """Vocabulary: p digit tokens + 1 op token + 1 eq token."""
    return p + 2


def get_seq_len(depth):
    """Sequence length for a given composition depth."""
    # depth=1: [a, op, b, eq] = 4
    # depth=2: [a, op, b, op, c, eq] = 6
    # depth=3: [a, op, b, op, c, op, d, eq] = 8
    return 2 * (depth + 1)
