# grokking-alpha-K

**An empirical map of the grokking exponent α(K) on modular arithmetic, with pre-registered robustness checks — code, raw data, and pre-registrations.**

Author: **Pablo L. Rainieri Blasco** — independent researcher, Tucumán, Argentina.
Hardware: a single RTX 3070 (8 GB). Everything here reproduces on consumer hardware.

## What this is

Grokking time on modular addition follows a power law in weight decay, `t_grok ∝ wd^α`.
This project maps how the exponent α depends on task size K (training-set size, driven by
the modulus p) and stress-tests that map:

| p | K = n_train | α (log–log slope) |
|---|---|---|
| 31 | 480 | -0.926 |
| 53 | 1404 | -0.866 |
| 67 | 2244 | -0.861 |
| 97 | 4704 | -0.731 |
| 113 | 6384 | -0.736 |
| 149 | 11100 | -0.577 |

**|α| shrinks as the task grows.** Robustness (Fase II, pre-registered 2026-04-27 before
execution — `prereg/PRE_REGISTRATION_FASE_II.md`):

- **Init (H3):** Kaiming/Xavier t_grok ratio 1.28 (criterion ≤ 1.5) — not an init artifact.
- **Batch size (H4):** t_grok rises monotonically with bs (1067 → 1314 → 1467) — SGD noise
  *accelerates* grokking, contradicting a naive gradient-SNR reading.
- **Task composition (H1):** at depth-2 the wd power law survives (slope -0.658) **but the
  regime changes** — no memorization phase exists (train and test converge together), so
  this is *not* grokking-delay invariance. Reported as a regime-boundary finding.
- **Optimizer (H2, negative result):** plain SGD-momentum never groks at these settings.
  Mechanism identified: coupled L2 + momentum collapses the weight norm (42 → 0.02 within
  1K steps) before gradient signal competes; AdamW survives via adaptive normalization.
  The wd → complexity mapping is **AdamW-class-specific**.

## The integrity disclosure (read this)

`theory/theory_extension_v0.2.md` §1 documents that an earlier theory "pre-registration"
(v0.1 §4.3, frozen in `prereg/`) was found **compromised**: its published predictions match,
to ≤0.006, a fit using data they claimed to predate, and the file postdates those data on
disk. The confirmatory claim was vacated, the reconstruction that exposed it is committed
(`code/fit_alpha_K.py`), and the replacement instrument is
`prereg/PRE_REGISTRATION_P1_P4.md` — pushed to this repository **before** the P1/P4 runs
start, so the commit hash is the timestamp. We publish the failure because catching it is
what the method is for.

## Repository layout

```
code/      training engine + phase runners + analysis (PyTorch, single GPU)
results/   raw per-run CSVs + auto-generated REPORTs for every phase (incl. failures/DNFs)
figures/   paper figures (α(K) map, prediction reconstruction)
prereg/    frozen pre-registration artifacts (Fase II; P1/P4; v0.1 with audit changelog)
theory/    working theoretical framework (v0.2) + rigorous program
```

## Reproduce

```bash
pip install -r code/requirements.txt        # torch>=2.1 (CUDA), numpy, pandas, matplotlib, scipy
python code/phase9.py                       # current experiment (P1+P4, ~14 h on a 3070)
python code/fit_alpha_K.py                  # regenerates the α(K) reconstruction + figure
```

Every runner is idempotent and crash-safe (re-run to resume). Analysis rules
(sustained-t_grok definition, outlier and DNF handling) are fixed in the pre-registration
files, not in the code reader's goodwill.

## License

Code: Apache-2.0 (see `LICENSE`). Documents, data, and figures: CC BY 4.0.
If you use this, cite the repository until the paper preprint is up.
