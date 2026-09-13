# TF-AdamW — code and results

Reference implementation and full experimental pipeline for

> **TF-AdamW: Tempered Fractional First-Moment Optimization with Adaptive
> Preconditioning for Robust Deep Learning**

TF-AdamW replaces AdamW's single exponential first moment by a normalized **tempered
fractional** kernel `κ_j = w_j^(α) e^(−λj)`, keeping RMS preconditioning and decoupled
weight decay. `SOE-TF-AdamW` realises that kernel with a bank of `M` exponential states, so
per-step cost and optimizer state are independent of the iteration count.

Everything in this repository runs on CPU. No GPU is required, and none was used for any
number in the paper.

---

## Quick start

```bash
pip install -r requirements.txt
python fetch_fmnist.py          # Fashion-MNIST -> ./data
python fetch_cifar.py           # CIFAR-10 -> ./data
python -c "import kernels as K; print(K.soe_fit_nnls(0.7, 0.05, 8))"
```

Using the optimizer on your own model:

```python
import torch, opt_zoo as Z, kernels as K

fit = K.soe_fit_nnls(alpha=0.7, lam=0.05, M=8)      # offline, no training data
opt = Z.SOETFAdamW(model.parameters(), lr=1e-3, fit=fit, weight_decay=5e-4)
```

`(α, λ, M) = (0.7, 0.05, 8)` are the paper's defaults. `λ ≈ α/W` sets the mean memory
length to about `W` steps. `M = 4` costs 2.5× Adam's optimizer state instead of 4.5× and is
statistically indistinguishable from `M = 8` in our experiments.

---

## What is here

| file | purpose |
|---|---|
| `kernels.py` | kernel identities, effective sample size, mean delay, certified NNLS sum-of-exponentials fit |
| `opt_zoo.py` | all 21 optimizers compared in the paper, including `SOETFAdamW` |
| `fetch_fmnist.py`, `fetch_cifar.py` | download the two datasets into `./data` |
| `dl_core.py` | datasets, models (SmallCNN, ResNet-20, character Transformer, LoRA), training loops |
| `build_jobs.py` | experiment definitions and job generation |
| `runner.py` | parallel worker/dispatch with per-job resume |
| `finish.py` | the full programme: tune → widen the grid → evaluate, per experiment |
| `extend_grid.py`, `extend_synth.py` | enforce the tuning rule: extend a method's grid until its selected learning rate is not at an edge |
| `synth.py`, `synth_shape.py` | controlled optimization over a distribution of noisy ill-conditioned problems, and the fixed-mean-delay kernel-shape sweep |
| `exp_geometry.py` | closed-form kernel geometry against brute-force summation |
| `exp_cost.py`, `exp_gradstats.py`, `exp_lora.py` | optimizer state and step time; gradient statistics under label corruption; LoRA-restricted state |
| `analyze.py` | paired statistics: bootstrap CIs, paired *t*, Wilcoxon, Holm–Bonferroni |
| `make_tables.py`, `make_figures.py` | regenerate every table and figure in the paper |
| `smoke_test.py` | one-minute check that the install works and reproduces the paper's kernel constants |
| `keepawake.py` | holds Windows out of Modern Standby while `finish.py` runs; without it an unattended run is suspended and lost |
| `check_numbers.py` | recompute every number and qualitative claim quoted in the paper and check it against the manuscript source |
| `results/` | analysis outputs (JSON) |
| `raw_results/` | **every individual run** — one JSON record per training run, 991 in total |
| `selected_lr_*.json` | the learning rate selected for each optimizer on each task |

## Reproducing the paper

The pipeline is resumable: a job already recorded is skipped, so it can be interrupted and
restarted freely.

```bash
python finish.py                 # all deep-learning stages, in order
python finish.py fmnist          # one stage
python synth.py                  # synthetic instance distribution
python synth_shape.py            # kernel-shape sweep at matched mean delay
python exp_geometry.py           # kernel geometry
python exp_cost.py               # state and step-time measurements
python exp_gradstats.py          # gradient statistics under label corruption
python exp_lora.py               # LoRA-restricted optimizer state
```

Then regenerate the analysis, tables and figures:

```bash
python analyze.py fmnist AdamW
python make_tables.py            # -> ./tables
python make_figures.py           # -> ./figures
```

To recompute the statistics **without re-training**, the raw per-run records are
sufficient. Copy each `raw_results/out_<experiment>_<stage>/` directory and the
`selected_lr_*.json` files into the working directory, then run `analyze.py`:

```bash
cp -r raw_results/out_* .
python analyze.py fmnist AdamW
```

The `selected_lr_*.json` files matter: re-tuning leaves records at superseded learning
rates in the same directory, and `analyze.py` filters on the selected rate so those cannot
contaminate a re-tuned cell. This path was checked from a clean directory and reproduces
the paper's numbers exactly (SOE-TF-AdamW 86.190 +- 0.389 against AdamW 86.03 at 40% label
noise, paired difference +0.158, p = 0.023, Holm p = 0.32).

The ResNet-20 stage dominates the compute: 55 runs of about 75 minutes each (tuning,
one widening round and evaluation), roughly 11–12 hours of wall-clock time with six workers
on an 8-core CPU. Every other stage together takes a few hours.

### Paths

Four locations are configurable by environment variable, with relative defaults:

| variable | default | used by |
|---|---|---|
| `TFADAMW_DATA` | `./data` | datasets |
| `TFADAMW_TABLES` | `./tables` | generated LaTeX tables |
| `TFADAMW_FIGURES` | `./figures` | generated PDF figures |
| `TFADAMW_MAIN` | `./main.tex` | `check_numbers.py`, the manuscript to verify against |

## Experimental protocol

Three rules are enforced by the code rather than by convention, because they materially
change the results:

1. **Selection and evaluation are disjoint**, in both the data and the metric. Learning
   rates are chosen on dedicated tuning seeds (and, on the synthetic tasks, dedicated
   problem instances) using validation accuracy; results are reported on unseen evaluation
   seeds using test accuracy.
2. **The grid is extended until no method sits at an edge** (`extend_grid.py`). This is not
   cosmetic: it improved AMSGrad by more than an order of magnitude on the synthetic tasks
   and TFGD by five accuracy points on Fashion-MNIST.
3. **Comparisons are paired and corrected.** At each evaluation seed every optimizer sees
   the bit-identical corrupted label set. Reported are the mean paired difference, a 95%
   bootstrap interval, a paired *t*-test and a Wilcoxon signed-rank test, all two-sided,
   with Holm–Bonferroni correction across the family of comparisons against the reference.

## Citation

```bibtex
@article{tfadamw,
  title  = {TF-AdamW: Tempered Fractional First-Moment Optimization with
            Adaptive Preconditioning for Robust Deep Learning},
  author = {Sallami, Mohamed Nidhal and Ben Fredj, Ouissem and
            Bouzida, Imed and Naifar, Omar},
  journal = {Machine Learning with Applications},
  note   = {Under review}
}
```
