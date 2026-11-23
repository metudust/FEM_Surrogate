# Abaqus to PyTorch surrogate, Stage 1: elastic-plastic cantilever

Generate a few hundred nonlinear Abaqus/Standard analyses of a cantilever with a tip load,
train a neural-network surrogate on them on an NVIDIA GPU, and measure what the surrogate
buys you in accuracy and speed. Every number this project reports comes from your own runs.

| step | script | what it does | needs |
|---|---|---|---|
| 1 | `01_make_doe.py` | Latin-hypercube sample of 7 design variables -> `data/doe.csv` | Python |
| 2 | `02_verify.py` | beam-theory check (elastic) and mesh convergence (nonlinear) -> `results/verification.md` | Abaqus |
| 3 | `03_run_doe.py` | writes one `.inp` per design and runs them; resumable | Abaqus |
| 4 | `04_collect.py` | reads the ODBs, checks equilibrium -> `data/dataset.csv`, load-deflection plot | Abaqus Python |
| 5 | `05_train_surrogate.py` | trains the MLP surrogate, test-set accuracy, parity plots | PyTorch (GPU) |
| 6 | `06_benchmark.py` | CPU vs GPU training and inference; surrogate vs Abaqus per design | PyTorch (GPU) |

Support files: `config.py` (all settings), `cantilever_inp.py` (input-file writer),
`abq_runner.py` (runs jobs, reads status and timing), `abq_extract.py` (runs inside Abaqus Python),
`surrogate_lib.py` (shared PyTorch code), `tests/smoke_test.py` (offline checks, no Abaqus needed).

## The model

- 2-D plane stress, length L, height h, thickness b; 8-node quads (CPS8R), 40 x 4 elements, 569 nodes.
- Root: U1 = 0 on the whole root edge and U2 = 0 at mid-height. This clamps the root section's
  rotation without the stress singularity you get at the corners when U2 is fixed along the whole edge.
- Tip load P (downward) as consistent nodal forces for a uniform shear traction on the tip edge.
- Mises plasticity with linear isotropic hardening (tangent modulus Et).
- Step LOAD ramps P with NLGEOM; step UNLOAD removes it. Outputs: peak tip deflection,
  residual (permanent) tip deflection, max equivalent plastic strain (PEEQ), max Mises stress,
  root reaction (equilibrium check) and the full tip load-deflection history.
- Units: mm, N, MPa.

## Design space and why only 4 numbers matter

`config.py` samples h, L/h, b, E, yield stress sy, hardening ratio Et/E and load factor
P / P_yield, where P_yield = sy b h^2 / (6 L) is the first-yield load. The upper load factor of
1.35 stays below the plastic collapse of a rectangular section (1.5).

In plane stress with P given as a multiple of P_yield, the thickness b cancels exactly and h only
sets the length scale. The normalized response therefore depends on four groups:
L/h, P/P_yield, sy/E and Et/E. `05_train_surrogate.py --features dimless` (default) uses them, with
targets normalized by the Euler-Bernoulli deflection; `--features physical` feeds the raw inputs
instead. Comparing the two shows what dimensional analysis is worth on a small dataset.

## Setup

**Abaqus.** The free Abaqus Learning Edition (non-commercial use, 1,000-node limit) is enough;
the default mesh has 569 nodes. A full license works too (raise `ABQ_MAX_NODES` for finer meshes).
Open an *Abaqus Command* prompt and confirm the command name. If it is not `abaqus`, set it:

```powershell
$env:ABAQUS_CMD = "abaqus"   # or the Learning Edition command on your machine
```

Abaqus can be fussy about spaces in paths, and this folder has some. If jobs fail to start:

```powershell
$env:ABQ_WORK_DIR = "C:\abq_work"
```

**Python** (3.10 or newer) in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib
```

Then install PyTorch **with CUDA** using the command that
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) gives for Windows + pip + CUDA.
A plain `pip install torch` may give you a CPU-only build. Check the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Optional offline check (no Abaqus): `python tests/smoke_test.py`. It trains on a **toy** dataset
made from an invented formula; its numbers only show the code runs and must never be reported.

## Run

```powershell
python 01_make_doe.py                 # 300 designs (data/doe.csv)
python 02_verify.py                   # ~27 short runs; read results/verification.md
python 03_run_doe.py --limit 5        # time a few runs first
python 03_run_doe.py --clean          # then all of them; re-run to resume after a stop
python 04_collect.py                  # data/dataset.csv + results/load_deflection.png
python 05_train_surrogate.py          # dimensionless features, GPU if available
python 05_train_surrogate.py --features physical
python 06_benchmark.py                # results/benchmark.md
```

`03_run_doe.py --workers 2` runs two jobs at once if your license allows it.

## What to check before you trust anything

1. **Verification (step 2).** In the elastic runs the FE tip deflection should sit close to
   Timoshenko and approach Euler-Bernoulli as L/h grows. In the nonlinear runs, peak and residual
   deflection should change little between the two finest meshes. If they don't, refine
   `config.MESH` (a full license lifts the node limit) before generating data.
2. **Data quality (step 4).** Runs where a step did not finish or vertical equilibrium is off by
   more than 0.5% are marked `qa_fail` and left out of training. Find out why before you drop them.
3. **Accuracy (step 5).** Judge the surrogate only on the held-out test set, and next to the
   linear hand calculation in `metrics_*.json`: the surrogate earns its place where the beam yields.
4. **Speed (step 6).** Abaqus time per design is dominated by start-up, so the end-to-end speedup
   looks huge. Report the solver time as well, and report it if the GPU is *slower* than the CPU
   for this small network. That is a normal result at this size and says something about when GPUs help.

## Reporting it

Fill this from `results/` after the runs. Never use the smoke-test numbers.

| quantity | value |
|---|---|
| Abaqus designs run / used | |
| Abaqus time per design (end to end / solver) | |
| Surrogate test error, peak deflection (mean / worst) | |
| Linear hand calculation error on the same designs (mean / worst) | |
| Surrogate time per design (batch 1) and throughput (best batch, GPU) | |
| GPU vs CPU training time (small and wide network) | |

Resume line in the Google XYZ form, numbers from your own runs:
*Built a GPU-trained PyTorch surrogate of nonlinear Abaqus cantilever analyses that predicts peak
deflection within [X]% (held-out test set) at [Y] designs per second, by scripting [N] parametric
Abaqus/Standard runs and benchmarking CPU vs GPU training and inference.*

## Limits

2-D plane stress; monotonic load then unload; linear hardening; one geometry family; the surrogate
is only valid inside the sampled ranges. The Learning Edition caps mesh refinement at 1,000 nodes.

## Licensing and data

The Learning Edition is for non-commercial use. If you use an employer's Abaqus license for this
public project, get permission first, and keep customer models and data out of the repository.

## Next: Stage 2

Additive-manufacturing distortion of a cantilever built on supports and cut free, with
layer-by-layer element activation (inherent-strain method), using the same pipeline:
DOE -> Abaqus -> dataset -> GPU surrogate -> benchmark.
