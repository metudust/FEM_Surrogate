"""Shared settings for Stage 1: Abaqus cantilever -> PyTorch surrogate on a GPU.

Units throughout: mm, N, MPa (consistent set; stresses in MPa, forces in N).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"          # DOE, run log, extracted dataset (small, commit these)
RESULTS = ROOT / "results"    # verification tables, trained models, plots, benchmark
# Abaqus scratch (.inp, .odb, .sta, ...). Abaqus can be fussy about spaces in paths,
# so point ABQ_WORK_DIR at something like C:\abq_work if runs fail to start.
WORK = Path(os.environ.get("ABQ_WORK_DIR", str(ROOT / "work")))

# Abaqus command. Full installs use "abaqus"; the Learning Edition may install a
# versioned command instead. Set ABAQUS_CMD in the environment to override.
ABAQUS_CMD = os.environ.get("ABAQUS_CMD", "abaqus")

# Mesh: nx x ny 8-node plane-stress quads (CPS8R).
# nodes = (2nx+1)(2ny+1) - nx*ny  ->  40 x 4 gives 569 nodes.
MESH = {"nx": 40, "ny": 4}
# The Abaqus Learning Edition is limited to 1,000 nodes. Raise this with a full license.
MAX_NODES = int(os.environ.get("ABQ_MAX_NODES", "1000"))

NU = 0.3  # Poisson's ratio, held fixed

# Design space sampled by 01_make_doe.py (Latin hypercube, uniform in each range).
#   L = slenderness * h
#   P = load_factor * P_yield,   P_yield = sy * b * h^2 / (6 L)   (first yield at the root)
#   A rectangular section reaches its plastic limit moment at load_factor = 1.5,
#   so 1.35 keeps every run below collapse while still producing large plastic zones.
PARAMS = {
    "h":           (10.0, 40.0),     # section height [mm]
    "slenderness": (5.0, 20.0),      # L / h [-]
    "b":           (5.0, 30.0),      # out-of-plane thickness [mm]
    "E":           (70e3, 210e3),    # Young's modulus [MPa]
    "sy":          (150.0, 500.0),   # initial yield stress [MPa]
    "hard_ratio":  (0.005, 0.05),    # tangent modulus / E, linear hardening [-]
    "load_factor": (0.3, 1.35),      # P / P_yield [-]
}
N_SAMPLES = 300
SEED = 2026
