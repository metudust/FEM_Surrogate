"""Shared PyTorch code for 05_train_surrogate.py and 06_benchmark.py.

Two ways to describe a design to the network:

  dimless  : 4 dimensionless groups  L/h, P/P_yield, sy/E, Et/E.
             In plane stress with P set as a multiple of P_yield, the thickness b cancels
             exactly and h only sets the length scale, so these 4 numbers determine the
             normalized response. Targets are normalized by beam-theory quantities.
  physical : the 7 raw inputs (log-scaled) and raw outputs. The network has to discover
             the scaling itself. Comparing the two shows what dimensional analysis buys.
"""
import csv
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUTS = ["h", "slenderness", "b", "E", "sy", "hard_ratio", "load_factor", "L", "P", "delta_EB", "delta_Timo"]
OUTPUTS = ["u2_tip_peak", "u2_tip_res", "peeq_max_peak"]
TARGET_NAMES = ["peak tip deflection (mm)", "residual tip deflection (mm)", "max equivalent plastic strain"]
FEATURES = {
    "dimless": ["L/h", "P/P_yield", "sy/E", "Et/E"],
    "physical": ["log h", "log L", "log b", "log E", "log sy", "Et/E", "log P"],
}


def load_dataset(path):
    """Columns of all usable rows (status == completed) as float arrays."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status", "completed") == "completed"]
    if not rows:
        raise SystemExit("no usable rows in %s" % path)
    return {k: np.array([float(r[k]) for r in rows]) for k in INPUTS + OUTPUTS}


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


def features(d, kind):
    if kind == "dimless":
        cols = [d["slenderness"], d["load_factor"], d["sy"] / d["E"], d["hard_ratio"]]
    else:
        cols = [np.log(d["h"]), np.log(d["L"]), np.log(d["b"]), np.log(d["E"]), np.log(d["sy"]),
                d["hard_ratio"], np.log(d["P"])]
    return np.stack(cols, 1)


def targets(d, kind):
    if kind == "dimless":
        cols = [d["u2_tip_peak"] / d["delta_EB"],          # amplification over linear Euler-Bernoulli
                d["u2_tip_res"] / d["delta_EB"],           # permanent set, same scale
                d["peeq_max_peak"] / (d["sy"] / d["E"])]   # plastic strain in yield strains
    else:
        cols = [np.log(d["u2_tip_peak"]), d["u2_tip_res"], d["peeq_max_peak"]]
    return np.stack(cols, 1)


def to_physical(Y, d, kind):
    """Invert targets(): network outputs -> mm, mm, strain."""
    if kind == "dimless":
        return np.stack([Y[:, 0] * d["delta_EB"], Y[:, 1] * d["delta_EB"], Y[:, 2] * d["sy"] / d["E"]], 1)
    return np.stack([np.exp(Y[:, 0]), Y[:, 1], Y[:, 2]], 1)


def split(n, seed, train=0.7, val=0.15):
    idx = np.random.default_rng(seed).permutation(n)
    a = int(round(train * n))
    b = a + int(round(val * n))
    return idx[:a], idx[a:b], idx[b:]


class Scaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    @classmethod
    def fit(cls, a):
        s = a.std(0)
        return cls(a.mean(0), np.where(s > 0, s, 1.0))

    def fwd(self, a):
        return (a - self.mean) / self.std

    def inv(self, a):
        return a * self.std + self.mean


def make_mlp(n_in, n_out, hidden=128, layers=3):
    mods, w = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(w, hidden), nn.SiLU()]
        w = hidden
    mods.append(nn.Linear(w, n_out))
    return nn.Sequential(*mods)


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False "
                         "(CPU-only PyTorch build or no NVIDIA driver)")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    import platform
    return "CPU: %s (%s logical cores)" % (platform.processor() or platform.machine(), __import__("os").cpu_count())


def train(Xtr, Ytr, Xva, Yva, device, epochs=5000, lr=2e-3, hidden=128, layers=3, seed=0,
          eval_every=10, patience=800):
    """Full-batch Adam with cosine learning-rate decay; keeps the best validation state.

    eval_every=None turns off validation (used for clean timing in the benchmark).
    Returns (model, history, train_seconds, best_epoch).
    """
    torch.manual_seed(seed)
    model = make_mlp(Xtr.shape[1], Ytr.shape[1], hidden, layers).to(device)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    xt, yt = t(Xtr), t(Ytr)
    xv, yv = (t(Xva), t(Yva)) if eval_every else (None, None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    best_loss, best_state, best_ep, hist = math.inf, None, 0, []

    sync(device)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xt), yt)
        loss.backward()
        opt.step()
        sched.step()
        if eval_every and (ep % eval_every == 0 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                vl = F.mse_loss(model(xv), yv).item()
            hist.append((ep, loss.item(), vl))
            if vl < best_loss:
                best_loss, best_ep = vl, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif ep - best_ep > patience:
                break
    sync(device)
    seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, seconds, best_ep


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def r2(y, p):
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")
