"""Step 6: measure, don't assume. CPU vs GPU for training and inference, vs Abaqus per design.

- Training: the same full-batch MLP training for a fixed number of epochs on each device,
  for the small production model and a wider one (to show where the GPU starts to pay off).
- Inference: designs per second for batch sizes from 1 to 1,000,000.
- Abaqus: median end-to-end and solver wall time per design from data/dataset.csv.

The numbers are whatever your machine produces. For a ~200-row dataset and a small MLP the
GPU can be slower than the CPU at training (launch overhead dominates); report that as found.

usage: python 06_benchmark.py [--features dimless] [--epochs 1000] [--quick]
Writes results/benchmark.md and results/benchmark.json
"""
import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch

import config
import surrogate_lib as sl


def abaqus_cost(path):
    if not Path(path).exists():
        return {}
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "completed"]
    out = {}
    for k in ("wall_s", "solver_wall_s"):
        v = [float(r[k]) for r in rows if r.get(k) not in (None, "", "None")]
        if v:
            out[k] = statistics.median(v)
    out["n"] = len(rows)
    return out


def time_inference(model, n_in, batch, device, min_seconds=0.3, max_reps=2000):
    x = torch.randn(batch, n_in, device=device)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        sl.sync(device)
        reps, t0 = 0, time.perf_counter()
        while True:
            model(x)
            reps += 1
            if reps >= max_reps or (reps % 5 == 0 and time.perf_counter() - t0 > min_seconds):
                break
        sl.sync(device)
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA / "dataset.csv"))
    ap.add_argument("--out", default=str(config.RESULTS))
    ap.add_argument("--features", choices=list(sl.FEATURES), default="dimless")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="smaller sizes, for a smoke test")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    hw = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
          "cpu": sl.device_name(torch.device("cpu")), "cpu_threads_used": torch.get_num_threads(),
          "gpu": sl.device_name(torch.device("cuda")) if torch.cuda.is_available() else "none"}
    print("hardware:", hw)

    d = sl.load_dataset(args.data)
    X, Y = sl.features(d, args.features), sl.targets(d, args.features)
    tr, va, _ = sl.split(len(X), 0)
    xs, ys = sl.Scaler.fit(X[tr]), sl.Scaler.fit(Y[tr])
    Xtr, Ytr = xs.fwd(X[tr]), ys.fwd(Y[tr])

    archs = [(128, 3)] if args.quick else [(128, 3), (512, 4)]
    train_rows = []
    for hidden, layers in archs:
        for dev in devices:
            sl.train(Xtr, Ytr, None, None, dev, epochs=20, hidden=hidden, layers=layers, eval_every=None)  # warm-up
            _, _, secs, _ = sl.train(Xtr, Ytr, None, None, dev, epochs=args.epochs, hidden=hidden,
                                     layers=layers, eval_every=None)
            train_rows.append({"model": "%dx%d" % (layers, hidden), "device": dev.type,
                               "seconds": secs, "epochs_per_s": args.epochs / secs})
            print("train %dx%d on %s: %.2f s for %d epochs" % (layers, hidden, dev.type, secs, args.epochs))

    ck = out / ("surrogate_%s.pt" % args.features)
    model_cpu = sl.make_mlp(X.shape[1], Y.shape[1])
    if ck.exists():
        state = torch.load(ck, map_location="cpu")
        model_cpu = sl.make_mlp(state["n_in"], state["n_out"], state["hidden"], state["layers"])
        model_cpu.load_state_dict(state["state_dict"])
    batches = [1, 1000, 100_000] if args.quick else [1, 1000, 100_000, 1_000_000]
    inf_rows = []
    for dev in devices:
        m = model_cpu.to(dev).eval()
        for b in batches:
            s = time_inference(m, X.shape[1], b, dev)
            inf_rows.append({"device": dev.type, "batch": b, "seconds_per_call": s, "designs_per_s": b / s})
            print("infer batch %8d on %s: %.3g ms per call, %.3g designs/s" % (b, dev.type, 1e3 * s, b / s))

    abq = abaqus_cost(args.data)
    best = max(inf_rows, key=lambda r: r["designs_per_s"])
    single = min((r for r in inf_rows if r["batch"] == 1), key=lambda r: r["seconds_per_call"])
    md = ["# Benchmark", "",
          "Hardware: %s, %s threads used; GPU: %s; PyTorch %s (CUDA %s)."
          % (hw["cpu"], hw["cpu_threads_used"], hw["gpu"], hw["torch"], hw["cuda"]), "",
          "## Training (%d epochs, full batch, %d designs)" % (args.epochs, len(tr)), "",
          "| model (layers x width) | device | seconds | epochs/s |", "|---|---|---|---|"]
    md += ["| %s | %s | %.2f | %.0f |" % (r["model"], r["device"], r["seconds"], r["epochs_per_s"]) for r in train_rows]
    md += ["", "## Inference", "", "| device | batch | ms per call | designs per second |", "|---|---|---|---|"]
    md += ["| %s | %d | %.3g | %.3g |" % (r["device"], r["batch"], 1e3 * r["seconds_per_call"], r["designs_per_s"])
           for r in inf_rows]
    md += ["", "## Against Abaqus", ""]
    if abq.get("wall_s"):
        md += ["- Abaqus, median per design over %d runs: %.1f s end to end (start-up, solve, files), %s s inside the solver."
               % (abq["n"], abq["wall_s"], abq.get("solver_wall_s", "n/a")),
               "- Surrogate, one design at a time (%s): %.3g ms, about %.0fx faster than one Abaqus run end to end."
               % (single["device"], 1e3 * single["seconds_per_call"], abq["wall_s"] / single["seconds_per_call"]),
               "- Surrogate, best batch (%s, %d designs): %.3g designs per second."
               % (best["device"], best["batch"], best["designs_per_s"]),
               "- The surrogate is only valid inside the sampled design space and is only as good as the "
               "test-set error in metrics_%s.json." % args.features]
    else:
        md += ["- No Abaqus timings found in %s." % args.data]
    (out / "benchmark.md").write_text("\n".join(md) + "\n")
    (out / "benchmark.json").write_text(json.dumps({"hardware": hw, "training": train_rows,
                                                    "inference": inf_rows, "abaqus": abq}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
