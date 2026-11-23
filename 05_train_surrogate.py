"""Step 5: train an MLP surrogate of the Abaqus results (on the GPU when one is available).

Predicts peak tip deflection, residual tip deflection after unloading, and the maximum
equivalent plastic strain. Accuracy is reported on a held-out test set that the model never
saw, next to the error of the linear hand calculation (Euler-Bernoulli / Timoshenko).

usage: python 05_train_surrogate.py [--features dimless|physical] [--device auto|cpu|cuda]
                                    [--epochs 5000] [--hidden 128] [--layers 3]
Writes results/surrogate_<features>.pt, metrics_<features>.json, parity_<features>.png,
training_<features>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
import surrogate_lib as sl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA / "dataset.csv"))
    ap.add_argument("--out", default=str(config.RESULTS))
    ap.add_argument("--features", choices=list(sl.FEATURES), default="dimless")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kind = args.features

    d = sl.load_dataset(args.data)
    X, Y = sl.features(d, kind), sl.targets(d, kind)
    tr, va, te = sl.split(len(X), args.seed)
    xs, ys = sl.Scaler.fit(X[tr]), sl.Scaler.fit(Y[tr])
    device = sl.pick_device(args.device)
    print("%d designs (train %d / val %d / test %d), features=%s, device=%s"
          % (len(X), len(tr), len(va), len(te), kind, sl.device_name(device)))

    model, hist, secs, best_ep = sl.train(xs.fwd(X[tr]), ys.fwd(Y[tr]), xs.fwd(X[va]), ys.fwd(Y[va]), device,
                                          epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                                          layers=args.layers, seed=args.seed)
    dt = sl.subset(d, te)
    pred = sl.to_physical(ys.inv(sl.predict(model, xs.fwd(X[te]), device)), dt, kind)
    true = np.stack([dt[k] for k in sl.OUTPUTS], 1)

    pk_t, pk_p = true[:, 0], pred[:, 0]
    ape = 100 * np.abs(pk_p - pk_t) / pk_t
    eb_ape = 100 * np.abs(dt["delta_EB"] - pk_t) / pk_t
    m = {
        "features": kind, "device": sl.device_name(device), "torch": torch.__version__,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "hidden": args.hidden, "layers": args.layers, "epochs_run": int(hist[-1][0] + 1), "best_epoch": int(best_ep),
        "train_seconds": round(secs, 2),
        "peak_deflection": {"r2": sl.r2(pk_t, pk_p), "mae_mm": float(np.mean(np.abs(pk_p - pk_t))),
                            "mape_pct": float(ape.mean()), "max_ape_pct": float(ape.max())},
        "residual_deflection": {"r2": sl.r2(true[:, 1], pred[:, 1]),
                                "mae_mm": float(np.mean(np.abs(pred[:, 1] - true[:, 1]))),
                                "max_abs_err_mm": float(np.max(np.abs(pred[:, 1] - true[:, 1]))),
                                "mae_pct_of_mean_peak": float(100 * np.mean(np.abs(pred[:, 1] - true[:, 1])) / pk_t.mean())},
        "peeq_max": {"r2": sl.r2(true[:, 2], pred[:, 2]), "mae": float(np.mean(np.abs(pred[:, 2] - true[:, 2])))},
        "hand_calc_peak_deflection": {
            "euler_bernoulli_mape_pct": float(eb_ape.mean()), "euler_bernoulli_max_ape_pct": float(eb_ape.max()),
            "timoshenko_mape_pct": float((100 * np.abs(dt["delta_Timo"] - pk_t) / pk_t).mean())},
    }

    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "features": kind,
                "hidden": args.hidden, "layers": args.layers, "n_in": X.shape[1], "n_out": Y.shape[1],
                "x_mean": xs.mean.tolist(), "x_std": xs.std.tolist(),
                "y_mean": ys.mean.tolist(), "y_std": ys.std.tolist()}, out / ("surrogate_%s.pt" % kind))
    (out / ("metrics_%s.json" % kind)).write_text(json.dumps(m, indent=2))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axs):
        ax.scatter(true[:, i], pred[:, i], s=12, alpha=0.8)
        lo, hi = min(true[:, i].min(), pred[:, i].min()), max(true[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("Abaqus")
        ax.set_ylabel("surrogate")
        ax.set_title("%s\nR2 = %.4f" % (sl.TARGET_NAMES[i], sl.r2(true[:, i], pred[:, i])), fontsize=10)
    fig.suptitle("Held-out test set (%d designs), features: %s" % (len(te), kind))
    fig.tight_layout()
    fig.savefig(out / ("parity_%s.png" % kind), dpi=150)

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(h[:, 0], h[:, 1], label="train")
    ax.semilogy(h[:, 0], h[:, 2], label="validation")
    ax.axvline(best_ep, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardized targets)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / ("training_%s.png" % kind), dpi=150)

    p, r = m["peak_deflection"], m["hand_calc_peak_deflection"]
    print("trained in %.1f s (best epoch %d)" % (secs, best_ep))
    print("peak deflection : R2 %.4f, mean error %.2f%%, worst %.2f%%  (Euler-Bernoulli: mean %.1f%%, worst %.1f%%)"
          % (p["r2"], p["mape_pct"], p["max_ape_pct"], r["euler_bernoulli_mape_pct"], r["euler_bernoulli_max_ape_pct"]))
    print("residual defl.  : R2 %.4f, MAE %.4g mm" % (m["residual_deflection"]["r2"], m["residual_deflection"]["mae_mm"]))
    print("max PEEQ        : R2 %.4f" % m["peeq_max"]["r2"])
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
