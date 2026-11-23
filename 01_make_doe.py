"""Step 1: sample the design space with a Latin hypercube and write data/doe.csv.

usage: python 01_make_doe.py [--n 300] [--seed 2026]
"""
import argparse
import csv
import json

import numpy as np

import cantilever_inp as ci
import config


def latin_hypercube(n, dims, rng):
    """One sample per stratum in every dimension, strata paired at random."""
    u = np.empty((n, dims))
    for k in range(dims):
        u[:, k] = (rng.permutation(n) + rng.random(n)) / n
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=config.N_SAMPLES)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    nodes = ci.node_count(**config.MESH)
    if nodes > config.MAX_NODES:
        raise SystemExit("mesh %s has %d nodes > MAX_NODES=%d" % (config.MESH, nodes, config.MAX_NODES))

    names = list(config.PARAMS)
    lo = np.array([config.PARAMS[k][0] for k in names])
    hi = np.array([config.PARAMS[k][1] for k in names])
    rng = np.random.default_rng(args.seed)
    X = lo + latin_hypercube(args.n, len(names), rng) * (hi - lo)

    config.DATA.mkdir(parents=True, exist_ok=True)
    out = config.DATA / "doe.csv"
    cols = ["run_id"] + names + ["L", "P_yield", "P", "delta_EB", "delta_Timo"]
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for i, x in enumerate(X, start=1):
            p = dict(zip(names, x))
            d = ci.derived(p, config.NU)
            wr.writerow(["run_%04d" % i] + ["%.6g" % v for v in x] +
                        ["%.6g" % d[k] for k in cols[len(names) + 1:]])

    meta = {"n": args.n, "seed": args.seed, "params": config.PARAMS, "mesh": config.MESH,
            "nodes_per_model": nodes, "nu": config.NU}
    (config.DATA / "doe_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote %s: %d designs, %d inputs, %d nodes per model" % (out, args.n, len(names), nodes))


if __name__ == "__main__":
    main()
