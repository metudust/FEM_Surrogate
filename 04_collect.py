"""Step 4: read every completed ODB and build the training table data/dataset.csv.

Also writes data/curves.json (tip load-deflection history of each run) and
results/load_deflection.png, and flags runs that fail basic quality checks.

usage: python 04_collect.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import abq_runner as ar  # noqa: E402
import cantilever_inp as ci  # noqa: E402
import config  # noqa: E402

OUT_COLS = ["u2_tip_peak", "u2_tip_res", "u1_tip_peak", "peeq_max_peak", "mises_max_peak",
            "mises_max_res", "rf2_root_peak", "eq_err"]


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    with open(config.DATA / "doe.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    work = config.WORK
    sts = {r["run_id"]: ar.job_status(work, r["run_id"]) for r in rows}
    done = [j for j, s in sts.items() if s["status"] == "completed" and s["odb"]]
    if not done:
        raise SystemExit("no completed jobs in %s - run 03_run_doe.py first" % work)
    res = ar.extract(done, work)

    table, curves, n_qa = [], {}, 0
    for r in rows:
        job = r["run_id"]
        st, x = sts[job], res.get(job, {})
        p = {k: float(r[k]) for k in config.PARAMS}
        d = ci.derived(p, config.NU)
        row = dict(r)
        row.update({"status": st["status"], "wall_s": st["wall_s"], "solver_wall_s": st["solver_wall_s"]})
        if x.get("ok"):
            row.update({
                "u2_tip_peak": -x["u2_peak"],          # downward deflection, positive
                "u2_tip_res": -x.get("u2_res", float("nan")),
                "u1_tip_peak": x["u1_peak"],           # axial shortening from large rotation
                "peeq_max_peak": x.get("peeq_max_peak") or 0.0,
                "mises_max_peak": x.get("mises_max_peak"),
                "mises_max_res": x.get("mises_max_res"),
                "rf2_root_peak": x["rf2_root_peak"],
                "eq_err": abs(x["rf2_root_peak"] - d["P"]) / d["P"],
            })
            # quality checks: both steps reached the end, vertical equilibrium holds
            if (abs(x["load_frac_end"] - 1) > 1e-6 or abs(x.get("unload_frac_end", 0) - 1) > 1e-6
                    or row["eq_err"] > 5e-3):
                row["status"] = "qa_fail"
                n_qa += 1
            curves[job] = {"lf": p["load_factor"], "L": d["L"],
                           "load": x.get("curve_load", []), "unload": x.get("curve_unload", [])}
        elif st["status"] == "completed":
            row["status"] = "extract_fail"
        table.append(row)

    cols = list(rows[0].keys()) + ["status"] + OUT_COLS + ["wall_s", "solver_wall_s"]
    with open(config.DATA / "dataset.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for row in table:
            wr.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in row.items()})
    (config.DATA / "curves.json").write_text(json.dumps(curves))

    ok = [t for t in table if t["status"] == "completed"]
    print("dataset.csv: %d rows, %d usable, %d failed QA, %d not completed"
          % (len(table), len(ok), n_qa, len(table) - len(ok) - n_qa))
    if ok:
        pk = np.array([t["u2_tip_peak"] / float(t["delta_EB"]) for t in ok])
        print("peak deflection / Euler-Bernoulli: min %.3f  median %.3f  max %.3f"
              % (pk.min(), np.median(pk), pk.max()))
    plot_curves(curves)


def plot_curves(curves, n=12):
    if not curves:
        return
    rng = np.random.default_rng(0)
    keys = sorted(curves)
    pick = [keys[i] for i in sorted(rng.choice(len(keys), size=min(n, len(keys)), replace=False))]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in pick:
        c = curves[k]
        if not c["load"]:
            continue
        t, u = np.array(c["load"]).T
        line, = ax.plot(-u / c["L"], t * c["lf"], lw=1.2)
        if c["unload"]:
            t2, u2 = np.array(c["unload"]).T
            ax.plot(-u2 / c["L"], (1 - t2) * c["lf"], lw=1.2, ls="--", color=line.get_color())
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("tip deflection / L")
    ax.set_ylabel("load / first-yield load")
    ax.set_title("Abaqus load-deflection paths (solid: loading, dashed: unloading)")
    fig.tight_layout()
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.RESULTS / "load_deflection.png", dpi=150)
    print("wrote results/load_deflection.png")


if __name__ == "__main__":
    main()
