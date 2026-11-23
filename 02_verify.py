"""Step 2: check the model before generating any training data.

A. Linear-elastic runs vs beam theory (Euler-Bernoulli and Timoshenko) for slenderness
   L/h = 5, 10, 20 on several meshes. Expect FE close to Timoshenko; the gap to
   Euler-Bernoulli should shrink as the beam gets more slender.
B. Mesh convergence of the full nonlinear model (plasticity + NLGEOM + unloading)
   for three designs, compared with the finest mesh that fits under MAX_NODES.

usage: python 02_verify.py [--workers 1] [--cpus 1]
Writes results/verification.md and results/verification.json.
"""
import argparse
import json

import abq_runner as ar
import cantilever_inp as ci
import config

MESHES = [(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)]
BASE = {"h": 20.0, "b": 10.0, "E": 200e3}
ELASTIC_P = 1000.0  # N; linear analysis, so the value only scales the answer
NL_DESIGNS = {      # job names must not contain dots
    "A": {"slenderness": 10.0, "load_factor": 0.9, "sy": 300.0, "hard_ratio": 0.01},   # stays elastic
    "B": {"slenderness": 10.0, "load_factor": 1.3, "sy": 300.0, "hard_ratio": 0.01},   # plastic
    "C": {"slenderness": 5.0, "load_factor": 1.3, "sy": 300.0, "hard_ratio": 0.005},   # short, low hardening
}


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=1)
    args = ap.parse_args()

    work = ar.check_workdir(config.WORK)
    meshes = [m for m in MESHES if ci.node_count(*m) <= config.MAX_NODES]
    print("meshes under %d nodes: %s" % (config.MAX_NODES, meshes))

    jobs, meta = [], {}
    for S in (5, 10, 20):
        for nx, ny in meshes:
            job = "ver_el_S%02d_m%dx%d" % (S, nx, ny)
            p = dict(BASE, slenderness=float(S), P=ELASTIC_P)
            meta[job] = ("elastic", S, (nx, ny), ci.write_inp(work / (job + ".inp"), p, nx, ny, config.NU,
                                                              plastic=False, nlgeom=False, unload=False, title=job))
            jobs.append(job)
    for key, d in NL_DESIGNS.items():
        for nx, ny in meshes[1:]:
            job = "ver_nl_%s_m%dx%d" % (key, nx, ny)
            p = dict(BASE, **d)
            meta[job] = ("nonlinear", key, (nx, ny), ci.write_inp(work / (job + ".inp"), p, nx, ny, config.NU, title=job))
            jobs.append(job)

    status = ar.run_many(jobs, work, args.workers, args.cpus)
    done = [j for j in jobs if status[j]["status"] == "completed"]
    res = ar.extract(done, work) if done else {}

    md = ["# Verification", "", "## A. Linear elastic vs beam theory", "",
          "| L/h | mesh | nodes | FE tip deflection (mm) | Euler-Bernoulli | Timoshenko | FE vs EB | FE vs Timoshenko |",
          "|---|---|---|---|---|---|---|---|"]
    table = {"elastic": [], "nonlinear": []}
    for job in jobs:
        kind, key, mesh, d = meta[job]
        r = res.get(job, {})
        if kind != "elastic":
            continue
        if not r.get("ok"):
            md.append("| %s | %dx%d | %d | FAILED | | | | |" % (key, mesh[0], mesh[1], d["nodes"]))
            continue
        fe = -r["u2_peak"]
        row = {"S": key, "mesh": mesh, "nodes": d["nodes"], "fe": fe, "eb": d["delta_EB"], "timo": d["delta_Timo"]}
        table["elastic"].append(row)
        md.append("| %d | %dx%d | %d | %.5f | %.5f | %.5f | %+.2f%% | %+.2f%% |"
                  % (key, mesh[0], mesh[1], d["nodes"], fe, d["delta_EB"], d["delta_Timo"],
                     pct(fe, d["delta_EB"]), pct(fe, d["delta_Timo"])))

    md += ["", "## B. Mesh convergence, nonlinear model (vs finest mesh)", "",
           "| design | mesh | nodes | peak deflection (mm) | residual (mm) | max PEEQ | peak vs finest | residual vs finest |",
           "|---|---|---|---|---|---|---|---|"]
    for key in NL_DESIGNS:
        rows = []
        for job in jobs:
            kind, k, mesh, d = meta[job]
            r = res.get(job, {})
            if kind == "nonlinear" and k == key and r.get("ok"):
                rows.append((mesh, d["nodes"], -r["u2_peak"], -r.get("u2_res", 0.0), r.get("peeq_max_peak") or 0.0))
        if not rows:
            md.append("| %s | all failed | | | | | | |" % key)
            continue
        ref = rows[-1]
        for mesh, nodes, pk, rs, pe in rows:
            dres = "n/a" if abs(ref[3]) < 1e-9 else "%+.2f%%" % pct(rs, ref[3])
            md.append("| %s | %dx%d | %d | %.5f | %.5f | %.4g | %+.2f%% | %s |"
                      % (key, mesh[0], mesh[1], nodes, pk, rs, pe, pct(pk, ref[2]), dres))
            table["nonlinear"].append({"design": key, "mesh": mesh, "nodes": nodes, "peak": pk, "res": rs, "peeq": pe})

    md += ["", "Designs: " + "; ".join("%s = %s" % (k, v) for k, v in NL_DESIGNS.items()),
           "Base section: %s. Elastic load P = %g N." % (BASE, ELASTIC_P)]
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "verification.md").write_text("\n".join(md) + "\n")
    (config.RESULTS / "verification.json").write_text(json.dumps(table, indent=1))
    print("\n".join(md))
    print("\nwrote results/verification.md")


if __name__ == "__main__":
    main()
