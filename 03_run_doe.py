"""Step 3: write one .inp per DOE row and run them in Abaqus/Standard.

Re-running skips jobs that already completed (use --force to redo them), so you can stop
and resume. Start with --limit 5 to check timing before launching all 300.

usage: python 03_run_doe.py [--limit N] [--workers 1] [--cpus 1] [--clean] [--force] [--dry-run]
"""
import argparse
import csv

import abq_runner as ar
import cantilever_inp as ci
import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N designs")
    ap.add_argument("--workers", type=int, default=1, help="jobs at the same time (check your license)")
    ap.add_argument("--cpus", type=int, default=1, help="cores per job")
    ap.add_argument("--clean", action="store_true", help="delete Abaqus scratch files after each job")
    ap.add_argument("--force", action="store_true", help="re-run jobs that already completed")
    ap.add_argument("--dry-run", action="store_true", help="write .inp files only")
    args = ap.parse_args()

    if ci.node_count(**config.MESH) > config.MAX_NODES:
        raise SystemExit("mesh exceeds MAX_NODES; change config.MESH")
    with open(config.DATA / "doe.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]

    work = ar.check_workdir(config.WORK)
    todo = []
    for r in rows:
        job = r["run_id"]
        p = {k: float(r[k]) for k in config.PARAMS}
        ci.write_inp(work / (job + ".inp"), p, nu=config.NU, title=job, **config.MESH)
        if args.force or ar.job_status(work, job)["status"] != "completed":
            todo.append(job)
    print("%d designs, %d to run, work dir %s" % (len(rows), len(todo), work))
    if args.dry_run or not todo:
        return

    ar.run_many(todo, work, args.workers, args.cpus, cleanup=args.clean)

    sts = [ar.job_status(work, r["run_id"]) for r in rows]
    with open(config.DATA / "runs.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["job", "status", "wall_s", "solver_wall_s", "odb"])
        wr.writeheader()
        wr.writerows(sts)
    ok = [s for s in sts if s["status"] == "completed"]
    walls = sorted(s["wall_s"] for s in ok if s["wall_s"])
    print("completed %d / %d" % (len(ok), len(sts)))
    if walls:
        print("end-to-end wall time per job: median %.1f s, total %.1f min"
              % (walls[len(walls) // 2], sum(walls) / 60.0))
    print("next: python 04_collect.py")


if __name__ == "__main__":
    main()
