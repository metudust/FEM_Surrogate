"""Run Abaqus jobs, read their status and timing, and call the ODB extractor.

Used by 02_verify.py, 03_run_doe.py and 04_collect.py.
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import config

EXTRACTOR = Path(__file__).resolve().parent / "abq_extract.py"
SCRATCH_EXT = (".com", ".prt", ".sim", ".mdl", ".stt", ".res", ".abq", ".pac", ".sel", ".023", ".ipm")


def check_workdir(workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    s = str(workdir)
    if " " in s or not s.isascii():
        print("NOTE: work directory has spaces or non-ASCII characters:\n  %s\n"
              "  If Abaqus fails to start, set ABQ_WORK_DIR to a simple path such as C:\\abq_work" % s)
    return workdir


def abaqus(args, cwd, timeout=None):
    """Run '<ABAQUS_CMD> <args>' in cwd. Returns (CompletedProcess, wall seconds)."""
    cmd = "%s %s" % (config.ABAQUS_CMD, args)
    t0 = time.perf_counter()
    # shell=True because the Abaqus launcher is a .bat file on Windows
    r = subprocess.run(cmd, cwd=str(cwd), shell=True, capture_output=True, text=True, timeout=timeout)
    return r, time.perf_counter() - t0


def job_status(workdir, job):
    """Status from the .sta/.dat files plus the solver's own wall-clock time."""
    w = Path(workdir)
    sta, msg, dat = (w / ("%s.%s" % (job, e)) for e in ("sta", "msg", "dat"))
    status = "not_run"
    if sta.exists():
        t = sta.read_text(errors="replace")
        if "COMPLETED SUCCESSFULLY" in t:
            status = "completed"
        elif "NOT BEEN COMPLETED" in t:
            status = "failed"
        else:
            status = "incomplete"
    elif dat.exists():
        status = "failed"  # input errors stop the job before the .sta file is written
    solver_wall = None
    for f in (msg, dat):
        if f.exists():
            m = re.findall(r"WALLCLOCK TIME \(SEC\)\s*=\s*([\d.]+)", f.read_text(errors="replace"))
            if m:
                solver_wall = float(m[-1])
                break
    timing = w / ("%s.timing.json" % job)
    wall = json.loads(timing.read_text()).get("wall_s") if timing.exists() else None
    odb = w / ("%s.odb" % job)
    return {"job": job, "status": status, "wall_s": wall, "solver_wall_s": solver_wall,
            "odb": odb.name if odb.exists() else ""}


def run_job(job, workdir, cpus=1, timeout=1800):
    """Run <job>.inp in workdir; record end-to-end wall time next to the results."""
    w = Path(workdir)
    for old in (w / ("%s.lck" % job),):
        if old.exists():
            old.unlink()
    try:
        r, wall = abaqus("job=%s input=%s.inp cpus=%d interactive ask_delete=OFF" % (job, job, cpus), w, timeout)
        rc, err = r.returncode, (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        wall, rc, err = float(timeout), -1, "timeout"
    (w / ("%s.timing.json" % job)).write_text(json.dumps({"wall_s": round(wall, 3), "returncode": rc}))
    st = job_status(w, job)
    if st["status"] != "completed":
        st["log_tail"] = err
    return st


def run_many(jobs, workdir, workers=1, cpus=1, cleanup=False, timeout=1800):
    """Run jobs (optionally several at once, each on `cpus` cores) and print progress."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out, n = {}, len(jobs)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(run_job, j, workdir, cpus, timeout): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), start=1):
            st = fut.result()
            out[st["job"]] = st
            print("[%d/%d] %s %s  wall %.1f s  solver %s s"
                  % (k, n, st["job"], st["status"], st["wall_s"] or 0, st["solver_wall_s"]))
            if st["status"] != "completed" and st.get("log_tail"):
                print("    last output:", st["log_tail"].strip().splitlines()[-3:])
            if cleanup:
                clean(workdir, st["job"])
    return out


def clean(workdir, job):
    """Delete Abaqus scratch files for one job (keeps .inp .odb .sta .msg .dat .log)."""
    for ext in SCRATCH_EXT:
        f = Path(workdir) / (job + ext)
        if f.exists():
            f.unlink()


def extract(jobs, workdir, timeout=3600):
    """Read ODB results for a list of jobs with one Abaqus Python session."""
    w = Path(workdir)
    shutil.copy(EXTRACTOR, w / EXTRACTOR.name)   # run it from the work dir: no paths with spaces
    lst = w / "extract_list.txt"
    out = w / "extract_out.json"
    lst.write_text("".join("%s\t%s.odb\n" % (j, j) for j in jobs))
    if out.exists():
        out.unlink()
    r, wall = abaqus("python %s %s %s" % (EXTRACTOR.name, lst.name, out.name), w, timeout)
    if not out.exists():
        raise RuntimeError("ODB extraction failed (%s python). Output:\n%s"
                           % (config.ABAQUS_CMD, (r.stdout + r.stderr)[-3000:]))
    print("extracted %d ODBs in %.1f s" % (len(jobs), wall))
    return json.loads(out.read_text())
