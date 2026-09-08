"""Run a list of run_seq.py jobs with a fixed number of single-threaded workers.

    python scripts/queue_runs.py jobs.txt --workers 8

jobs.txt: one job per line, "<config> <seed> [extra run_seq flags...]"; blank lines and '#'
comments are skipped.  Every job is its own process (OMP/MKL threads = 1), logs go to
tmp/queue_logs/<config>_s<seed>[_flags].log, and one line per job is printed on completion:
DONE / FAIL / SKIP (checkpoint already present), so a log tail shows the queue's state.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "tmp", "queue_logs")


def job_name(config, seed, flags):
    tag = "".join(f.replace("--", "_").replace(" ", "") for f in flags)
    return f"{config}_s{seed}{tag}"


def part_exists(config, seed, flags):
    val = "--val" in flags
    vs = None
    if "--val-seed" in flags:
        vs = flags[flags.index("--val-seed") + 1]
    prefix = "" if not val else ("val_" if vs in (None, "1234") else f"val{vs}_")
    return os.path.exists(os.path.join(ROOT, "results", "bio", "parts" if False else "seq", "parts",
                                       f"{prefix}{config}_s{seed}.pkl"))


def run_one(py, config, seed, flags):
    name = job_name(config, seed, flags)
    if part_exists(config, seed, flags):
        return "SKIP", name, 0.0
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    cmd = [py, os.path.join(ROOT, "src", "run_seq.py"), "--configs", config, "--seeds", str(seed),
           "--resume", *flags]
    t0 = time.time()
    with open(os.path.join(LOGS, name + ".log"), "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and part_exists(config, seed, flags)
    return ("DONE" if ok else "FAIL"), name, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    jobs = []
    for line in open(args.jobs, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        jobs.append((parts[0], int(parts[1]), parts[2:]))
    print(f"queue: {len(jobs)} jobs, {args.workers} workers, python {args.python}", flush=True)
    n_done = n_fail = n_skip = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.python, c, s, f) for c, s, f in jobs]
        for fut in as_completed(futs):
            status, name, mins = fut.result()
            n_done += status == "DONE"; n_fail += status == "FAIL"; n_skip += status == "SKIP"
            print(f"{status} {name} {mins:.1f} min  [{n_done + n_fail + n_skip}/{len(jobs)} "
                  f"done={n_done} fail={n_fail} skip={n_skip} {(time.time() - t0) / 60:.0f} min]", flush=True)
    print(f"QUEUE FINISHED done={n_done} fail={n_fail} skip={n_skip} in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
