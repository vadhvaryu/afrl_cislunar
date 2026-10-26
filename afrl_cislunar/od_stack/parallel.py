"""
Run many independent stack cases on all cores, streaming rows to a CSV that can be resumed.

    rows = run_jobs(jobs, "results/x/runs.csv", key=("mode", "size", "direction", "obs_seed"),
                    init=init_sweeper, init_args=(cfg,), work=my_module.do_job)

- Processes are started with the "spawn" method: JPype cannot start a JVM in a forked
  child. The parent never starts the JVM; each worker builds its own state once
  (default: one Sweeper(cfg), kept in the module global WORKER) and reuses it.
- `work(job) -> row` must be a module-level function (spawn pickles it by name). It may
  also return (row, extra_rows); extra rows go to `extra_csv_path` (e.g. every run made
  inside one bisection job), main rows to `csv_path`.
- Resumable: jobs whose key is already in csv_path are skipped. Rows are written and
  flushed as they arrive, so a crash or Ctrl-C loses only the jobs in flight. Extra rows
  of jobs that never finished are dropped on resume, so nothing is counted twice.
- One failing job cannot kill the pool: the exception text is stored in its row.
- A worker process that dies outright (e.g. the JVM crashing with SIGSEGV) breaks the pool; the
  pool is then restarted and the unfinished jobs rerun (at most max_restarts times).
- Workers are recycled so no JVM lives long enough to accumulate state: jobs run in batches of
  workers x max_tasks_per_child, each batch on a fresh pool that is shut down before the next one.
  (Not ProcessPoolExecutor's own max_tasks_per_child: on Python 3.13 / macOS it hung once every
  worker had retired and no replacements were started; seen at exactly 9 workers x 100 jobs.)
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

WORKER: dict = {}          # per-process state built by the initializer


def init_sweeper(cfg: dict) -> None:
    """Default worker initializer: one Sweeper (JVM, estimator, propagator) per process."""
    import warnings

    warnings.filterwarnings("ignore", message="No observer columns")
    from .sweep import Sweeper

    WORKER["sweeper"] = Sweeper(cfg)


def _init(init, init_args):
    init(*init_args)


def _run(work, job):
    t0 = time.time()
    try:
        return work(job)
    except Exception as e:                       # never let one case kill the pool
        tb = traceback.format_exc().strip().splitlines()
        return {**job, "outcome": "other_error", "error": f"{type(e).__name__}: {e}"[:300],
                "traceback_tail": " | ".join(tb[-3:])[:500], "seconds": time.time() - t0}


# ----------------------------------------------------------------------------- CSV helpers

def _norm(v) -> str:
    """Key normalization so a value read back from the CSV equals the one in the job."""
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return str(v)


def job_key(d: dict, key) -> tuple:
    return tuple(_norm(d.get(k)) for k in key)


def read_rows(path: str) -> list[dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class CsvAppender:
    """Appends dict rows; widens the header (rewriting the file once) when new columns appear."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = read_rows(path)
        self.fields = list(rows[0].keys()) if rows else []

    def append(self, rows: list[dict]) -> None:
        if not rows:
            return
        new = [k for r in rows for k in r if k not in self.fields]
        if new or not os.path.exists(self.path):
            old = read_rows(self.path)
            self.fields += list(dict.fromkeys(new))
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()
                w.writerows(old)
            os.replace(tmp, self.path)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerows(rows)
            f.flush()
            os.fsync(f.fileno())


def _drop_orphans(extra_path: str, done: set, key) -> None:
    rows = read_rows(extra_path)
    keep = [r for r in rows if job_key(r, key) in done]
    if len(keep) == len(rows):
        return
    tmp = extra_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    os.replace(tmp, extra_path)


# ----------------------------------------------------------------------------- driver

def _fmt_s(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def run_jobs(jobs: list[dict], csv_path: str, key, work, init=init_sweeper, init_args=(),
             workers: int | None = None, extra_csv_path: str | None = None, label: str = "",
             max_restarts: int = 5, max_tasks_per_child: int = 100) -> list[dict]:
    """Run every job not already in csv_path; returns all rows of csv_path afterwards."""
    restarts = 0
    while True:
        try:
            return _run_jobs_once(jobs, csv_path, key, work, init, init_args, workers, extra_csv_path, label,
                                  max_tasks_per_child)
        except BrokenProcessPool:
            restarts += 1
            print(f"\n{label + ': ' if label else ''}a worker process died (pool broken); "
                  f"restart {restarts}/{max_restarts}, finished rows are kept", file=sys.stderr, flush=True)
            if restarts > max_restarts:
                raise


def _run_jobs_once(jobs, csv_path, key, work, init, init_args, workers, extra_csv_path, label, max_tasks_per_child):
    key = tuple(key)
    done = {job_key(r, key) for r in read_rows(csv_path)}
    if extra_csv_path:
        _drop_orphans(extra_csv_path, done, key)
    todo, seen = [], set(done)
    for j in jobs:
        k = job_key(j, key)
        if k not in seen:
            seen.add(k)
            todo.append(j)
    total = len(todo)
    tag = f"{label}: " if label else ""
    if not total:
        print(f"{tag}all {len(jobs)} jobs already in {csv_path}")
        return read_rows(csv_path)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"{tag}{total} jobs to run ({len(jobs) - total} already done), {workers} workers -> {csv_path}")

    main, extra = CsvAppender(csv_path), CsvAppender(extra_csv_path) if extra_csv_path else None
    t0, n = time.time(), 0
    batch = max(1, workers * int(max_tasks_per_child or 10**9))
    for start in range(0, total, batch):
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init, initargs=(init, init_args))
        try:
            futures = [ex.submit(_run, work, j) for j in todo[start:start + batch]]
            for fut in as_completed(futures):
                out = fut.result()
                row, extra_rows = (out if isinstance(out, tuple) else (out, []))
                if extra is not None:
                    extra.append(extra_rows)              # extras first: a main row means the job is complete
                main.append([row])
                n += 1
                el = time.time() - t0
                eta = el / n * (total - n)
                print(f"\r  {tag}{n}/{total} done, {_fmt_s(el)} elapsed, ~{_fmt_s(eta)} left   ",
                      end="", flush=True)
        except KeyboardInterrupt:
            print(f"\nInterrupted after {n}/{total}; finished rows are saved. Rerun to resume.", file=sys.stderr)
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    print()
    return read_rows(csv_path)
