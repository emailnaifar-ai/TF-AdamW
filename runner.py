"""Parallel job runner.

Usage:
    python runner.py worker  <jobs.json> <out.jsonl>      # run one shard sequentially
    python runner.py dispatch <jobs.json> <outdir> [W]    # split across W processes

Thread scaling of these small models is poor (1 thread ~= 8 threads), so running W
single-threaded worker processes gives close to a W-fold speed-up on an 8-core box.
"""
import os, sys, json, subprocess, time, hashlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _key(job):
    return hashlib.md5(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]


def worker(jobfile, outfile):
    import torch
    torch.set_num_threads(1)
    import dl_core as C
    jobs = json.load(open(jobfile))
    ctxspec = jobs["ctx"]
    ctx = C.make_ctx(**ctxspec)
    # resume: a job already completed by ANY shard in this output directory is skipped
    done = set()
    import glob as _glob
    for f in _glob.glob(os.path.join(os.path.dirname(outfile) or ".", "shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:          # a failed job is retried, not inherited
                done.add(r["key"])
    with open(outfile, "a") as f:
        for i, job in enumerate(jobs["jobs"]):
            k = _key(job)
            if k in done:
                continue
            try:
                r = C.run_job(job, ctx)
                r["key"] = k
            except Exception as e:
                r = dict(key=k, job=job, error=repr(e))
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{i+1}/{len(jobs['jobs'])}] {job.get('tag','')} {job['opt']} "
                  f"lr={job['lr']:g} seed={job.get('seed')} -> "
                  f"{r.get('test', r.get('error'))} ({r.get('secs','?')}s)", flush=True)


def dispatch(jobfile, outdir, W=7):
    spec = json.load(open(jobfile))
    jobs = spec["jobs"]
    os.makedirs(outdir, exist_ok=True)
    # round-robin so each shard gets a similar mix of costs
    shards = [[] for _ in range(W)]
    for i, j in enumerate(jobs):
        shards[i % W].append(j)
    procs = []
    for w, sh in enumerate(shards):
        if not sh:
            continue
        sf = os.path.join(outdir, f"shard{w}.json")
        json.dump(dict(ctx=spec["ctx"], jobs=sh), open(sf, "w"))
        of = os.path.join(outdir, f"shard{w}.jsonl")
        lg = open(os.path.join(outdir, f"shard{w}.log"), "w")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        procs.append(subprocess.Popen([sys.executable, "runner.py", "worker", sf, of],
                                      stdout=lg, stderr=subprocess.STDOUT, env=env))
    t0 = time.time()
    for p in procs:
        p.wait()
    print(f"ALL DONE in {time.time()-t0:.0f}s ({len(jobs)} jobs, {W} workers)")


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        dispatch(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 7)
