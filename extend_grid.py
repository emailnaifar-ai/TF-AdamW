"""Enforce the stated fairness rule: widen the learning-rate grid until no method selects a
boundary value.

Reads the completed tuning results, finds every (method, noise) cell whose best learning
rate is the smallest or largest value tried for that cell, and emits a supplementary tuning
job file that probes one geometric step beyond the boundary. Repeat until it emits nothing.

    python extend_grid.py <exp>          -> jobs_<exp>_tune.json (supplementary round)
"""
import os, sys, json, glob
import numpy as np
import build_jobs as BJ

FACTOR = 2.0
MAXLR, MINLR = 1.0, 1e-5


def cells(exp):
    rows = []
    for f in glob.glob(f"out_{exp}_tune/shard*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                rows.append(r)
    D = {}
    for r in rows:
        j = r["job"]
        D.setdefault((j["opt"], j["noise"]), {}).setdefault(j["lr"], []).append(r["val"])
    return D


def build(exp):
    e = BJ.exp_spec(exp)
    D = cells(exp)
    if not D:
        raise SystemExit(f"no tuning results for {exp}")
    jobs, pinned, capped = [], [], []
    seeds = e.get("tune_seeds", BJ.TUNE_SEEDS)
    for (m, nz), byl in D.items():
        tried = sorted(byl)
        best = max(byl.items(), key=lambda kv: np.mean(kv[1]))[0]
        new = []
        if best == tried[-1] and best * FACTOR <= MAXLR:
            new.append(best * FACTOR)
        if best == tried[0] and best / FACTOR >= MINLR:
            new.append(best / FACTOR)
        if new:
            pinned.append((m, nz, best, new))
        elif len(tried) > 1 and best in (tried[0], tried[-1]):
            capped.append((m, nz, best))
        for lr in new:
            for s in seeds:
                jobs.append(BJ.base_job(e, m, lr, s, nz, f"{exp}-tune"))
    out = f"jobs_{exp}_tune.json"
    json.dump(dict(ctx=e["ctx"], jobs=jobs), open(out, "w"))
    print(f"{len(pinned)} boundary-selected cells; {len(jobs)} supplementary jobs -> {out}")
    if capped:
        print(f"   !! {len(capped)} cell(s) still at a boundary but at the grid cap "
              f"[{MINLR:g},{MAXLR:g}] - the no-boundary rule cannot be met for these:")
        for m, nz, b in sorted(capped):
            print(f"      {m:22s} noise={nz} best={b:g}")
    for m, nz, b, new in sorted(pinned):
        print(f"   {m:22s} noise={nz} best={b:<8g} -> probing {', '.join(f'{x:g}' for x in new)}")
    return len(jobs)


if __name__ == "__main__":
    n = build(sys.argv[1] if len(sys.argv) > 1 else "fmnist")
    sys.exit(0 if n else 3)          # exit 3 == nothing left to widen
