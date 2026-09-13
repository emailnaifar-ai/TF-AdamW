"""Enforce the stated tuning rule on the synthetic instance study.

`synth.py` records, for every optimizer, whether its selected step size sat at an edge of
the searched grid.  This script extends the grid geometrically past any such edge, on the
DEDICATED TUNING INSTANCES ONLY, until the selected value is interior, then re-evaluates
that optimizer on the held-out instances and rewrites its record.  Nothing else is touched.

    python extend_synth.py            # fix every edge-selected method
    python extend_synth.py --dry      # report what would be probed, run nothing
"""
import os, sys, json, time
import numpy as np
import synth as S

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "synth_instances.json")
FACTOR = 3.0                      # the shipped grids are geometric with ratio ~3
LR_MAX, LR_MIN = 30.0, 1e-7


def tune_score(mth, tune, lr):
    try:
        v = S.evaluate(mth, tune, S.TUNE_SEEDS, lr)
        return float(np.exp(np.mean(np.log(v))))
    except Exception:
        return np.inf


def main(dry=False):
    d = json.load(open(RES, encoding="utf-8"))
    R = d["results"]
    flagged = [(m, r) for m, r in R.items() if r.get("edge")]
    print(f"{len(flagged)} method(s) selected a grid edge: "
          + ", ".join(f"{m} (lr={r['lr']:g}, {r['edge']})" for m, r in flagged))
    if dry or not flagged:
        return
    tune = S.instance_grid(S.TUNE_INST)
    ev = S.instance_grid(S.EVAL_INST)
    for mth, rec in flagged:
        t0 = time.time()
        up = rec["edge"] == "UPPER-EDGE"
        best_lr = rec["lr"]
        best_sc = tune_score(mth, tune, best_lr)
        probed = []
        lr = best_lr
        while True:
            lr = lr * FACTOR if up else lr / FACTOR
            if lr > LR_MAX or lr < LR_MIN:
                print(f"   {mth}: reached the grid cap at {lr:g}; stopping")
                break
            sc = tune_score(mth, tune, lr)
            probed.append((lr, sc))
            print(f"   {mth}: lr={lr:<8g} tuning gmean={sc:.4e}"
                  f"{'  <-- better' if sc < best_sc else ''}", flush=True)
            if sc < best_sc:
                best_sc, best_lr = sc, lr
            else:
                break                      # the optimum is now interior
        if best_lr == rec["lr"]:
            print(f"   {mth}: unchanged at lr={best_lr:g}")
            rec["edge"] = ""
            rec["extended_grid"] = [float(x) for x, _ in probed]
            continue
        v = S.evaluate(mth, ev, S.EVAL_SEEDS, best_lr)
        old = rec["gmean"]
        rec.update(lr=best_lr,
                   gmean=float(np.exp(np.mean(np.log(v)))),
                   median=float(np.median(v)),
                   q25=float(np.percentile(v, 25)),
                   q75=float(np.percentile(v, 75)),
                   vals=[float(x) for x in v],
                   edge="",
                   extended_grid=[float(x) for x, _ in probed])
        print(f"   {mth}: lr {rec['lr']:g}, gmean {old:.3e} -> {rec['gmean']:.3e} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    d.setdefault("meta", {})["grid_extended"] = True
    json.dump(d, open(RES, "w"), indent=1)
    print("rewrote", os.path.basename(RES))


if __name__ == "__main__":
    main("--dry" in sys.argv)
