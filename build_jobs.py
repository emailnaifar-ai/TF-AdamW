"""Build job specs for each experiment, with a STRICT separation between the seeds used
for hyper-parameter selection and the seeds used for evaluation (Reviewer 1 comment 8,
Reviewer 2 comment 9), and with PAIRED label-noise realizations shared by every
optimizer (Reviewer 2 comment 11).

    python build_jobs.py tune <exp>            -> jobs_<exp>_tune.json
    python build_jobs.py eval <exp>            -> jobs_<exp>_eval.json  (reads tune out)
"""
import os, sys, json, glob
import numpy as np

TUNE_SEEDS = [100]               # dedicated tuning seed, disjoint from evaluation seeds
EVAL_SEEDS = list(range(10))
LR = [5e-4, 1e-3, 2e-3, 4e-3]
LR_LION = [1e-4, 3e-4, 1e-3, 3e-3]
LR_HI = [1e-3, 3e-3, 1e-2, 3e-2]       # non-adaptive / TFGD-style methods
B1_GRID = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

# ---- method sets -----------------------------------------------------------
CORE = ["Adam", "AdamW", "AMSGrad", "RMSProp", "Lion", "F-Adam", "SOE-TF-AdamW"]
MATCHED = ["AdamW-MM", "MultiEMA-8", "Boxcar-MM", "F-Adam-MM"]
STRONG = ["RAdam", "AdaBelief", "Adafactor", "SAM-AdamW"]
FRACTIONAL = ["FracAdam-Shin", "lambda-FAdaMax", "FOSGD-ME", "TFGD"]
B1SWEEP = [f"AdamW-b1={b}" for b in B1_GRID]
THEORY = ["SOE-TF-AdamW-AMS"]

FULLSET = CORE + MATCHED + STRONG + FRACTIONAL + B1SWEEP + THEORY
FULLSET_NOB1 = CORE + MATCHED + STRONG + FRACTIONAL + THEORY


def grid(m, e=None):
    if e and "lrs" in e:
        if m == "Lion":
            return [3e-4, 1e-3]
        if m in ("TFGD", "FOSGD-ME"):
            return [3e-3, 1e-2]
        return e["lrs"]
    if m == "Lion":
        return LR_LION
    if m in ("TFGD", "FOSGD-ME"):
        return LR_HI
    if m == "RMSProp":
        return [3e-4, 1e-3, 2e-3, 4e-3]
    return LR


EXPS = {
    # headline statistical study: 10 seeds, paired noise, full method set
    "fmnist": dict(dataset="fmnist", methods=FULLSET, noises=[0.0, 0.4],
                   epochs=12, n_train=12000, eval_seeds=EVAL_SEEDS, ctx={}),
    # standard full training set + longer schedule (Reviewer 2 comment 10)
    "fmnistfull": dict(dataset="fmnist", methods=CORE + ["AdamW-MM", "MultiEMA-8"],
                       noises=[0.0, 0.4], epochs=20, n_train=60000, n_val=5000,
                       n_test=5000, eval_seeds=[0, 1, 2, 3, 4], tune_seeds=[100],
                       lrs=[1e-3, 2e-3, 4e-3], ctx={}),
    # ResNet-20 on the FULL CIFAR-10 training set with a cosine schedule
    "cifar": dict(dataset="cifar", methods=["AdamW", "AdamW-MM", "MultiEMA-8",
                                            "SOE-TF-AdamW"],
                  noises=[0.0, 0.4], epochs=5, n_train=45000, sched="cosine",
                  eval_seeds=[0, 1, 2], tune_seeds=[100], lrs=[1e-3, 2e-3, 4e-3], ctx={}),
    # non-vision sequence model (Reviewer 1 comment 1 / Reviewer 2 comment 10)
    "lm": dict(dataset="lm", methods=["AdamW", "AdamW-MM", "Lion",
                                      "MultiEMA-8", "F-Adam-MM", "SOE-TF-AdamW"],
               noises=[0.0, 0.2], steps=1200, eval_seeds=[0, 1, 2],
               tune_seeds=[100], lrs=[1e-3, 3e-3], ctx={}),
    # (alpha, lambda) sensitivity on a REAL network (Reviewer 2 comment 15)
    "heat": dict(dataset="fmnist", methods=None, noises=[0.4], epochs=12,
                 n_train=12000, eval_seeds=[0, 1, 2], tune_seeds=[100], ctx={}),
    # SOE rank M cost/accuracy trade-off (Reviewer 1 comment 6 / Reviewer 2 comment 14)
    "rank": dict(dataset="fmnist", methods=[f"SOE-TF-AdamW-M={m}" for m in (1, 2, 4, 8, 12)],
                 noises=[0.0, 0.4], epochs=12, n_train=12000,
                 eval_seeds=[0, 1, 2, 3, 4], tune_seeds=[100], ctx={}),
}

AL_GRID = [(a, l) for a in (0.3, 0.5, 0.7, 0.9) for l in (0.02, 0.05, 0.1, 0.2)]


def exp_spec(name):
    e = dict(EXPS[name])
    if name == "heat":
        e["methods"] = [f"SOE-TF-AdamW-al={a}_{l}" for a, l in AL_GRID]
        e["ctx"] = dict(extra_al=AL_GRID)
    if name == "rank":
        e["ctx"] = dict(extra_M=(1, 2, 4, 8, 12))
    return e


def base_job(e, m, lr, seed, noise, tag):
    j = dict(dataset=e["dataset"], opt=m, lr=lr, seed=seed, noise=noise,
             noise_seed=seed, tag=tag)
    for k in ("epochs", "steps", "n_train", "n_val", "n_test", "sched", "batch"):
        if k in e:
            j[k] = e[k]
    return j


def build_tune(name):
    e = exp_spec(name)
    seeds = e.get("tune_seeds", TUNE_SEEDS)
    jobs = []
    for noise in e["noises"]:
        for m in e["methods"]:
            for lr in grid(m, e):
                for s in seeds:
                    jobs.append(base_job(e, m, lr, s, noise, f"{name}-tune"))
    out = f"jobs_{name}_tune.json"
    json.dump(dict(ctx=e["ctx"], jobs=jobs), open(out, "w"))
    print(f"{out}: {len(jobs)} jobs")


def read_results(pattern):
    rows = []
    for f in glob.glob(pattern):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return [r for r in rows if "error" not in r]


def build_eval(name):
    e = exp_spec(name)
    rows = read_results(f"out_{name}_tune/shard*.jsonl")
    if not rows:
        raise SystemExit(f"no tuning results in out_{name}_tune/")
    best = {}
    for r in rows:
        j = r["job"]
        k = (j["opt"], j["noise"])
        best.setdefault(k, {}).setdefault(j["lr"], []).append(r["val"])
    sel = {k: max(v.items(), key=lambda kv: np.mean(kv[1]))[0] for k, v in best.items()}
    # The beta1 sweep exists ONLY to give AdamW a first-moment-memory search of at
    # least the same budget the proposed method gets.  Carry forward the winner as
    # "AdamW-b1tuned" and drop the losing sweep points from the evaluation.
    scores = {k: max(np.mean(v) for v in d.values()) for k, d in best.items()}
    noises = sorted({n for (_, n) in sel})
    for nz in noises:
        cands = [(m, n) for (m, n) in sel if n == nz and m.startswith("AdamW-b1=")]
        if not cands:
            continue
        win = max(cands, key=lambda k: scores[k])
        sel[("AdamW-b1tuned:" + win[0].split("=")[1], nz)] = sel[win]
        for c in cands:
            sel.pop(c)
    jobs = []
    for (m, noise), lr in sorted(sel.items(), key=lambda x: str(x[0])):
        mm = m.split(":")[0] if m.startswith("AdamW-b1tuned:") else m
        real = ("AdamW-b1=" + m.split(":")[1]) if m.startswith("AdamW-b1tuned:") else m
        # 10 seeds for the KEY (noisy) condition; 5 for the clean control condition
        seeds = e["eval_seeds"] if (noise > 0 or len(e["eval_seeds"]) <= 5)             else e["eval_seeds"][:5]
        for s in seeds:
            j = base_job(e, real, lr, s, noise, f"{name}-eval")
            j["alias"] = mm
            jobs.append(j)
    out = f"jobs_{name}_eval.json"
    json.dump(dict(ctx=e["ctx"], jobs=jobs), open(out, "w"))
    json.dump({f"{m}|{n}": lr for (m, n), lr in sel.items()},
              open(f"selected_lr_{name}.json", "w"), indent=1)
    print(f"{out}: {len(jobs)} jobs; selected LRs written")


if __name__ == "__main__":
    (build_tune if sys.argv[1] == "tune" else build_eval)(sys.argv[2])
