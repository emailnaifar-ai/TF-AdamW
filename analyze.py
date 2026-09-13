"""Aggregate evaluation results and run the PAIRED statistical tests that Reviewer 1
(comment 2) and Reviewer 2 (comment 11) require: paired seeds with identical label-noise
realizations, paired differences, bootstrap confidence intervals, a paired t-test and a
Wilcoxon signed-rank test, with a Holm correction over the family of comparisons against
the reference optimizer.
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

OUT = "results"
os.makedirs(OUT, exist_ok=True)


def load(exp, kind="eval"):
    rows = []
    for f in glob.glob(f"out_{exp}_{kind}/shard*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                rows.append(r)
    return rows


def selected_lr(exp):
    """(optimizer, noise) -> learning rate chosen on the validation split.

    Written by build_jobs.build_eval.  Re-tuning after a grid widening leaves stale
    evaluation rows at the superseded learning rate in the same output directory, so
    every aggregation filters on this map rather than trusting the directory.
    """
    p = f"selected_lr_{exp}.json"
    if not os.path.exists(p):
        return None
    sel = {}
    for k, lr in json.load(open(p)).items():
        m, nz = k.rsplit("|", 1)
        if m.startswith("AdamW-b1tuned:"):
            m = "AdamW-b1=" + m.split(":", 1)[1]
        sel[(m, float(nz))] = lr
    return sel


def table(exp, metric="test"):
    """method -> noise -> {seed: value}"""
    sel = selected_lr(exp)
    D = {}
    for r in load(exp):
        j = r["job"]
        if sel is not None:
            lr = sel.get((j["opt"], float(j["noise"])))
            if lr is not None and abs(j["lr"] - lr) > 1e-12:
                continue
        D.setdefault(j["opt"], {}).setdefault(j["noise"], {})[j["seed"]] = r[metric]
    return D


def boot_ci(d, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(D, ref, noise, alt="two-sided"):
    """Two-sided by default: the reported confidence intervals are two-sided, and a star
    in the tables means the interval excludes zero, so a one-sided p-value alongside them
    would be incoherent (and would halve every quoted p).  Paired comparison of every method
    against `ref` at one noise level."""
    if ref not in D or noise not in D[ref]:
        return []
    out = []
    for m in D:
        if m == ref or noise not in D[m]:
            continue
        seeds = sorted(set(D[m][noise]) & set(D[ref][noise]))
        if len(seeds) < 3:
            continue
        a = np.array([D[m][noise][s] for s in seeds])
        b = np.array([D[ref][noise][s] for s in seeds])
        d = a - b
        t, pt = stats.ttest_rel(a, b, alternative=alt)
        try:
            w, pw = stats.wilcoxon(a, b, alternative=alt)
        except Exception:
            pw = np.nan
        lo, hi = boot_ci(d)
        out.append(dict(method=m, n=len(seeds), mean=float(a.mean()),
                        sd=float(a.std(ddof=1)), ref_mean=float(b.mean()),
                        diff=float(d.mean()), ci=[lo, hi], t=float(t),
                        p_t=float(pt), p_wilcoxon=float(pw),
                        cohen_dz=float(d.mean() / (d.std(ddof=1) + 1e-12))))
    # Holm-Bonferroni over the family
    ps = sorted(range(len(out)), key=lambda i: out[i]["p_t"])
    K = len(out)
    prev = 0.0
    for rank, i in enumerate(ps):
        adj = min(1.0, (K - rank) * out[i]["p_t"])
        adj = max(adj, prev); prev = adj
        out[i]["p_holm"] = adj
    return sorted(out, key=lambda r: -r["mean"])


def report(exp, ref="AdamW", metric="test", scale=100.0):
    D = table(exp, metric)
    if not D:
        print(f"[{exp}] no results"); return {}
    res = {}
    for noise in sorted({n for m in D for n in D[m]}):
        rows = paired(D, ref, noise)
        res[str(noise)] = rows
        print(f"\n=== {exp}  noise={noise}  (reference = {ref}, metric = {metric}) ===")
        if ref in D and noise in D[ref]:
            b = np.array(list(D[ref][noise].values()))
            print(f"  {ref:22s} n={len(b):2d}  {b.mean()*scale:7.3f} +- {b.std(ddof=1)*scale:5.3f}")
        print(f"  {'method':22s} {'n':>2} {'mean':>8} {'sd':>6} {'diff':>8} "
              f"{'95% CI of diff':>20} {'p(t)':>9} {'p(Holm)':>9}")
        for r in rows:
            print(f"  {r['method']:22s} {r['n']:2d} {r['mean']*scale:8.3f} "
                  f"{r['sd']*scale:6.3f} {r['diff']*scale:+8.3f} "
                  f"[{r['ci'][0]*scale:+7.3f},{r['ci'][1]*scale:+7.3f}] "
                  f"{r['p_t']:9.4f} {r['p_holm']:9.4f}")
    json.dump(res, open(os.path.join(OUT, f"stats_{exp}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "fmnist"
    ref = sys.argv[2] if len(sys.argv) > 2 else "AdamW"
    report(exp, ref)
