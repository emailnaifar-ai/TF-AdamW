"""Kernel-SHAPE experiment at FIXED mean delay (Reviewer 2, comment 4).

Reviewer 2 asks what the tempered-fractional weighting profile buys over an exponential
one.  The stationary kernel geometry (exp_geometry.py) shows that, at matched mean delay
mu, the exponential kernel actually attains a HIGHER effective sample size, while the
tempered-fractional kernel attains a higher HEAD WEIGHT p_0 = 1/d.  This experiment tests
which of the two quantities actually governs optimization performance: we hold mu FIXED
and sweep the kernel shape, so that staleness is constant by construction and only the
weight profile changes.

Kernel families at fixed mu (all normalized to total mass 1):
  boxcar        : uniform over W = 2 mu + 1 lags                (lowest  p_0)
  exponential   : (1-b) b^j with b = mu/(1+mu)                  (AdamW's kernel)
  multi-EMA     : positive mixture of 8 exponentials, mean delay mu
  tempered frac : alpha in {0.3,...,0.95}, lambda = ln(1+alpha/mu)  (increasing p_0)
"""
import os, json, time
import numpy as np
import kernels as K
import synth as S

OUT = "results"
os.makedirs(OUT, exist_ok=True)
MU = 13.652916545146125
JMAX = 4000


def family(mu):
    fams = {}
    W = int(round(2 * mu + 1))
    fams["boxcar"] = np.ones(W) / W
    b = K.beta_for_mean_delay(mu)
    j = np.arange(JMAX)
    fams["exponential (AdamW)"] = (1 - b) * b ** j
    rho, c = K.multiema_logspaced(8, mu, "matched")
    fams["multi-EMA (M=8)"] = sum(c[i] * rho[i] ** j for i in range(len(c)))
    for a in (0.3, 0.5, 0.7, 0.9, 0.95):
        lam = K.lam_for_mean_delay(a, mu)
        kap = K.tempered_kernel(a, lam, JMAX - 1)
        fams[f"tempered fractional $\\alpha$={a}"] = kap / kap.sum()
    return fams


def run_kernel(kern, gfn, loss, x0, nit, lr, trunc=400):
    """AdamW update with an arbitrary normalized causal kernel as the first moment."""
    k_ = np.asarray(kern[:trunc], float)
    J = len(k_)
    x = x0.copy(); p = len(x)
    v = np.zeros(p); buf = np.zeros((J, p)); pos = 0
    b2, eps = 0.999, 1e-8
    out = np.empty(nit)
    for k in range(1, nit + 1):
        g = gfn(x)
        buf[pos] = g; pos = (pos + 1) % J
        L = min(k, J)
        idx = (pos - 1 - np.arange(L)) % J
        mbar = (k_[:L] @ buf[idx]) / k_[:L].sum()
        v = b2 * v + (1 - b2) * g ** 2
        x = x - lr * mbar / (np.sqrt(v / (1 - b2 ** k)) + eps)
        out[k - 1] = loss(x)
    return out


def evaluate(kern, insts, seeds, lr, nu=2.0, nit=1500):
    vals = []
    for (isd, p, cond, spec, noise) in insts:
        A, b, fstar, x0 = S.make_quadratic(p, cond, spec, isd)
        loss = lambda x: 0.5 * x @ A @ x - b @ x - fstar
        for s in seeds:
            rng = np.random.default_rng(1000 * isd + s)
            g = S.grad_fn(A, b, nu, noise, p, rng)
            vals.append(max(np.mean(run_kernel(kern, g, loss, x0, nit, lr)[-200:]), 1e-14))
    return np.array(vals)


if __name__ == "__main__":
    t0 = time.time()
    tune = S.instance_grid(S.TUNE_INST)
    ev = S.instance_grid(S.EVAL_INST)
    fams = family(MU)
    res = {}
    for name, kern in fams.items():
        st = K.kernel_stats(kern[:JMAX])
        best, bv = None, np.inf
        for lr in S.LRGRID:
            v = evaluate(kern, tune, S.TUNE_SEEDS, lr)
            sc = float(np.exp(np.mean(np.log(v))))
            if np.isfinite(sc) and sc < bv:
                bv, best = sc, lr
        v = evaluate(kern, ev, S.EVAL_SEEDS, best)
        res[name] = dict(lr=best, mu=st["mu"], ess=st["ess"], p0=st["p0"],
                         gmean=float(np.exp(np.mean(np.log(v)))),
                         median=float(np.median(v)),
                         vals=[float(x) for x in v])
        print(f"{name:32s} mu={st['mu']:6.2f} ESS={st['ess']:6.2f} p0={st['p0']:.4f} "
              f"lr={best:<6g} gmean={res[name]['gmean']:.3e}  [{time.time()-t0:.0f}s]",
              flush=True)
    # rank correlations: which geometric quantity predicts performance?
    from scipy.stats import spearmanr
    names = list(res)
    g = np.array([res[n]["gmean"] for n in names])
    p0 = np.array([res[n]["p0"] for n in names])
    es = np.array([res[n]["ess"] for n in names])
    r_p0 = spearmanr(p0, g); r_es = spearmanr(es, g)
    print(f"\nSpearman(head weight p0, final gap) = {r_p0.statistic:+.3f} (p={r_p0.pvalue:.4f})")
    print(f"Spearman(ESS,           final gap) = {r_es.statistic:+.3f} (p={r_es.pvalue:.4f})")
    json.dump(dict(mu=MU, results=res,
                   spearman_p0=[float(r_p0.statistic), float(r_p0.pvalue)],
                   spearman_ess=[float(r_es.statistic), float(r_es.pvalue)]),
              open(os.path.join(OUT, "synth_shape.json"), "w"), indent=1)
    print("wrote results/synth_shape.json")
