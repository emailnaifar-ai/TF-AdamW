"""Controlled stochastic optimization over a DISTRIBUTION of problem instances.

Addresses:
  * Reviewer 1 comment 9 / Reviewer 2 comment 19 -- results are aggregated over many
    independently generated problem instances (condition numbers, dimensions, spectra,
    noise models, random orthogonal bases), not a single favourable construction.
  * Reviewer 1 comment 8 / Reviewer 2 comment 9 -- learning rates are selected on
    DISJOINT tuning instances and tuning seeds, then evaluated on unseen instances and
    unseen seeds.
  * Reviewer 2 comments 3, 5, 7 -- memory-matched controls (tuned-beta1 AdamW,
    mean-delay-matched AdamW, equal-state multi-EMA, boxcar, matched-memory untempered
    fractional kernel).
  * Reviewer 1 comment 5 / Reviewer 2 comment 6 -- published fractional optimizers.
"""
import os, sys, json, time, itertools
import numpy as np
import kernels as K

OUT = "results"
os.makedirs(OUT, exist_ok=True)

ALPHA, LAM, M = 0.7, 0.05, 8
FIT = K.soe_fit_nnls(ALPHA, LAM, M)
MU = K.mean_delay_tf(ALPHA, LAM)
B1_MM = K.beta_for_mean_delay(MU)
BOX_W = K.boxcar_window_for_mean_delay(MU)
_J = 2
while (np.arange(_J) * K.w_alpha(ALPHA, _J - 1)).sum() / K.w_alpha(ALPHA, _J - 1).sum() < MU:
    _J += 1
FADAM_J_MM = _J


# ------------------------------------------------------------------ problems
def make_quadratic(p, cond, spectrum, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    if spectrum == "log":
        ev = np.logspace(0, np.log10(cond), p)
    elif spectrum == "cluster":
        ev = np.concatenate([np.ones(p // 2), np.full(p - p // 2, cond)])
    else:                                             # power-law spectrum
        ev = (np.arange(1, p + 1) ** -1.0)
        ev = 1 + (cond - 1) * (ev - ev.min()) / (ev.max() - ev.min())
    A = (Q * ev) @ Q.T
    b = rng.standard_normal(p)
    xstar = np.linalg.solve(A, b)
    fstar = 0.5 * xstar @ A @ xstar - b @ xstar
    return A, b, fstar, rng.standard_normal(p)


def grad_fn(A, b, nu, noise, p, rng):
    def g(x):
        gr = A @ x - b
        if noise == "mult":
            return gr + nu * rng.standard_normal(p) * np.linalg.norm(gr) / np.sqrt(p)
        if noise == "add":
            return gr + nu * rng.standard_normal(p)
        return gr + nu * rng.standard_t(3, p) / np.sqrt(3.0)     # heavy-tailed
    return g


# ------------------------------------------------------------------ optimizers
def run(method, gfn, loss, x0, nit, lr, seed):
    """Generic numpy driver.  Returns the loss curve."""
    rng = np.random.default_rng(seed)
    x = x0.copy(); p = len(x)
    b2, eps = 0.999, 1e-8
    v = np.zeros(p); vmax = np.zeros(p); m = np.zeros(p)
    buf = None; pos = 0; states = None; energy = 1.0
    kern = None; b1 = 0.9; ams = False; adaptive = True; ada_kind = "rms"

    if method == "AdamW":
        pass
    elif method.startswith("AdamW-b1="):
        b1 = float(method.split("=")[1])
    elif method == "AdamW-MM":
        b1 = B1_MM
    elif method == "AMSGrad":
        ams = True
    elif method == "RMSProp":
        b1 = 0.0
    elif method == "TF-AdamW (exact)":
        kern = K.tempered_kernel(ALPHA, LAM, 999)
    elif method in ("SOE-TF-AdamW", "SOE-TF-AdamW-AMS", "TFGD"):
        states = np.zeros((len(FIT["c"]), p))
        ams = method.endswith("AMS")
        adaptive = method != "TFGD"
    elif method == "MultiEMA-8":
        rho_m, c_m = K.multiema_logspaced(8, MU, "matched")
        states = np.zeros((len(c_m), p))
    elif method == "Boxcar-MM":
        kern = np.ones(BOX_W)
    elif method == "F-Adam":
        kern = K.w_alpha(ALPHA, 63)
    elif method == "F-Adam-MM":
        kern = K.w_alpha(ALPHA, FADAM_J_MM - 1)
    elif method == "FracAdam-Shin":
        J = 8; cg = np.empty(J); cg[0] = 1.0
        for j in range(1, J):
            cg[j] = cg[j - 1] * (j - 1 - 0.9) / j
        kern = cg; ada_kind = "shin"
    elif method == "lambda-FAdaMax":
        kern = K.w_alpha(ALPHA, 31); ada_kind = "adamax"
    elif method == "FOSGD-ME":
        kern = K.w_alpha(ALPHA, 31); ada_kind = "energy"
    else:
        raise ValueError(method)

    if kern is not None:
        buf = np.zeros((len(kern), p))
    if method == "MultiEMA-8":
        cc, rr = c_m, rho_m
    elif states is not None:
        cc, rr = FIT["c"], FIT["rho"]

    out = np.empty(nit)
    for k in range(1, nit + 1):
        g = gfn(x)
        # ---- first moment
        if states is not None:
            states = states * rr[:, None] + g
            mm = cc @ states
            D = float(np.sum(cc * (1 - rr ** k) / (1 - rr)))
            mbar = mm / D
        elif kern is not None:
            buf[pos] = g; pos = (pos + 1) % len(kern)
            L = min(k, len(kern))
            idx = (pos - 1 - np.arange(L)) % len(kern)
            mbar = kern[:L] @ buf[idx]
            if ada_kind != "shin":
                mbar = mbar / kern[:L].sum()
        else:
            m = b1 * m + (1 - b1) * g
            mbar = m / (1 - b1 ** k) if b1 > 0 else g
        # ---- second moment / update
        if not adaptive:                                   # TFGD: constant denominator
            x = x - lr * mbar
        elif ada_kind == "adamax":
            vmax = np.maximum(vmax * 0.999 * 0.999, np.abs(g))
            x = x - lr * mbar / (vmax + eps)
        elif ada_kind == "energy":
            energy = energy / (1.0 + lr * float(mbar @ mbar))
            x = x - lr * max(energy, 1e-3) * mbar
        else:
            gg = mbar if ada_kind == "shin" else g         # Shin: moments on frac. grad
            v = b2 * v + (1 - b2) * gg ** 2
            vh = v / (1 - b2 ** k)
            if ams:
                vmax = np.maximum(vmax, vh); vh = vmax
            if ada_kind == "shin":
                m = 0.9 * m + 0.1 * mbar
                x = x - lr * (m / (1 - 0.9 ** k)) / (np.sqrt(vh) + eps)
            else:
                x = x - lr * mbar / (np.sqrt(vh) + eps)
        out[k - 1] = loss(x)
    return out


METHODS = ["AdamW", "AdamW-b1=0.5", "AdamW-b1=0.8", "AdamW-b1=0.95", "AdamW-b1=0.98",
           "AdamW-b1=0.99", "AdamW-MM", "AMSGrad", "RMSProp", "MultiEMA-8", "Boxcar-MM",
           "F-Adam", "F-Adam-MM", "FracAdam-Shin", "lambda-FAdaMax", "FOSGD-ME", "TFGD",
           "TF-AdamW (exact)", "SOE-TF-AdamW", "SOE-TF-AdamW-AMS"]
LRGRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
LRGRID_TFGD = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

TUNE_INST = [500, 501, 502, 503]
EVAL_INST = list(range(20))
TUNE_SEEDS = [900, 901]
EVAL_SEEDS = [0, 1, 2]
NIT = 1500


def instance_grid(inst_seeds):
    """A distribution of problem instances (Reviewer 2 comment 19)."""
    combos = list(itertools.product([50, 100, 200], [1e2, 1e3, 1e4],
                                    ["log", "cluster", "powerlaw"], ["mult", "add"]))
    rng = np.random.default_rng(7)
    pick = rng.choice(len(combos), size=len(inst_seeds), replace=True)
    return [(inst_seeds[i], *combos[pick[i]]) for i in range(len(inst_seeds))]


def final_gap(curve, w=200):
    return float(np.mean(curve[-w:]))


def evaluate(method, insts, seeds, lr, nu=2.0):
    vals = []
    for (isd, p, cond, spec, noise) in insts:
        A, b, fstar, x0 = make_quadratic(p, cond, spec, isd)
        loss = lambda x: 0.5 * x @ A @ x - b @ x - fstar
        for s in seeds:
            rng = np.random.default_rng(1000 * isd + s)
            g = grad_fn(A, b, nu, noise, p, rng)
            c = run(method, g, loss, x0, NIT, lr, s)
            vals.append(max(final_gap(c), 1e-14))
    return np.array(vals)


def main():
    t0 = time.time()
    tune = instance_grid(TUNE_INST)
    ev = instance_grid(EVAL_INST)
    res = {}
    for mth in METHODS:
        grid = LRGRID_TFGD if mth in ("TFGD", "FOSGD-ME") else LRGRID
        best, bestv = None, np.inf
        for lr in grid:
            try:
                v = evaluate(mth, tune, TUNE_SEEDS, lr)
                sc = float(np.exp(np.mean(np.log(v))))        # geometric mean
            except Exception:
                sc = np.inf
            if np.isfinite(sc) and sc < bestv:
                bestv, best = sc, lr
        if best is None:
            best = grid[0]
        edge = "LOWER-EDGE" if best == grid[0] else ("UPPER-EDGE" if best == grid[-1] else "")
        v = evaluate(mth, ev, EVAL_SEEDS, best)
        res[mth] = dict(lr=best, gmean=float(np.exp(np.mean(np.log(v)))),
                        median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                        q75=float(np.percentile(v, 75)), vals=[float(x) for x in v])
        res[mth]["edge"] = edge
        print(f"{mth:20s} lr={best:<8g} gmean={res[mth]['gmean']:.3e} "
              f"median={res[mth]['median']:.3e} {edge:10s} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(meta=dict(alpha=ALPHA, lam=LAM, M=M, mu=MU, b1_mm=B1_MM,
                             boxcar_W=BOX_W, fadam_J_mm=FADAM_J_MM,
                             tune_inst=TUNE_INST, eval_inst=EVAL_INST,
                             tune_seeds=TUNE_SEEDS, eval_seeds=EVAL_SEEDS,
                             n_iter=NIT, instances=[list(map(str, x)) for x in ev]),
                   results=res), open(os.path.join(OUT, "synth_instances.json"), "w"),
              indent=1)
    print("TOTAL", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
