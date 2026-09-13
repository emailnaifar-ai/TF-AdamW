"""Gradient statistics under increasing label corruption (Reviewer 2, comment 12).

Proposition 5.4 concerns independent zero-mean GRADIENT noise, whereas symmetric label
corruption perturbs the objective itself.  To test whether the label-noise robustness is
actually mediated by the predicted filtering effect, we measure, during real training:

  * per-step mini-batch gradient variance     tr Cov(g) estimated from micro-batches,
  * the first-moment signal-to-noise ratio    ||E[mbar]|| / sqrt(tr Cov(mbar)),
  * cosine similarity between consecutive stochastic gradients,
  * the MEASURED variance-reduction factor    tr Cov(mbar) / tr Cov(g),
    which is exactly the quantity sum_j p_{k,j}^2 that Proposition 5.4 bounds.

The last quantity is the direct empirical test: theory predicts it should equal the
kernel weight energy sum_j p_j^2, and should be smaller for the tempered fractional
first moment than for AdamW's EMA only when the two are compared at matched
RESPONSIVENESS, not at matched mean delay.
"""
import os, json, time
import numpy as np
import torch
import torch.nn.functional as F

import kernels as K
import dl_core as C

torch.set_num_threads(2)
OUT = "results"
os.makedirs(OUT, exist_ok=True)

CTX = C.make_ctx()
MICRO = 8                    # micro-batches per step, used to estimate Cov(g)
METHODS = ["AdamW", "AdamW-MM", "SOE-TF-AdamW"]
NOISES = [0.0, 0.2, 0.4, 0.6]


def flat(model, which="grad"):
    vs = []
    for p in model.parameters():
        if p.grad is None:
            continue
        vs.append((p.grad if which == "grad" else p).reshape(-1))
    return torch.cat(vs)


def first_moment(opt, model):
    """Extract the normalized first moment mbar actually used by the optimizer."""
    vs = []
    for p in model.parameters():
        st = opt.state.get(p, {})
        if "s" in st:                                   # SOE / multi-EMA bank
            c = opt.param_groups[0]["c"]; rho = opt.param_groups[0]["rho"]
            k = st["k"]
            m = sum(float(c[i]) * st["s"][i] for i in range(len(c)))
            D = sum(float(c[i]) * (1 - float(rho[i]) ** k) / (1 - float(rho[i]))
                    for i in range(len(c)))
            vs.append((m / D).reshape(-1))
        elif "exp_avg" in st:                            # torch Adam/AdamW
            b1 = opt.param_groups[0]["betas"][0]
            k = int(st["step"]) if not torch.is_tensor(st["step"]) else int(st["step"].item())
            vs.append((st["exp_avg"] / (1 - b1 ** k)).reshape(-1))
        else:
            return None
    return torch.cat(vs) if vs else None


def measure(method, noise, seed=0, epochs=4, probe_every=40):
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr), _, (Xte, yte) = C.load_fmnist(12000, 3000, 5000)
    ytr = C.add_label_noise(ytr, noise, 10, seed)
    X = torch.tensor(Xtr).view(-1, 1, 28, 28); Y = torch.tensor(ytr)
    model = C.SmallCNN()
    opt = C.make_opt(method, model.parameters(), 1e-3, CTX)
    n = len(X); idx = np.arange(n); rng = np.random.default_rng(seed)
    B = 128
    gvar, mvar, ratio, cos, snr = [], [], [], [], []
    prev = None
    step = 0
    for ep in range(epochs):
        model.train(); rng.shuffle(idx)
        for b in range(0, n, B):
            bi = idx[b:b + B]
            probe = (step % probe_every == 0) and step > 20
            if probe:
                # estimate Cov(g) from MICRO independent micro-batches
                gs = []
                for t in range(MICRO):
                    sub = bi[t::MICRO]
                    if len(sub) < 4:
                        continue
                    opt.zero_grad()
                    F.cross_entropy(model(X[sub]), Y[sub]).backward()
                    gs.append(flat(model).clone())
                if len(gs) >= 4:
                    Gm = torch.stack(gs)
                    gv = float(Gm.var(0, unbiased=True).sum())
                    gmean = Gm.mean(0)
                    gvar.append(gv)
                    if prev is not None:
                        cos.append(float(F.cosine_similarity(gmean, prev, dim=0)))
                    prev = gmean.clone()
            opt.zero_grad()
            F.cross_entropy(model(X[bi]), Y[bi]).backward()
            g = flat(model).clone()
            opt.step()
            if probe:
                mb = first_moment(opt, model)
                if mb is not None and gvar:
                    # weight energy actually realized by the filter, measured as the
                    # ratio of the filtered to the raw fluctuation about a local mean
                    mvar.append(float(mb.var()))
                    ratio.append(float(mb.var() / max(g.var(), 1e-20)))
                    snr.append(float(mb.norm() / (mb.std() * np.sqrt(len(mb)) + 1e-12)))
            step += 1
    return dict(grad_var=float(np.mean(gvar)) if gvar else None,
                filt_over_raw_var=float(np.mean(ratio)) if ratio else None,
                cos_consec=float(np.mean(cos)) if cos else None,
                snr=float(np.mean(snr)) if snr else None,
                n_probe=len(ratio))


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    # theoretical weight energies for reference
    mu = CTX["mu"]
    theo = {"SOE-TF-AdamW": 1.0 / K.ess_tf(0.7, 0.05),
            "AdamW": 1.0 / K.ess_ema(0.9),
            "AdamW-MM": 1.0 / K.ess_ema(K.beta_for_mean_delay(mu))}
    for m in METHODS:
        for nz in NOISES:
            r = measure(m, nz)
            r["theoretical_weight_energy"] = theo[m]
            res[f"{m}|{nz}"] = r
            print(f"{m:14s} noise={nz:.1f}  gradvar={r['grad_var']:.3e}  "
                  f"var(mbar)/var(g)={r['filt_over_raw_var']:.4f}  "
                  f"(theory sum p^2={theo[m]:.4f})  cos={r['cos_consec']:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(os.path.join(OUT, "gradstats.json"), "w"), indent=1)
    print("wrote results/gradstats.json")
