"""Optimizer zoo for the TF-AdamW revision.

Contains (a) the proposed methods, (b) the MEMORY-MATCHED controls demanded by the
reviewers (tuned-beta1 AdamW, equal-state multi-EMA, boxcar, matched-memory F-Adam),
(c) additional strong adaptive baselines, and (d) re-implementations of published
fractional optimizers.
"""
import math
import numpy as np
import torch
from torch.optim import Optimizer
import kernels as K


# ===================================================================== proposed
class SOETFAdamW(Optimizer):
    """SOE-TF-AdamW: AdamW with the first moment replaced by a normalized tempered
    fractional kernel realized by M exponential states."""

    def __init__(self, params, lr=1e-3, alpha=0.7, lam=0.05, M=8, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, amsgrad=False, fit=None):
        if fit is None:
            fit = K.soe_fit_nnls(alpha, lam, M)
        c = np.asarray(fit["c"], float)
        rho = np.asarray(fit["rho"], float)
        defaults = dict(lr=lr, beta2=beta2, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, c=c, rho=rho, alpha=alpha, lam=lam)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            ct = [float(x) for x in gr["c"]]
            rt = [float(x) for x in gr["rho"]]
            M = len(ct)
            b2 = gr["beta2"]; eps = gr["eps"]; lr = gr["lr"]
            wd = gr["weight_decay"]; ams = gr["amsgrad"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["k"] = 0
                    st["v"] = torch.zeros_like(p)
                    st["s"] = [torch.zeros_like(p) for _ in range(M)]
                    if ams:
                        st["vmax"] = torch.zeros_like(p)
                st["k"] += 1
                k = st["k"]
                m = torch.zeros_like(p)
                for i in range(M):
                    st["s"][i].mul_(rt[i]).add_(g)
                    m.add_(st["s"][i], alpha=ct[i])
                Dk = sum(ct[i] * (1 - rt[i] ** k) / (1 - rt[i]) for i in range(M))
                mbar = m / Dk
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                vhat = st["v"] / (1 - b2 ** k)
                if ams:
                    torch.maximum(st["vmax"], vhat, out=st["vmax"])
                    den = st["vmax"].sqrt().add_(eps)
                else:
                    den = vhat.sqrt().add_(eps)
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(mbar, den, value=-lr)
        return loss


class MultiEMAAdamW(Optimizer):
    """EQUAL-STATE CONTROL (Reviewer 2, comment 5): AdamW whose first moment is a
    normalized positive mixture of M generic exponential moving averages spanning
    comparable memory scales -- the same M-state budget as SOE-TF-AdamW, but WITHOUT
    the tempered-fractional shape."""

    def __init__(self, params, lr=1e-3, rho=None, c=None, M=8, mu_target=13.65,
                 kind="matched", beta2=0.999, eps=1e-8, weight_decay=0.0):
        if rho is None:
            rho, c = K.multiema_logspaced(M, mu_target, kind)
        defaults = dict(lr=lr, beta2=beta2, eps=eps, weight_decay=weight_decay,
                        c=np.asarray(c, float), rho=np.asarray(rho, float),
                        amsgrad=False, alpha=None, lam=None)
        super().__init__(params, defaults)

    step = SOETFAdamW.step


class KernelAdamW(Optimizer):
    """AdamW whose first moment is an arbitrary NONNEGATIVE causal kernel, realized by a
    ring buffer of the J most recent gradients and normalized by the running partial
    mass.  Used for (i) the untempered F-Adam baseline, (ii) matched-memory truncated
    kernels, and (iii) the BOXCAR (uniform sliding window) control."""

    def __init__(self, params, lr=1e-3, kernel=None, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        w = np.asarray(kernel, float)
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps,
                                      weight_decay=weight_decay, w=w))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            wt = [float(x) for x in gr["w"]]
            J = len(wt)
            b2 = gr["beta2"]; eps = gr["eps"]; lr = gr["lr"]; wd = gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["k"] = 0
                    st["v"] = torch.zeros_like(p)
                    st["buf"] = [torch.zeros_like(p) for _ in range(J)]
                    st["pos"] = 0
                st["k"] += 1
                k = st["k"]
                st["buf"][st["pos"]].copy_(g)
                st["pos"] = (st["pos"] + 1) % J
                L = min(k, J)
                m = torch.zeros_like(p)
                D = 0.0
                for j in range(L):
                    m.add_(st["buf"][(st["pos"] - 1 - j) % J], alpha=wt[j])
                    D += wt[j]
                mbar = m / D
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                den = (st["v"] / (1 - b2 ** k)).sqrt_().add_(eps)
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(mbar, den, value=-lr)
        return loss


# ============================================================ extra baselines
class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            b1, b2 = gr["betas"]; lr = gr["lr"]; wd = gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(st["m"].mul(b1).add(g, alpha=1 - b1).sign_(), alpha=-lr)
                st["m"].mul_(b2).add_(g, alpha=1 - b2)
        return loss


class AdaBelief(Optimizer):
    """AdaBelief (Zhuang et al., 2020): second moment tracks (g - m)^2."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-12,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            b1, b2 = gr["betas"]; lr = gr["lr"]; eps = gr["eps"]; wd = gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["k"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["s"] = torch.zeros_like(p)
                st["k"] += 1
                k = st["k"]
                if wd:
                    p.mul_(1 - lr * wd)
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                diff = g - st["m"]
                st["s"].mul_(b2).addcmul_(diff, diff, value=1 - b2).add_(eps)
                mh = st["m"] / (1 - b1 ** k)
                sh = st["s"] / (1 - b2 ** k)
                p.addcdiv_(mh, sh.sqrt().add_(eps), value=-lr)
        return loss


class Adafactor(Optimizer):
    """Adafactor (Shazeer & Stern, 2018): memory-efficient factored second moment.
    Factored for >=2-D parameters, full for 1-D; constant learning rate with RMS
    update clipping, matching the tuning protocol used for every other baseline."""

    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-30, weight_decay=0.0,
                 clip=1.0):
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps,
                                      weight_decay=weight_decay, clip=clip))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            lr = gr["lr"]; b2 = gr["beta2"]; eps = gr["eps"]; wd = gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                fac = g.dim() >= 2
                if not st:
                    st["k"] = 0
                    if fac:
                        st["r"] = torch.zeros(g.shape[:-1])
                        st["c"] = torch.zeros(g.shape[:-2] + g.shape[-1:])
                    else:
                        st["v"] = torch.zeros_like(p)
                st["k"] += 1
                k = st["k"]
                g2 = g * g + eps
                if fac:
                    st["r"].mul_(b2).add_(g2.mean(-1), alpha=1 - b2)
                    st["c"].mul_(b2).add_(g2.mean(-2), alpha=1 - b2)
                    rf = st["r"] / st["r"].mean(-1, keepdim=True).clamp_min(eps)
                    v = rf.unsqueeze(-1) * st["c"].unsqueeze(-2)
                else:
                    st["v"].mul_(b2).add_(g2, alpha=1 - b2)
                    v = st["v"]
                v = v / (1 - b2 ** k)
                u = g / v.sqrt().clamp_min(1e-16)
                rms = float(u.norm()) / max(1.0, math.sqrt(u.numel()))
                u = u / max(1.0, rms / gr["clip"])
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(u, alpha=-lr)
        return loss


class SAMWrapper:
    """Sharpness-Aware Minimization (Foret et al., 2021) around a base optimizer."""

    def __init__(self, base, params, rho=0.05):
        self.base = base
        self.params = [p for p in params]
        self.rho = rho
        self.e = []

    @torch.no_grad()
    def first_step(self):
        gs = [p.grad.norm() for p in self.params if p.grad is not None]
        gn = torch.norm(torch.stack(gs)) if gs else torch.tensor(0.0)
        scale = self.rho / (gn + 1e-12)
        self.e = []
        for p in self.params:
            if p.grad is None:
                self.e.append(None)
                continue
            ew = p.grad * scale
            p.add_(ew)
            self.e.append(ew)

    @torch.no_grad()
    def second_step(self):
        for p, ew in zip(self.params, self.e):
            if ew is not None:
                p.sub_(ew)
        self.base.step()

    def zero_grad(self):
        self.base.zero_grad()


# ================================================== published fractional methods
class FracAdamShin(Optimizer):
    """Fractional Adam after Shin, Darbon & Karniadakis (2023): Adam in which the
    gradient is replaced by a Caputo fractional derivative approximated by a truncated
    Gruenwald-Letnikov series, then fed through the standard Adam moments."""

    def __init__(self, params, lr=1e-3, alpha=0.9, J=8, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        cg = np.empty(J)
        cg[0] = 1.0
        for j in range(1, J):
            cg[j] = cg[j - 1] * (j - 1 - alpha) / j
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, cg=cg))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            cg = [float(x) for x in gr["cg"]]
            J = len(cg)
            b1, b2 = gr["betas"]; lr = gr["lr"]; eps = gr["eps"]; wd = gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["k"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["buf"] = [torch.zeros_like(p) for _ in range(J)]
                    st["pos"] = 0
                st["k"] += 1
                k = st["k"]
                st["buf"][st["pos"]].copy_(g)
                st["pos"] = (st["pos"] + 1) % J
                gf = torch.zeros_like(p)
                for j in range(min(k, J)):
                    gf.add_(st["buf"][(st["pos"] - 1 - j) % J], alpha=cg[j])
                st["m"].mul_(b1).add_(gf, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(gf, gf, value=1 - b2)
                mh = st["m"] / (1 - b1 ** k)
                vh = st["v"] / (1 - b2 ** k)
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)
        return loss


class LambdaFAdaMax(Optimizer):
    """lambda-FAdaMax after Chen et al. (2025): AdaMax (infinity-norm second moment)
    with a fractional power-law first moment and a decayed second moment."""

    def __init__(self, params, lr=2e-3, alpha=0.7, J=32, beta1=0.9, beta2=0.999,
                 lam_decay=0.999, eps=1e-8, weight_decay=0.0):
        w = K.w_alpha(alpha, J - 1)
        super().__init__(params, dict(lr=lr, betas=(beta1, beta2), eps=eps,
                                      weight_decay=weight_decay, w=w, lam=lam_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            wt = [float(x) for x in gr["w"]]
            J = len(wt)
            b1, b2 = gr["betas"]; lr = gr["lr"]; eps = gr["eps"]
            wd = gr["weight_decay"]; lam = gr["lam"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["k"] = 0
                    st["u"] = torch.zeros_like(p)
                    st["buf"] = [torch.zeros_like(p) for _ in range(J)]
                    st["pos"] = 0
                st["k"] += 1
                k = st["k"]
                st["buf"][st["pos"]].copy_(g)
                st["pos"] = (st["pos"] + 1) % J
                L = min(k, J)
                m = torch.zeros_like(p)
                D = 0.0
                for j in range(L):
                    m.add_(st["buf"][(st["pos"] - 1 - j) % J], alpha=wt[j])
                    D += wt[j]
                mbar = m / D
                st["u"].mul_(lam * b2)
                torch.maximum(st["u"], g.abs(), out=st["u"])
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(mbar, st["u"].add(eps), value=-lr / (1 - b1 ** k))
        return loss


class FOSGDME(Optimizer):
    """FOSGD-ME after Zhou et al. (2025): fractional-order SGD with momentum and an
    adaptive scalar energy variable that damps large updates."""

    def __init__(self, params, lr=1e-2, alpha=0.7, J=32, eps=1e-8, weight_decay=0.0):
        w = K.w_alpha(alpha, J - 1)
        super().__init__(params, dict(lr=lr, w=w, eps=eps, weight_decay=weight_decay))
        self._r = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        tot = 0.0
        lr0 = self.param_groups[0]["lr"]
        for gr in self.param_groups:
            wt = [float(x) for x in gr["w"]]
            J = len(wt)
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["k"] = 0
                    st["buf"] = [torch.zeros_like(p) for _ in range(J)]
                    st["pos"] = 0
                st["k"] += 1
                k = st["k"]
                st["buf"][st["pos"]].copy_(g)
                st["pos"] = (st["pos"] + 1) % J
                L = min(k, J)
                m = torch.zeros_like(p)
                D = 0.0
                for j in range(L):
                    m.add_(st["buf"][(st["pos"] - 1 - j) % J], alpha=wt[j])
                    D += wt[j]
                st["upd"] = m / D
                tot += float(st["upd"].pow(2).sum())
        self._r = self._r / (1.0 + lr0 * tot)
        scale = max(self._r, 1e-3)
        for gr in self.param_groups:
            lr = gr["lr"]; wd = gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(self.state[p]["upd"], alpha=-lr * scale)
        return loss


class TFGD(Optimizer):
    """TFGD (Naifar, 2026): tempered fractional GRADIENT DESCENT -- the normalized
    tempered fractional first moment with a constant (non-adaptive) denominator."""

    def __init__(self, params, lr=1e-2, alpha=0.7, lam=0.05, M=8, fit=None,
                 weight_decay=0.0):
        if fit is None:
            fit = K.soe_fit_nnls(alpha, lam, M)
        super().__init__(params, dict(lr=lr, c=np.asarray(fit["c"], float),
                                      rho=np.asarray(fit["rho"], float),
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gr in self.param_groups:
            ct = [float(x) for x in gr["c"]]
            rt = [float(x) for x in gr["rho"]]
            M = len(ct); lr = gr["lr"]; wd = gr["weight_decay"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["k"] = 0
                    st["s"] = [torch.zeros_like(p) for _ in range(M)]
                st["k"] += 1
                k = st["k"]
                m = torch.zeros_like(p)
                for i in range(M):
                    st["s"][i].mul_(rt[i]).add_(g)
                    m.add_(st["s"][i], alpha=ct[i])
                Dk = sum(ct[i] * (1 - rt[i] ** k) / (1 - rt[i]) for i in range(M))
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(m / Dk, alpha=-lr)
        return loss
