"""Kernel geometry, SOE fitting with a CERTIFIED l1 error bound, and matched-memory
kernel construction.  Used by every experiment and by the manuscript tables."""
import numpy as np
from scipy.special import gamma as Gamma, gammaincc, hyp2f1
from scipy.optimize import nnls, least_squares


# ---------------------------------------------------------------- kernels
def w_alpha(alpha, J):
    """Generalized binomial coefficients w_j^{(alpha)}, j=0..J."""
    a = np.empty(J + 1); a[0] = 1.0
    for j in range(1, J + 1):
        a[j] = a[j - 1] * (j - 1 + alpha) / j
    return a


def tempered_kernel(alpha, lam, J):
    return w_alpha(alpha, J) * np.exp(-lam * np.arange(J + 1))


def d_mass(alpha, lam):
    """Exact total mass d(alpha,lambda) = (1-e^{-lam})^{-alpha}."""
    return (1.0 - np.exp(-lam)) ** (-alpha)


def mean_delay_tf(alpha, lam):
    """mu(alpha,lambda) = alpha/(e^{lam}-1)."""
    return alpha / (np.expm1(lam))


def ess_tf(alpha, lam):
    """Stationary effective sample size of the normalized tempered fractional kernel:
    ESS = d^2 / sum_j kappa_j^2, with sum_j kappa_j^2 = 2F1(a,a;1;e^{-2 lam})."""
    s2 = hyp2f1(alpha, alpha, 1.0, np.exp(-2.0 * lam))
    return d_mass(alpha, lam) ** 2 / s2


def ess_ema(beta):
    """ESS of the normalized exponential kernel (1-b)b^j is (1+b)/(1-b) = 1 + 2*mu."""
    return (1.0 + beta) / (1.0 - beta)


def mean_delay_ema(beta):
    return beta / (1.0 - beta)


def beta_for_mean_delay(mu):
    """beta_1 giving mean delay mu:  mu = b/(1-b)  =>  b = mu/(1+mu)."""
    return mu / (1.0 + mu)


def lam_for_mean_delay(alpha, mu):
    """lambda giving mean delay mu at fixed alpha:  mu = alpha/(e^lam - 1)."""
    return np.log1p(alpha / mu)


# ------------------------------------------------- SOE fit + certified error
def soe_fit_nnls(alpha, lam, M, J=4000, seed=0, n_restart=5):
    """Fit kappa_j ~ sum_m c_m rho_m^j with GUARANTEED c_m >= 0 (NNLS) and rho_m in (0,1)."""
    kappa = tempered_kernel(alpha, lam, J); j = np.arange(J + 1)
    rng = np.random.default_rng(seed)

    def unpack(t): return 1.0 / (1.0 + np.exp(-t))

    def solve_c(rho):
        Phi = rho[None, :] ** j[:, None]
        c, _ = nnls(Phi, kappa)
        return Phi, c

    def residual(t):
        rho = unpack(t); Phi, c = solve_c(rho)
        return Phi @ c - kappa

    init = np.clip(1 - np.geomspace(1e-4, 0.9, M), 1e-4, 1 - 1e-5)
    best = None
    for trial in range(n_restart):
        if trial == 0:
            t0 = np.log(init / (1 - init))
        else:
            r = np.clip(rng.uniform(0.05, 0.999, M), 1e-3, 1 - 1e-4)
            t0 = np.log(r / (1 - r))
        try:
            sol = least_squares(residual, t0, method="lm", max_nfev=4000)
        except Exception:
            continue
        rho = unpack(sol.x); Phi, c = solve_c(rho)
        err = float(np.sum(np.abs(Phi @ c - kappa)))
        if best is None or err < best[0]:
            best = (err, rho.copy(), c.copy())
    err, rho, c = best
    # keep rho strictly inside (0,1): the logit parameterization can saturate to
    # exactly 1.0 in floating point at small M, which would divide by zero below
    rho = np.clip(rho, 1e-12, 1.0 - 1e-9)
    keep = c > 1e-14
    rho, c = rho[keep], c[keep]
    out = dict(rho=rho, c=c, eps_fit=err, J=J, alpha=alpha, lam=lam, M=M)
    out.update(certified_eps(alpha, lam, rho, c, J, err))
    return out


def kernel_tail_bound(alpha, lam, J):
    """Rigorous upper bound on sum_{j>J} kappa_j = sum_{j>J} w_j^{(a)} e^{-lam j}.

    Uses w_j^{(a)} <= j^{a-1}/Gamma(a) for j>=1, a in (0,1) (verified numerically in
    verify_tail_inequality()), then bounds the sum by the integral
    int_J^inf x^{a-1} e^{-lam x} dx = lam^{-a} Gamma(a) Q(a, lam J)
    where Q is the regularized upper incomplete gamma function.  Because
    x^{a-1}e^{-lam x} is decreasing for a<=1, sum_{j>J} f(j) <= int_J^inf f(x) dx."""
    return float(lam ** (-alpha) * gammaincc(alpha, lam * J))


def certified_eps(alpha, lam, rho, c, J, eps_fit):
    """eps_SOE <= (fitted l1 error on j<=J) + (kernel tail bound) + (SOE tail, exact)."""
    ker_tail = kernel_tail_bound(alpha, lam, J)
    soe_tail = float(np.sum(c * rho ** (J + 1) / (1.0 - rho)))
    return dict(eps_kernel_tail=ker_tail, eps_soe_tail=soe_tail,
                eps_certified=eps_fit + ker_tail + soe_tail)


def verify_tail_inequality(alphas=(0.1, 0.3, 0.5, 0.7, 0.9, 0.99), J=200000):
    """Check w_j^{(a)} <= j^{a-1}/Gamma(a) for all 1<=j<=J."""
    bad = []
    for a in alphas:
        w = w_alpha(a, J)[1:]
        ub = np.arange(1, J + 1) ** (a - 1) / Gamma(a)
        if np.any(w > ub * (1 + 1e-12)):
            bad.append((a, int(np.argmax(w - ub)) + 1, float(np.max(w - ub))))
    return bad


# ------------------------------------------------- matched multi-EMA / boxcar
def multiema_logspaced(M, mu_target, kind="uniform"):
    """A generic bank of M exponential timescales spanning comparable memory scales,
    with weights c_m normalized so the mixture kernel has TOTAL MASS 1 and, for
    kind='matched', MEAN DELAY equal to mu_target.

    Returns (rho, c) with c>0.  This is the 'equal-state multi-EMA' control that
    Reviewer 2 (comment 5) asks for: same M-state budget, generic timescales,
    NOT a tempered-fractional shape."""
    # geometric ladder of mean delays from ~1 to ~4*mu_target
    mus = np.geomspace(max(0.5, mu_target / 8.0), max(1.0, 4.0 * mu_target), M)
    rho = mus / (1.0 + mus)
    if kind == "uniform":
        c = np.ones(M) / M * (1 - rho)          # each component normalized then averaged
        c = c / np.sum(c / (1 - rho))            # total mass 1
        return rho, c
    # 'matched': choose nonnegative c to hit total mass 1 and mean delay mu_target
    # mass_m = 1/(1-rho_m) ; meandelay contribution = rho_m/(1-rho_m)^2
    A = np.vstack([1.0 / (1.0 - rho), rho / (1.0 - rho) ** 2])
    b = np.array([1.0, mu_target])
    c, _ = nnls(A, b)
    if np.sum(c) <= 0:
        return multiema_logspaced(M, mu_target, "uniform")
    return rho, c


def boxcar_window_for_mean_delay(mu):
    """Uniform (boxcar) average over W most recent gradients has mean delay (W-1)/2."""
    return max(1, int(round(2 * mu + 1)))


def kernel_stats(kap):
    """(mass, mean delay, ESS, p0, weight energy) of a nonnegative kernel array."""
    kap = np.asarray(kap, float); D = kap.sum()
    p = kap / D
    return dict(mass=float(D), mu=float(np.sum(np.arange(len(kap)) * p)),
                ess=float(1.0 / np.sum(p ** 2)), p0=float(p[0]),
                energy=float(np.sum(p ** 2)))
