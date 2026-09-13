"""Kernel geometry under MATCHED memory (Reviewer 2, comment 4).

Compares the tempered fractional kernel with the exponential (AdamW) kernel on the
three quantities the reviewer asks for -- weight energy sum_j p_j^2, effective sample
size ESS = 1/sum_j p_j^2, and mean delay mu -- plus the head weight p_0, which turns
out to be the quantity that actually separates the two families.

Key exact identities (verified numerically here):
    exponential:  mu_E = b/(1-b),  ESS_E = (1+b)/(1-b) = 1 + 2 mu_E     (a CURVE)
    tempered frac: mu   = alpha/(e^lam - 1),  d = (1-e^-lam)^-alpha,
                   sum_j kappa_j^2 = 2F1(alpha,alpha;1;e^{-2 lam}),
                   ESS = d^2 / 2F1(alpha,alpha;1;e^{-2 lam}),
                   p_0 = 1/d                                             (a REGION)
"""
import json, os
import numpy as np
from scipy.special import hyp2f1, gamma as G
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kernels as K

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8.5,
                     "lines.linewidth": 1.7, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif",
                     "mathtext.fontset": "cm"})


def verify_identities():
    rows = []
    for a in (0.3, 0.5, 0.7, 0.9):
        for lam in (0.02, 0.05, 0.1, 0.3):
            J = int(max(60000, 40 / lam))
            kap = K.tempered_kernel(a, lam, J)
            brute_d = kap.sum()
            brute_s2 = (kap ** 2).sum()
            brute_mu = (np.arange(J + 1) * kap).sum() / brute_d
            cf_d = K.d_mass(a, lam)
            cf_s2 = hyp2f1(a, a, 1.0, np.exp(-2 * lam))
            cf_mu = K.mean_delay_tf(a, lam)
            rows.append(dict(alpha=a, lam=lam,
                             rel_d=abs(brute_d / cf_d - 1),
                             rel_s2=abs(brute_s2 / cf_s2 - 1),
                             rel_mu=abs(brute_mu / cf_mu - 1),
                             d=cf_d, mu=cf_mu, ess=cf_d ** 2 / cf_s2, p0=1 / cf_d))
    return rows


def asymptotic_ratio(a):
    """lim_{lam->0} ESS_TF / ESS_EMA at matched mean delay."""
    if a > 0.5:
        C = 2 ** (2 * a - 1) * G(a) ** 2 / G(2 * a - 1)
        return C / (2 * a)
    return np.inf if a < 0.5 else np.nan     # a<0.5: ESS_TF/ESS_EMA -> 0 (see note)


def matched_table():
    """At matched MEAN DELAY and at matched HEAD WEIGHT p_0."""
    rows = []
    for a in (0.3, 0.5, 0.7, 0.9):
        for lam in (0.02, 0.05, 0.1, 0.2):
            mu = K.mean_delay_tf(a, lam)
            d = K.d_mass(a, lam)
            ess = K.ess_tf(a, lam)
            p0 = 1.0 / d
            # (i) exponential matched in MEAN DELAY
            b_mu = K.beta_for_mean_delay(mu)
            ess_e_mu = K.ess_ema(b_mu)
            # (ii) exponential matched in HEAD WEIGHT (p_0 = 1-b)
            b_p0 = 1.0 - p0
            ess_e_p0 = K.ess_ema(b_p0)
            mu_e_p0 = K.mean_delay_ema(b_p0)
            rows.append(dict(alpha=a, lam=lam, mu=mu, d=d, ess=ess, p0=p0,
                             beta_mu=b_mu, ess_ema_mu=ess_e_mu,
                             ratio_ess_at_matched_mu=ess / ess_e_mu,
                             beta_p0=b_p0, mu_ema_p0=mu_e_p0, ess_ema_p0=ess_e_p0,
                             ratio_ess_at_matched_p0=ess / ess_e_p0))
    return rows


def figure():
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.5))

    # ---- (a) the two kernels at matched mean delay
    a, lam = 0.7, 0.05
    mu = K.mean_delay_tf(a, lam)
    J = 120
    j = np.arange(J + 1)
    kap = K.tempered_kernel(a, lam, J); p_tf = kap / K.d_mass(a, lam)
    b = K.beta_for_mean_delay(mu); p_e = (1 - b) * b ** j
    rho_m, c_m = K.multiema_logspaced(8, mu, "matched")
    p_me = sum(c_m[i] * rho_m[i] ** j for i in range(len(c_m)))
    W = K.boxcar_window_for_mean_delay(mu)
    p_bx = np.where(j < W, 1.0 / W, 0.0)
    ax[0].semilogy(j, p_tf, "-", color="#d62728", label=r"tempered fractional")
    ax[0].semilogy(j, p_e, "--", color="#1f77b4", label=r"exponential (AdamW)")
    ax[0].semilogy(j, p_me, "-.", color="#2ca02c", label=r"multi-EMA ($M{=}8$)")
    ax[0].semilogy(j, np.where(p_bx > 0, p_bx, np.nan), ":", color="#7f7f7f",
                   label=r"boxcar")
    ax[0].set_xlabel(r"lag $j$"); ax[0].set_ylabel(r"normalized weight $p_j$")
    ax[0].set_ylim(1e-5, 0.3)
    ax[0].set_title(r"(a) matched mean delay $\mu=%.1f$" % mu)
    ax[0].legend(frameon=False, loc="upper right")

    # ---- (b) the (mu, ESS) plane: exponential CURVE vs tempered-fractional REGION
    mus = np.geomspace(0.5, 200, 200)
    ax[1].loglog(mus, 1 + 2 * mus, "k--", lw=2, label=r"exponential: $\mathrm{ESS}=1+2\mu$")
    for aa, cc in zip((0.3, 0.5, 0.7, 0.9), ("#9467bd", "#2ca02c", "#d62728", "#ff7f0e")):
        lams = np.geomspace(1e-3, 2.0, 200)
        mm = K.mean_delay_tf(aa, lams); ee = K.ess_tf(aa, lams)
        ax[1].loglog(mm, ee, "-", color=cc, label=rf"TF, $\alpha={aa}$")
    ax[1].set_xlabel(r"mean delay $\mu$"); ax[1].set_ylabel(r"$\mathrm{ESS}=1/\sum_j p_j^2$")
    ax[1].set_title(r"(b) variance reduction vs. staleness")
    ax[1].legend(frameon=False, fontsize=7.5)

    # ---- (c) the (p_0, ESS) plane: TF region lies ABOVE the exponential curve
    b_ = 1 - np.geomspace(1e-3, 0.9, 300)
    ax[2].loglog(1 - b_, K.ess_ema(b_), "k--", lw=2,
                 label=r"exponential: $\mathrm{ESS}=\frac{2}{p_0}-1$")
    for aa, cc in zip((0.3, 0.5, 0.7, 0.9), ("#9467bd", "#2ca02c", "#d62728", "#ff7f0e")):
        lams = np.geomspace(1e-3, 2.0, 300)
        p0 = 1.0 / K.d_mass(aa, lams); ee = K.ess_tf(aa, lams)
        ax[2].loglog(p0, ee, "-", color=cc, label=rf"TF, $\alpha={aa}$")
    ax[2].set_xlabel(r"head weight $p_0$ (responsiveness)")
    ax[2].set_ylabel(r"$\mathrm{ESS}$")
    ax[2].set_title(r"(c) responsiveness vs. averaging")
    ax[2].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_geometry.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    v = verify_identities()
    worst = max(max(r["rel_d"], r["rel_s2"], r["rel_mu"]) for r in v)
    print(f"closed-form identities: worst relative error over 16 (alpha,lambda) pairs = {worst:.2e}")
    t = matched_table()
    print(f"\n{'alpha':>5} {'lam':>5} {'mu':>7} {'ESS_TF':>8} {'p0_TF':>7} "
          f"{'ESS_EMA@mu':>11} {'ratio':>6} | {'mu_EMA@p0':>9} {'ESS_EMA@p0':>11} {'ratio':>6}")
    for r in t:
        print(f"{r['alpha']:5.1f} {r['lam']:5.2f} {r['mu']:7.2f} {r['ess']:8.2f} "
              f"{r['p0']:7.4f} {r['ess_ema_mu']:11.2f} {r['ratio_ess_at_matched_mu']:6.3f} | "
              f"{r['mu_ema_p0']:9.2f} {r['ess_ema_p0']:11.2f} {r['ratio_ess_at_matched_p0']:6.3f}")
    print("\nasymptotic ESS_TF/ESS_EMA at matched mu, lam->0:")
    for a in (0.6, 0.7, 0.8, 0.9, 0.99):
        print(f"  alpha={a}: {asymptotic_ratio(a):.4f}")
    figure()
    json.dump(dict(identities=v, matched=t), open(os.path.join(OUT, "geometry.json"), "w"),
              indent=1, default=float)
    print("\nwrote results/geometry.json and results/fig_geometry.pdf")
