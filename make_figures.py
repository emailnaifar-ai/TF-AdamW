"""Regenerate every figure of the revised manuscript from the stored result files."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kernels as K

RES = "results"
FIG = os.environ.get("TFADAMW_FIGURES",
                     os.path.join(os.path.dirname(
                         os.path.abspath(__file__)), "figures"))
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8.5,
                     "lines.linewidth": 1.7, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif",
                     "mathtext.fontset": "cm"})

COL = {"SOE-TF-AdamW": "#d62728", "TF-AdamW (exact)": "#9467bd",
       "AdamW": "#1f77b4", "AdamW-MM": "#2ca02c", "MultiEMA-8": "#ff7f0e",
       "Boxcar-MM": "#7f7f7f", "F-Adam": "#17becf", "F-Adam-MM": "#8c564b",
       "AMSGrad": "#e377c2", "RMSProp": "#bcbd22"}


def load(f):
    p = os.path.join(RES, f)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ Fig: instances
def fig_instances():
    d = load("synth_instances.json")
    if not d:
        return print("skip fig_instances")
    R = d["results"]
    show = ["SOE-TF-AdamW", "MultiEMA-8", "AdamW-MM", "AdamW", "AMSGrad", "RMSProp",
            "F-Adam-MM", "Boxcar-MM"]
    show = [m for m in show if m in R]
    b1 = [m for m in R if m.startswith("AdamW-b1=")]
    best_b1 = min(b1, key=lambda m: R[m]["gmean"]) if b1 else None
    if best_b1:
        show.insert(3, best_b1)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.8))
    data = [np.log10(np.array(R[m]["vals"])) for m in show]
    bp = ax[0].boxplot(data, vert=True, patch_artist=True, widths=0.6,
                       medianprops=dict(color="k"))
    for patch, m in zip(bp["boxes"], show):
        patch.set_facecolor(COL.get(m.split("-b1")[0], "#cccccc"))
        patch.set_alpha(0.55)
    ax[0].set_xticks(range(1, len(show) + 1))
    ax[0].set_xticklabels([m.replace("AdamW-b1=", r"AdamW $\beta_1$=") for m in show],
                          rotation=38, ha="right", fontsize=8)
    ax[0].set_ylabel(r"$\log_{10}(f-f^\star)$, final window")
    ax[0].set_title(r"(a) 20 problem instances $\times$ 3 unseen seeds")
    # (b) shape experiment: performance vs head weight at fixed mean delay
    s = load("synth_shape.json")
    if s:
        R2 = s["results"]
        names = list(R2)
        p0 = np.array([R2[n]["p0"] for n in names])
        gm = np.array([R2[n]["gmean"] for n in names])
        es = np.array([R2[n]["ess"] for n in names])
        o = np.argsort(p0)
        ax[1].loglog(p0[o], gm[o], "o-", color="#d62728")
        for i in o:
            ax[1].annotate(names[i].replace("tempered fractional ", "TF ")
                           .replace("exponential (AdamW)", "AdamW")
                           .replace("multi-EMA (M=8)", "multi-EMA"),
                           (p0[i], gm[i]), fontsize=6.5,
                           textcoords="offset points", xytext=(4, 4))
        ax[1].set_xlabel(r"head weight $p_0$ (at fixed mean delay $\mu=13.65$)")
        ax[1].set_ylabel(r"final $f-f^\star$ (geometric mean)")
        ax[1].set_title(r"(b) kernel shape at fixed staleness")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_instances.pdf"))
    plt.close(fig)
    print("wrote fig_instances.pdf")


# ------------------------------------------------------------------ Fig: DL stats
def fig_dl(exp="fmnist"):
    d = load(f"stats_{exp}.json")
    if not d:
        return print(f"skip fig_dl({exp})")
    noises = sorted(d, key=float)
    fig, ax = plt.subplots(1, len(noises), figsize=(5.6 * len(noises), 4.0))
    if len(noises) == 1:
        ax = [ax]
    for a, nz in zip(ax, noises):
        rows = d[nz]
        rows = sorted(rows, key=lambda r: r["diff"])
        y = np.arange(len(rows))
        diff = np.array([r["diff"] for r in rows]) * 100
        lo = np.array([r["ci"][0] for r in rows]) * 100
        hi = np.array([r["ci"][1] for r in rows]) * 100
        cols = ["#d62728" if "TF-AdamW" in r["method"] else "#1f77b4" for r in rows]
        a.errorbar(diff, y, xerr=[diff - lo, hi - diff], fmt="o", ms=4, lw=1.2,
                   ecolor="#888888", mfc="none", ls="none")
        a.scatter(diff, y, c=cols, s=26, zorder=3)
        a.axvline(0, color="k", lw=1, ls="--")
        a.set_yticks(y)
        a.set_yticklabels([r["method"] for r in rows], fontsize=7.5)
        a.set_xlabel("paired difference vs AdamW (percentage points)")
        a.set_title(f"label noise {float(nz)*100:.0f}\\%  ($n$={rows[0]['n']} paired seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_{exp}_paired.pdf"))
    plt.close(fig)
    print(f"wrote fig_{exp}_paired.pdf")


# ------------------------------------------------------------------ Fig: heat map
def fig_heat():
    import analyze as A
    D = A.table("heat")
    if not D:
        return print("skip fig_heat")
    cells = {}
    for m, byn in D.items():
        if "al=" not in m:
            continue
        a, l = m.split("al=")[1].split("_")
        for nz, seeds in byn.items():
            cells[(float(a), float(l))] = np.mean(list(seeds.values()))
    if not cells:
        return print("skip fig_heat (no cells)")
    A_ = sorted({k[0] for k in cells}); L_ = sorted({k[1] for k in cells})
    Z = np.array([[cells.get((a, l), np.nan) * 100 for l in L_] for a in A_])
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7))
    im = ax[0].imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax[0].set_xticks(range(len(L_))); ax[0].set_xticklabels(L_)
    ax[0].set_yticks(range(len(A_))); ax[0].set_yticklabels(A_)
    ax[0].set_xlabel(r"tempering $\lambda$"); ax[0].set_ylabel(r"fractional order $\alpha$")
    ax[0].set_title(r"(a) test accuracy (\%), Fashion-MNIST, 40\% label noise")
    for i in range(len(A_)):
        for j in range(len(L_)):
            if np.isfinite(Z[i, j]):
                ax[0].text(j, i, f"{Z[i,j]:.1f}", ha="center", va="center",
                           fontsize=7, color="w")
    fig.colorbar(im, ax=ax[0])
    # same data against the derived memory window
    mus = np.array([[K.mean_delay_tf(a, l) for l in L_] for a in A_])
    ax[1].semilogx(mus.ravel(), Z.ravel(), "o", ms=5, color="#d62728")
    ax[1].set_xlabel(r"mean memory length $\mu(\alpha,\lambda)$")
    ax[1].set_ylabel(r"test accuracy (\%)")
    ax[1].set_title(r"(b) accuracy against the derived memory window")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_heat.pdf"))
    plt.close(fig)
    print("wrote fig_heat.pdf")


# ------------------------------------------------------------------ Fig: SOE error
def fig_soe_certified():
    d = load("soe_error.json")
    if not d or "table" not in d:
        return print("skip fig_soe_certified")
    rows = d["table"]
    M = np.array([r["M"] for r in rows]); E = np.array([r["cert"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.semilogy(M, E, "o-", color="#d62728", label="certified $\\varepsilon_{\\mathrm{SOE}}$")
    ok = E > 1e-14
    c = np.polyfit(M[ok], np.log(E[ok]), 1)
    ax.semilogy(M, np.exp(np.polyval(c, M)), "k--", lw=1,
                label=rf"fit $\exp({c[1]:.2f}{c[0]:+.2f}M)$")
    ax.set_xlabel("number of exponentials $M$")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{SOE}}$ (certified upper bound)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_soe_certified.pdf"))
    plt.close(fig)
    print("wrote fig_soe_certified.pdf")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["instances", "dl", "heat", "soe"]
    if "instances" in which: fig_instances()
    if "dl" in which:
        for e in ("fmnist", "cifar", "lm"):
            fig_dl(e)
    if "heat" in which: fig_heat()
    if "soe" in which: fig_soe_certified()
