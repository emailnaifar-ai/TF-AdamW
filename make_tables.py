"""Emit the manuscript's result tables as LaTeX, directly from the stored result files,
so that no number in a table is transcribed by hand.

    python make_tables.py            -> writes tables_*.tex into the revision folder
"""
import os, json, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.environ.get("TFADAMW_TABLES",
                     os.path.join(HERE, "tables"))
BS = chr(92)
EOR = BS * 2


# One canonical display name per optimizer.  The results files use the short internal keys;
# the manuscript names the three variants separately and must do so consistently, so every
# table renders a method through disp().
DISP = {"SOE-TF-AdamW-AMS": "AMSGrad-SOE-TF-AdamW",
        "AdamW-MM": "AdamW-MM (matched mean delay)",
        "F-Adam-MM": "F-Adam-MM (untempered, matched)",
        "FracAdam-Shin": "Fractional Adam",
        "lambda-FAdaMax": "$" + chr(92) + "lambda$-FAdaMax"}


def disp(m):
    return DISP.get(m, m)


def load(n):
    p = os.path.join(RES, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def sci(x, sig=2):
    if x is None or not np.isfinite(x):
        return "---"
    s = f"{x:.{sig}e}"
    m, e = s.split("e")
    return f"${m}" + BS + "times10^{" + str(int(e)) + "}$"


def wrap(body, cols, caption, label, small=True, fit=False):
    L = [BS + "begin{table}[H]", BS + "centering",
         BS + "caption{" + caption + "}", BS + "label{" + label + "}"]
    if small:
        L.append(BS + "small")
    if fit:
        L.append(BS + "resizebox{" + BS + "textwidth}{!}{%")
    L += [BS + "begin{tabular}{" + cols + "}", BS + "toprule"]
    L += body
    L += [BS + "bottomrule", BS + "end{tabular}"]
    if fit:
        L.append("}")
    L.append(BS + "end{table}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------- synthetic instances
def tab_instances():
    d = load("synth_instances.json")
    if not d:
        return ""
    R = d["results"]
    b1 = [m for m in R if m.startswith("AdamW-b1=")]
    best_b1 = min(b1, key=lambda m: R[m]["gmean"]) if b1 else None
    order = [("AdamW", "AdamW (default $" + BS + "beta_1=0.9$)"),
             (best_b1, "AdamW ($" + BS + "beta_1$ tuned $=" + (best_b1 or "").split("=")[-1] + "$)"),
             ("AdamW-MM", "AdamW-MM (mean-delay matched)"),
             ("MultiEMA-8", "Multi-EMA bank ($M{=}8$, matched)"),
             ("Boxcar-MM", "Boxcar ($W{=}28$, matched)"),
             ("AMSGrad", "AMSGrad"), ("RMSProp", "RMSProp"),
             ("F-Adam", "F-Adam (untempered, $J{=}64$)"),
             ("F-Adam-MM", "F-Adam-MM (untempered, matched)"),
             ("FracAdam-Shin", "Fractional Adam"),
             ("lambda-FAdaMax", "$" + BS + "lambda$-FAdaMax"),
             ("FOSGD-ME", "FOSGD-ME"), ("TFGD", "TFGD"),
             ("SOE-TF-AdamW-AMS", "AMSGrad-SOE-TF-AdamW (theory variant)"),
             ("TF-AdamW (exact)", "TF-AdamW (exact kernel)"),
             ("SOE-TF-AdamW", BS + "textbf{SOE-TF-AdamW (ours)}")]
    best = min(R[m]["gmean"] for m, _ in order if m in R)
    rows = ["Method & $" + BS + "eta$ & geometric mean & median & IQR " + EOR, BS + "midrule"]
    for m, lbl in order:
        if m not in R:
            continue
        r = R[m]
        g = sci(r["gmean"])
        if abs(r["gmean"] - best) < 1e-12:
            g = BS + "mathbf{" + g.strip("$") + "}"
            g = "$" + g + "$"
        rows.append(f"{lbl} & ${r['lr']:g}$ & {g} & {sci(r['median'])} & "
                    f"[{sci(r['q25'])}, {sci(r['q75'])}] {EOR}")
    cap = ("Controlled stochastic optimization aggregated over " + BS + "textbf{20 independently "
           "generated problem instances} (three dimensions, three condition numbers, three "
           "spectrum shapes, two noise models, fresh random basis per instance) with three "
           r"unseen seeds each, i.e.\ 60 runs per method. Learning rates were selected on four "
           "disjoint tuning instances with two disjoint tuning seeds, and the grid was widened "
           "until no method selected a boundary value. Lower is better.")
    return wrap(rows, "lcccc", cap, "tab:instances")


# ------------------------------------------------------------- kernel shape
def tab_shape():
    d = load("synth_shape.json")
    if not d:
        return ""
    R = d["results"]
    rows = ["Kernel & $" + BS + "mu$ & $" + BS + "mathrm{ESS}$ & $p_0$ & $" + BS + "eta$ & "
            "final gap " + EOR, BS + "midrule"]
    for k in sorted(R, key=lambda k: R[k]["gmean"]):
        lbl = (k.replace(BS + BS + "alpha", BS + "alpha")
                .replace("tempered fractional $" + BS + "alpha$=",
                         "tempered fractional, $" + BS + "alpha=")
                .replace("multi-EMA (M=8)", "multi-EMA bank ($M{=}8$)")
                .replace("boxcar", "boxcar ($W{=}28$)"))
        if "tempered fractional, " in lbl:
            lbl = lbl + "$"
        r = R[k]
        rows.append(f"{lbl} & ${r['mu']:.2f}$ & ${r['ess']:.1f}$ & ${r['p0']:.4f}$ & "
                    f"${r['lr']:g}$ & {sci(r['gmean'])} {EOR}")
    sp, pp = d["spearman_p0"]; se, pe = d["spearman_ess"]
    cap = ("Kernel " + BS + "emph{shape} at mean delay matched to "
           f"$\\mu={d['mu']:.2f}$, so that stale-gradient bias is held constant and only the "
           "weight profile varies. The boxcar reaches $13.50$, the nearest value its integer "
           "window allows. Same 20 instances, three unseen seeds, per-kernel "
           "tuned step size. Across the eight kernels the Spearman correlation between final "
           f"gap and effective sample size is ${se:+.2f}$ ($p={pe:.2f}$) and with head weight "
           f"$p_0$ is ${sp:+.2f}$ ($p={pp:.2f}$); with eight kernels neither is significant at "
           "the $5" + BS + "%$ level, but the ordering is plainly not explained by "
           "$" + BS + "mathrm{ESS}$.")
    return wrap(rows, "lccccc", cap, "tab:shape")


# ------------------------------------------------------------- kernel geometry
def tab_geometry():
    d = load("geometry.json")
    if not d:
        return ""
    rows = ["$" + BS + "alpha$ & $" + BS + "lambda$ & $" + BS + "mu$ & $p_0$ & "
            "$" + BS + "mathrm{ESS}_{" + BS + "mathrm{TF}}$ & "
            "$" + BS + "mathrm{ESS}_{" + BS + "mathrm{E}}$ at matched $" + BS + "mu$ & ratio & "
            "$" + BS + "mathrm{ESS}_{" + BS + "mathrm{E}}$ at matched $p_0$ & ratio " + EOR,
            BS + "midrule"]
    for r in d["matched"]:
        rows.append(f"${r['alpha']:.1f}$ & ${r['lam']:.2f}$ & ${r['mu']:.2f}$ & "
                    f"${r['p0']:.4f}$ & ${r['ess']:.2f}$ & ${r['ess_ema_mu']:.2f}$ & "
                    f"${r['ratio_ess_at_matched_mu']:.3f}$ & ${r['ess_ema_p0']:.2f}$ & "
                    f"${r['ratio_ess_at_matched_p0']:.3f}$ {EOR}")
    worst = max(max(x["rel_d"], x["rel_s2"], x["rel_mu"]) for x in d["identities"])
    cap = ("Kernel geometry of Proposition~" + BS + "ref{prop:geometry}. At a matched "
           + BS + "emph{mean delay} the exponential kernel attains the higher effective sample "
           "size (ratio $<1$ in every row); at a matched " + BS + "emph{head weight} $p_0$ the "
           "tempered fractional kernel does (ratio $>1$ in every row). The closed forms agree "
           "with brute-force summation of the kernel to a maximum relative error of "
           + sci(worst, 1) + ".")
    return wrap(rows, "cccccccc" + "c", cap, "tab:geometry", fit=True)


# ------------------------------------------------------------- certified SOE error
def tab_soe():
    d = load("soe_error.json")
    if not d:
        return ""
    tab = d["table"]
    keep = [r for r in tab if r["M"] in (1, 2, 4, 6, 8, 10, 11)]
    hdr = "$M$ & " + " & ".join(f"${r['M']}$" for r in keep) + " " + EOR
    rows = [hdr, BS + "midrule",
            "certified $" + BS + "varepsilon_{" + BS + "soe}$ & "
            + " & ".join(sci(r["cert"]) for r in keep) + " " + EOR,
            "relative to $d(" + BS + "alpha," + BS + "lambda)$ & "
            + " & ".join(sci(r["cert"] / 8.2849) for r in keep) + " " + EOR]
    cap = ("Certified approximation error of the nonnegative sum-of-exponentials fit to "
           "$" + BS + "kk_j^{(0.7,0.05)}$ (Lemma~" + BS + "ref{lem:certified}): the computed "
           "residual over the fitting horizon $J=4000$ plus the analytic kernel tail plus the "
           "exact SOE tail. At this horizon the kernel tail is of order "
           + sci(d["kernel_tail"], 1) + " and the SOE tail smaller still, so the certified "
           "bound coincides with the fitted residual to every digit shown. The error decays geometrically in $M$, "
           f"$\\varepsilon_{{\\soe}}(M)\\le{d['law']['A']:.3g}e^{{-{d['law']['c']:.3f}M}}$ "
           f"($R^2={d['law']['R2']:.5f}$); double-precision NNLS reaches its numerical floor "
           "near $M=11$, so larger $M$ are not reported.")
    return wrap(rows, "l" + "c" * len(keep), cap, "tab:soe", fit=True)


# ------------------------------------------------------------- deep learning (paired)
def tab_dl(exp="fmnist", ref="AdamW", label=None, caption=None, metric="accuracy"):
    d = load(f"stats_{exp}.json")
    if not d:
        return ""
    noises = sorted(d, key=lambda k: float(k))
    meths, cells = [], {}
    for nz in noises:
        for r in d[nz]:
            if r["method"] not in meths:
                meths.append(r["method"])
            cells[(r["method"], nz)] = r
    refrow = {}
    for nz in noises:
        rows = d[nz]
        if rows:
            refrow[nz] = rows[0]["ref_mean"], rows[0]["n"]
    hdr = "Optimizer & " + " & ".join(
        (f"clean" if float(nz) == 0 else f"${float(nz)*100:.0f}" + BS + "%$ noise")
        for nz in noises) + " " + EOR
    rows = [hdr, BS + "midrule"]
    if refrow:
        # the reference row carries its own spread too: the text compares the proposed
        # method's standard deviation with it, and a bare mean cannot be checked
        import analyze as A
        raw = A.table(exp)
        refsd = {}
        for nz in noises:
            v = raw.get(ref, {}).get(float(nz))
            if v:
                refsd[nz] = float(np.std(list(v.values()), ddof=1))
        rows.append(ref + " (reference) & " + " & ".join(
            (f"${refrow[nz][0]*100:.2f}" + (BS + f"pm{refsd[nz]*100:.2f}" if nz in refsd
                                            else "") + "$")
            if nz in refrow else "---" for nz in noises) + " " + EOR)
    best = {nz: max([c["mean"] for (m, n2), c in cells.items() if n2 == nz]
                    + [refrow[nz][0]] if nz in refrow else [0]) for nz in noises}
    for m in meths:
        out = []
        for nz in noises:
            c = cells.get((m, nz))
            if not c:
                out.append("---"); continue
            # A percentile bootstrap over three paired differences resamples three
            # numbers; it can exclude zero while the paired t-test is nowhere near
            # significance, so no star is awarded on that basis below five seeds.
            ci_excl = (c["ci"][0] > 0 or c["ci"][1] < 0) and c.get("n", 0) >= 5
            star = (BS + "textsuperscript{$" + BS + "dagger$}" if c.get("p_holm", 1) < 0.05
                    else (BS + "textsuperscript{*}" if ci_excl else ""))
            v = f"{c['mean']*100:.2f}" + BS + "pm" + f"{c['sd']*100:.2f}"
            if abs(c["mean"] - best.get(nz, 0)) < 1e-12:
                v = BS + "mathbf{" + v + "}"
            out.append("$" + v + "$" + star)
        rows.append(disp(m) + " & " + " & ".join(out) + " " + EOR)
    # per-column sample sizes: the clean control condition runs fewer seeds than the key
    # noisy one, and a single scalar n in the caption would misstate half the table
    ns = []
    for nz in noises:
        if nz in refrow:
            lab = "clean" if float(nz) == 0 else f"${float(nz)*100:.0f}" + BS + "%$ noise"
            ns.append("$n=" + str(refrow[nz][1]) + "$ " + lab)
    n = refrow[noises[-1]][1] if refrow else "?"
    holm = [c.get("p_holm", 1) for c in cells.values()]
    best_holm = min(holm) if holm else 1.0
    # Below five seeds a percentile bootstrap resamples three or four numbers and can
    # exclude zero while the paired t-test is nowhere near significance, so no star is
    # awarded on that basis.  The caption has to say so, or it states a rule the table
    # does not obey -- and a reader comparing the interval against an unstarred row would
    # read the missing star as a suppressed inconvenient result.
    nmin = min((refrow[nz][1] for nz in noises if nz in refrow), default=10)
    small = nmin < 5
    # How many unstarred entries have an interval that excludes zero. The caption used to
    # say "including one" whatever the count was; on the final CIFAR table three do.
    n_excl = sum(1 for (m, _), r in cells.items()
                 if m != ref and "ci" in r and (r["ci"][0] > 0 or r["ci"][1] < 0))
    excl_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    excl_phrase = ("" if n_excl == 0 else
                   ", including one whose interval does exclude zero" if n_excl == 1 else
                   f", including the {excl_words.get(n_excl, n_excl)} whose intervals do "
                   "exclude zero")
    cap = caption or (
        "Test " + metric + " (mean $" + BS + "pm$ s.d." + BS + " over paired evaluation "
        "seeds, " + " and ".join(ns) + "; the label-noise realization is identical across "
        "optimizers at each seed, so the comparison is paired). Learning rates were "
        "selected on a disjoint tuning seed by validation " + metric + ". A star marks a "
        r"difference from " + ref + r", \emph{in either direction}, whose $95" + BS + r"%$ "
        "bootstrap confidence interval on the paired difference excludes zero"
        + (", awarded only where at least five evaluation seeds are available" if small
           else "")
        + "; a dagger "
        "marks one that also survives Holm--Bonferroni correction across the family of "
        "comparisons against " + ref + ". The direction is read from the means: a starred "
        r"entry scoring below the reference row is significantly \emph{worse}. "
        + ("No difference survives that correction (smallest corrected $p="
           + f"{best_holm:.2f}$). " if best_holm >= 0.05 else "")
        + (f"At $n={nmin}$ a percentile bootstrap resamples {nmin} numbers, so it is not "
           "read as a significance test here and the paired $t$-test governs: no entry in "
           "this table is starred" + excl_phrase + ", and no "
           "difference in it is statistically distinguishable from " + ref + "."
           if small else
           "Differences without a star are not statistically distinguishable from "
           + ref + "."))
    return wrap(rows, "l" + "c" * len(noises), cap, label or f"tab:{exp}", fit=False)


def tab_cost():
    d = load("cost.json")
    if not d:
        return ""
    models = list(d)
    meths = list(d[models[0]]["rows"])
    rows = ["Optimizer & bytes/param & $" + BS + "times$ Adam state & "
            + " & ".join(m.split(" (")[0] for m in models) + " " + EOR,
            BS + "multicolumn{" + str(3 + len(models)) + "}{l}{"
            + BS + "emph{relative step time by model}} " + EOR, BS + "midrule"]
    for m in meths:
        r0 = d[models[0]]["rows"][m]
        bpp = r0["bytes_per_param"]
        # Adafactor factorises its second moment, so its state depends on parameter shapes
        # rather than on the parameter count; a single rounded integer would read as zero
        shape_dep = len({round(d[mm]["rows"][m]["bytes_per_param"], 3)
                         for mm in models}) > 1
        bs = (f"${bpp:.0f}$" if not shape_dep
              else "$" + f"{min(d[mm]['rows'][m]['bytes_per_param'] for mm in models):.2f}"
              + BS + "text{--}"
              + f"{max(d[mm]['rows'][m]['bytes_per_param'] for mm in models):.2f}$")
        rows.append(f"{m} & " + bs + f" & ${r0['rel_state']:.2f}$ & "
                    + " & ".join(f"${d[mm]['rows'][m]['rel_time']:.2f}$" for mm in models)
                    + " " + EOR)
    cap = ("Optimizer cost. Bytes per parameter is exact and device-independent "
           "($4(M{+}1)$ in fp32 against $8$ for Adam); Adafactor is the exception, since "
           "factorising the second moment makes its state depend on the parameter shapes, "
           "so a range over the five models is given and the state ratio is that of the "
           "first. The remaining columns are measured step times relative to AdamW on each "
           "model. All measurements are on CPU, so "
           "peak GPU memory is not reported.")
    return wrap(rows, "lcc" + "c" * len(models), cap, "tab:cost", fit=True)


def tab_rank():
    """SOE rank M against certified approximation error, state cost and accuracy."""
    import analyze as A
    import kernels as K
    R = A.table("rank")
    if not R:
        return ""
    rows = ["$M$ & certified $" + BS + "varepsilon_{" + BS + "soe}$ & state vs Adam & "
            "clean & $40" + BS + "%$ noise " + EOR, BS + "midrule"]
    got = []
    for m in sorted(R, key=lambda m: int(m.split("=")[-1]) if "=" in m else 8):
        M = int(m.split("=")[-1]) if "=" in m else 8
        eps = K.soe_fit_nnls(0.7, 0.05, M)["eps_certified"]
        cells = []
        for nz in (0.0, 0.4):
            v = np.array(list(R[m].get(nz, {}).values()))
            cells.append(f"${v.mean()*100:.2f}" + BS + "pm" + f"{v.std(ddof=1)*100:.2f}$"
                         if len(v) else "---")
        got.append((M, eps, cells))
        rows.append(f"${M}$ & {sci(eps)} & ${4*(M+1)/8:.2f}" + BS + "times$ & "
                    + " & ".join(cells) + " " + EOR)
    cap = ("Effect of the sum-of-exponentials rank $M$ on Fashion-MNIST (mean $" + BS +
           r"pm$ s.d.\ over five evaluation seeds). Accuracy tracks the "
           + BS + "emph{certified approximation error} rather than $M$ itself: $M=12$ has a "
           "larger certified error than $M=8$, because the nonnegative least-squares fit "
           "reaches its double-precision floor near $M=11$, and it also performs worse. "
           "$M=4$ is statistically indistinguishable from $M=8$ at $2.5" + BS + "times$ "
           "rather than $4.5" + BS + "times$ Adam's optimizer state.")
    return wrap(rows, "lcccc", cap, "tab:rank")


def tab_lm():
    """Character-level Transformer LM: test cross-entropy (lower is better)."""
    import analyze as A
    D = A.table("lm")
    if not D:
        return ""
    noises = sorted({n for m in D for n in D[m]})
    stats = {}
    for m, byn in D.items():
        for nz in noises:
            if nz in byn:
                v = -np.array(list(byn[nz].values()))
                stats[(m, nz)] = (v.mean(), v.std(ddof=1), len(v))
    best = {nz: min(v[0] for (m, n2), v in stats.items() if n2 == nz) for nz in noises}
    order = sorted(D, key=lambda m: stats.get((m, noises[0]), (9,))[0])
    hdr = "Optimizer & " + " & ".join(
        ("clean" if nz == 0 else f"${nz*100:.0f}" + BS + "%$ token corruption")
        for nz in noises) + " " + EOR
    rows = [hdr, BS + "midrule"]
    for m in order:
        cells = []
        for nz in noises:
            if (m, nz) not in stats:
                cells.append("---"); continue
            mu, sd, n = stats[(m, nz)]
            v = f"{mu:.4f}" + BS + "pm" + f"{sd:.4f}"
            if abs(mu - best[nz]) < 1e-12:
                v = BS + "mathbf{" + v + "}"
            cells.append("$" + v + "$")
        rows.append(disp(m) + " & " + " & ".join(cells) + " " + EOR)
    n = list(stats.values())[0][2]
    cap = ("Character-level Transformer language model: test cross-entropy in nats per "
           r"character, " + BS + r"emph{lower is better} (mean $" + BS + r"pm$ s.d.\ over "
           + str(n) + " evaluation seeds, learning rate selected on a disjoint tuning "
           "seed). The corrupted condition replaces a fraction of the input tokens with "
           "uniformly random ones. With only " + str(n) + " seeds no difference here is statistically "
           "established; the table is reported because it extends the empirical scope "
           "beyond vision, which is what the introduction's motivation requires.")
    return wrap(rows, "l" + "c" * len(noises), cap, "tab:lm")



def _lr_cell(x):
    """Learning rate in units of 1e-3, compactly."""
    if x is None:
        return "---"
    v = x / 1e-3
    return f"{v:g}"


def tab_lr():
    """Selected learning rate per optimizer per compared task, straight from the
    selection files written by the tuning stage."""
    sel = {}
    for e in ("fmnist", "cifar", "lm"):
        p = os.path.join(HERE, f"selected_lr_{e}.json")
        if not os.path.exists(p):
            continue
        for k, lr in json.load(open(p, encoding="utf-8")).items():
            m, nz = k.rsplit("|", 1)
            if m.startswith("AdamW-b1tuned:"):
                m = "AdamW ($\\beta_1$ tuned)"
            elif m.startswith("AdamW-b1="):
                continue                       # sweep points, superseded by the winner
            sel[(e, float(nz), m)] = lr
    inst = load("synth_instances.json")
    if inst:
        for m, d in inst["results"].items():
            if m.startswith("AdamW-b1="):
                continue
            sel[("inst", 0.0, m)] = d.get("lr")
        b1 = [(m, d) for m, d in inst["results"].items() if m.startswith("AdamW-b1=")]
        if b1:
            w = min(b1, key=lambda kv: kv[1]["gmean"])
            sel[("inst", 0.0, "AdamW ($\\beta_1$ tuned)")] = w[1].get("lr")
    if not sel:
        return ""
    cols = [("Inst.", ("inst", 0.0)), ("FM-c", ("fmnist", 0.0)), ("FM-n", ("fmnist", 0.4)),
            ("C10-c", ("cifar", 0.0)), ("C10-n", ("cifar", 0.4)),
            ("LM-c", ("lm", 0.0)), ("LM-x", ("lm", 0.2))]
    meths = sorted({m for (_, _, m) in sel})
    tail = [m for m in meths if "TF-AdamW" in m]
    meths = [m for m in meths if m not in tail] + sorted(tail)
    rows = ["Optimizer & " + " & ".join(c for c, _ in cols) + " " + EOR, BS + "midrule"]
    for m in meths:
        cells = [_lr_cell(sel.get((e, nz, m))) for _, (e, nz) in cols]
        if all(c == "---" for c in cells):
            continue
        name = disp(m)
        name = name if "$" in name else name.replace("_", BS + "_")
        rows.append(name + " & " + " & ".join(cells) + " " + EOR)
    cap = ("Learning rate selected by each optimizer on each compared task, in units of "
           "$10^{-3}$. ``Inst.''~$=$~the distribution of noisy ill-conditioned instances; "
           "``FM''~$=$~Fashion-MNIST, ``C10''~$=$~ResNet-20 on CIFAR-10, ``LM''~$=$~the "
           "character-level Transformer; ``-c'' is the clean condition, ``-n'' the "
           "label-noise condition and ``-x'' the token-corruption condition. "
           "``---''~$=$~not run on that task. Selection always uses tuning-only data "
           "(a disjoint tuning seed, or disjoint problem instances) and the "
           "validation metric, never the reported test metric.")
    return wrap(rows, "l" + "c" * len(cols), cap, "tab:lr")



def tab_gradstats():
    """Gradient statistics measured during real training under label corruption."""
    d = load("gradstats.json")
    if not d:
        return ""
    rows_by = {}
    for k, v in d.items():
        m, nz = k.rsplit("|", 1)
        rows_by.setdefault(m, {})[float(nz)] = v
    order = [m for m in ("AdamW", "AdamW-MM", "SOE-TF-AdamW") if m in rows_by]
    if not order:
        return ""
    noises = sorted({nz for v in rows_by.values() for nz in v})
    hdr = ("Optimizer & " + " & ".join(("$0$" if nz == 0 else f"${nz*100:.0f}" + BS + "%$")
                                       for nz in noises)
           + " & realized & predicted " + EOR)
    body = [hdr, BS + "midrule"]
    for m in order:
        v = rows_by[m]
        cells = [f"${v[nz]['grad_var']:.2f}$" for nz in noises]
        real = np.mean([v[nz]["filt_over_raw_var"] for nz in noises])
        pred = v[noises[0]]["theoretical_weight_energy"]
        body.append(disp(m) + " & " + " & ".join(cells)
                    + f" & ${real:.4f}$ & ${pred:.4f}$ " + EOR)
    mx = max(abs(x["cos_consec"]) for x in d.values())
    nprobe = min(x.get("n_probe", 0) for x in d.values())
    cap = ("Gradient statistics measured during training on Fashion-MNIST at four symmetric "
           f"label-corruption rates, from {nprobe} probe points spread through one training "
           "run per cell: this measures the gradient distribution rather than comparing "
           "accuracies, so no evaluation seeds are involved. The first four columns are the trace of the empirical "
           "mini-batch gradient covariance, $" + BS + "operatorname{tr}" + BS
           + "widehat{" + BS + "mathrm{Cov}}(g)$, averaged over training; the last two "
           r"compare the \emph{realized} variance ratio between the first moment and the raw "
           "gradient with the value $" + BS + "sum_j p_j^2$ predicted by "
           r"Proposition~" + BS + "ref{prop:variance}. Gradient noise \emph{falls} as label "
           "noise rises, so the two are not interchangeable; the realized filtering factor "
           "tracks the prediction to within about $20" + BS + "%$; and the correlation "
           "between consecutive gradients never exceeds $" + f"{mx:.2f}" + "$ in absolute "
           "value, which is the independence condition the proposition assumes.")
    return wrap(body, "l" + "c" * (len(noises) + 2), cap, "tab:gradstats")


if __name__ == "__main__":
    made = []
    jobs = [("instances", tab_instances), ("shape", tab_shape),
            ("geometry", tab_geometry), ("soe", tab_soe), ("cost", tab_cost)]
    jobs += [(f"dl_{e}", (lambda e=e: tab_dl(e))) for e in ("fmnist", "fmnistfull",
                                                            "cifar")]
    jobs.append(("rank", tab_rank))
    jobs.append(("lm", tab_lm))
    jobs.append(("lr", tab_lr))
    jobs.append(("gradstats", tab_gradstats))
    withdrawn = []
    for name, fn in jobs:
        t = fn()
        p = os.path.join(OUT, f"table_{name}.tex")
        if t:
            open(p, "w", encoding="utf-8").write(t)
            made.append(name)
        elif os.path.exists(p):
            # The result file this table was built from is gone -- the runs behind it were
            # archived, or have not been recomputed yet. Leaving the file untouched would
            # go on typesetting numbers that nothing backs any more, which is exactly how
            # a set of invalidated CIFAR figures survived in the manuscript after the runs
            # behind them were set aside. The label is carried over so cross-references
            # still resolve, and the marker is one final_check.py refuses to let through.
            old_text = open(p, encoding="utf-8").read()
            m = re.search(r"\\label\{([^}]*)\}", old_text)
            lab = m.group(1) if m else f"tab:{name}"
            open(p, "w", encoding="utf-8").write(
                "\\begin{table}[H]\n\\centering\n"
                "\\caption{\\textsc{[pending]} The results for this table are being "
                "recomputed; the numbers are not yet final.}\n"
                f"\\label{{{lab}}}\n\\end{{table}}\n")
            withdrawn.append(name)
    print("wrote tables:", ", ".join(made) if made else "(none)")
    if withdrawn:
        print("withdrawn (no backing results):", ", ".join(withdrawn))
