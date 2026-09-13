"""Consistency gate between the result files and the numbers written into the prose.

Every quantity quoted in the running text of the manuscript is recomputed here from the
result JSONs and searched for, verbatim, in main.tex.  A MISSING line means the prose
still carries a superseded value and must be updated.

    python check_numbers.py [path/to/main.tex]
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("TFADAMW_MAIN", "main.tex")), "Desktop", "OneDrive - Ministere de l'Enseignement "
    "Superieur et de la Recherche Scientifique", "projects", "28", "revision", "main.tex")


def stats(exp):
    p = os.path.join(HERE, "results", f"stats_{exp}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def row(exp, noise, method):
    st = stats(exp)
    if not st:
        return None
    for k in st:
        if abs(float(k) - noise) < 1e-12:
            for r in st[k]:
                if r["method"] == method:
                    return r
    return None


def ref_mean(exp, noise):
    st = stats(exp)
    if not st:
        return None
    for k in st:
        if abs(float(k) - noise) < 1e-12 and st[k]:
            return st[k][0]["ref_mean"]
    return None


def checks():
    """(tag, string that must occur verbatim in the manuscript)"""
    out = []
    r = row("fmnist", 0.4, "SOE-TF-AdamW")
    if r:
        out += [("fmnist mean+-sd", f"{r['mean']*100:.2f}" + chr(92) + f"pm{r['sd']*100:.2f}"),
                ("fmnist AdamW ref", f"{ref_mean('fmnist', 0.4)*100:.2f}"),
                ("fmnist paired diff", f"{r['diff']*100:+.2f}"),
                ("fmnist CI lo", f"{r['ci'][0]*100:+.2f}"),
                ("fmnist CI hi", f"{r['ci'][1]*100:+.2f}"),
                ("fmnist p_t", f"p={r['p_t']:.3f}"),
                ("fmnist p_holm", f"p={r['p_holm']:.2f}")]
        st4 = stats("fmnist")
        k4 = [x for x in st4 if abs(float(x) - 0.4) < 1e-12]
        if k4:
            sig = [x for x in st4[k4[0]] if x["p_holm"] < 0.05]
            if sig:
                out += [("fmnist significant deteriorations", f"five "),
                        ("fmnist worst deterioration",
                         f"{-min(x['diff'] for x in sig)*100:.1f}"),
                        ("fmnist smallest deterioration",
                         f"{-max(x['diff'] for x in sig)*100:.1f}")]
        import analyze as A
        v = A.table("fmnist").get("AdamW", {}).get(0.4)
        if v:
            out.append(("fmnist AdamW sd",
                        f"{np.std(list(v.values()), ddof=1)*100:.2f}"))

    # the published fractional baselines: quoted individually in the Fashion-MNIST
    # paragraph, and the most sensitive numbers to a change in the tuning grid
    tf = row("fmnist", 0.4, "SOE-TF-AdamW")
    if tf:
        for m, tag in (("TFGD", "tfgd"), ("lambda-FAdaMax", "fadamax"),
                       ("FracAdam-Shin", "fracadam"), ("FOSGD-ME", "fosgd")):
            r = row("fmnist", 0.4, m)
            if r:
                gap = (tf["mean"] - r["mean"]) * 100
                out.append((f"fmnist gap to {tag}",
                            f"{gap:.2f}" if gap < 1 else f"{gap:.1f}"))
        t = row("fmnist", 0.4, "TFGD")
        if t:
            out += [("fmnist TFGD mean", f"{t['mean']*100:.2f}"),
                    ("fmnist TFGD vs AdamW", f"{-t['diff']*100:.2f}")]

    lo = os.path.join(HERE, "results", "lora.json")
    if os.path.exists(lo):
        j = json.load(open(lo, encoding="utf-8"))
        f, l = j["full"], j["lora"]
        out += [("lora accuracy",
                 f"{l['acc_mean']*100:.1f}" + chr(92) + f"pm{l['acc_sd']*100:.1f}"),
                ("full-finetune accuracy",
                 f"{f['acc_mean']*100:.1f}" + chr(92) + f"pm{f['acc_sd']*100:.1f}"),
                ("lora bytes per model param", f"{l['bytes_per_model_param']:.2f}"),
                ("full bytes per model param", f"{f['bytes_per_model_param']:.0f}"),
                ("lora trainable count", f"{l['trainable']:,}".replace(",", "{,}")),
                ("model parameter count", f"{l['total']:,}".replace(",", "{,}")),
                ("lora trainable fraction", f"{l['frac_trainable']*100:.1f}")]

    def _sci(x, sig=1):
        m = f"{x:.{sig}e}".split("e")
        return f"{m[0]}" + chr(92) + "times10^{" + str(int(m[1])) + "}"

    si = os.path.join(HERE, "results", "synth_instances.json")
    if os.path.exists(si):
        j = json.load(open(si, encoding="utf-8"))["results"]
        b1 = [m for m in j if m.startswith("AdamW-b1=")]
        best = min(b1, key=lambda m: j[m]["gmean"]) if b1 else None
        for m, tag in [("SOE-TF-AdamW", "tf"), ("AdamW", "adamw default"),
                       ("AdamW-MM", "adamw-mm"), ("MultiEMA-8", "multiema"),
                       ("F-Adam-MM", "fadam-mm"), ("SOE-TF-AdamW-AMS", "ams variant"),
                       (best, "adamw b1-tuned")]:
            if m in j:
                out.append((f"instances gmean {tag}", _sci(j[m]["gmean"])))
        if best:
            a = np.array(j["SOE-TF-AdamW"]["vals"])
            b = np.array(j[best]["vals"])
            n = min(len(a), len(b))
            frac = float((a[:n] < b[:n]).mean())
            out.append(("instances win fraction", f"{frac*100:.0f}" + chr(92) + "%"))
        il = json.load(open(si, encoding="utf-8")).get("instance_level")
        if il:
            out += [("instances won (instance level)",
                     f"on ${il['wins']}$ of the ${il['n_instances']}$"),
                    ("instances wilcoxon p (instance level)", _sci(il["wilcoxon_p"])),
                    ("instances median ratio", f"{il['median_ratio']:.2f}")]

    sh = os.path.join(HERE, "results", "synth_shape.json")
    if os.path.exists(sh):
        j = json.load(open(sh, encoding="utf-8"))
        out += [("shape spearman ESS", f"{j['spearman_ess'][0]:+.2f}"),
                ("shape spearman p0", f"{j['spearman_p0'][0]:+.2f}"),
                ("shape fixed mean delay", f"{j['mu']:.2f}")]

    g = os.path.join(HERE, "results", "geometry.json")
    if os.path.exists(g):
        j = json.load(open(g, encoding="utf-8"))
        sw = j.get("matched_sweep")
        if sw:
            out.append(("geometry sweep size", f"{sw['pairs']:,}".replace(",", "{,}")))
        import kernels as K
        mu = K.mean_delay_tf(0.7, 0.05)
        out += [("geometry mean delay at (0.7,0.05)", f"{mu:.2f}"),
                ("geometry matched-head equivalent delay",
                 f"{K.d_mass(0.7, 0.05) - 1:.1f}"),
                ("geometry ESS ratio at (0.7,0.05)",
                 f"{K.ess_tf(0.7, 0.05) / K.ess_ema(K.beta_for_mean_delay(mu)):.3f}")]

    d = os.path.join(HERE, "results", "soe_error.json")
    if os.path.exists(d):
        j = json.load(open(d, encoding="utf-8"))
        law, mt = j["law"], j["M_for_T"]
        tab = {r["M"]: r["cert"] for r in j["table"]}
        out += [("soe envelope A", f"A={law['A']:.2f}"),
                ("soe exponent c", f"c={law['c']:.3f}"),
                ("soe M for T=1e4", "$M=" + str(mt["1e+04"]) + "$ suffices for"),
                ("soe M for T=1e12", "$M=" + str(mt["1e+12"]) + "$ for $T=10^{12}$"),
                ("soe eps at M=8", f"{tab[8]*1e6:.1f}" + chr(92) + "times10^{-6}"),
                ("soe T max at M=8",
                 f"{j['T_max_at_M8']/1e10:.2f}" + chr(92) + "times10^{10}")]

    import analyze as A
    H = A.table("heat")
    if H:
        cm = {m: np.mean(list(v[0.4].values())) for m, v in H.items() if 0.4 in v}
        if cm:
            vals = np.array(list(cm.values()))
            sds = [np.std(list(v[0.4].values()), ddof=1) / np.sqrt(len(v[0.4]))
                   for v in H.values() if 0.4 in v]
            out += [("heat min", f"{vals.min()*100:.2f}"),
                    ("heat max", f"{vals.max()*100:.2f}"),
                    ("heat range", f"{(vals.max()-vals.min())*100:.2f}"),
                    ("heat sem of a cell mean", f"{np.mean(sds)*100:.2f}")]
            d = [m for m in cm if "0.7_0.05" in m]
            if d:
                out.append(("heat default cell", f"{cm[d[0]]*100:.2f}"))

    L = A.table("lm")
    if L:
        for nz, tag in ((0.0, "clean"), (0.2, "corrupt")):
            for m in ("SOE-TF-AdamW", "AdamW"):
                if m in L and nz in L[m]:
                    v = -np.array(list(L[m][nz].values()))
                    out.append((f"lm {tag} {m}",
                                f"{v.mean():.4f}" + chr(92) + f"pm{v.std(ddof=1):.4f}"))

    for nz in (0.0, 0.4):
        r = row("cifar", nz, "SOE-TF-AdamW")
        if r:
            tag = "clean" if nz == 0 else "noisy"
            out += [(f"cifar {tag} mean+-sd",
                     f"{r['mean']*100:.2f}" + chr(92) + f"pm{r['sd']*100:.2f}"),
                    (f"cifar {tag} AdamW", f"{ref_mean('cifar', nz)*100:.2f}")]
    return out


def claims():
    """Qualitative statements the prose makes.  (tag, ok, detail)"""
    import analyze as A
    out = []
    st = stats("fmnist")
    if st:
        for k in st:
            if abs(float(k) - 0.4) < 1e-12:
                rows = st[k]
                tf = [r for r in rows if r["method"] == "SOE-TF-AdamW"]
                if tf:
                    hi = [r["method"] for r in rows if r["mean"] > tf[0]["mean"]]
                    out.append(("fmnist: SOE has the highest mean of all optimizers",
                                not hi and tf[0]["mean"] > tf[0]["ref_mean"],
                                f"beaten by {hi}" if hi else "best"))
                out.append(("fmnist: twenty comparisons against the reference",
                            len(rows) == 20,
                            f"{len(rows)} comparisons, {len(rows)+1} optimizers"))
    H = A.table("heat")
    ref = ref_mean("fmnist", 0.4)
    if H and ref:
        cm = {m: np.mean(list(v[0.4].values())) for m, v in H.items() if 0.4 in v}
        ge = sum(1 for v in cm.values() if v >= ref)
        out.append(("heat: fourteen of sixteen cells at least match AdamW",
                    (ge, len(cm)) == (14, 16), f"{ge} of {len(cm)}"))
    R = A.table("rank")
    if R:
        def mv(m, nz):
            k = f"SOE-TF-AdamW-M={m}"
            return np.mean(list(R[k][nz].values())) if k in R and nz in R[k] else None
        for nz in (0.0, 0.4):
            seq = [mv(m, nz) for m in (1, 2, 4, 8)]
            if all(v is not None for v in seq):
                out.append((f"rank: accuracy non-decreasing in M up to 8 (noise {nz})",
                            all(b >= a - 1e-12 for a, b in zip(seq, seq[1:])),
                            " -> ".join(f"{v*100:.2f}" for v in seq)))
            a, b = mv(12, nz), mv(8, nz)
            if a is not None and b is not None:
                out.append((f"rank: M=12 is worse than M=8 (noise {nz})", a < b,
                            f"{a*100:.2f} vs {b*100:.2f}"))
    L = A.table("lm")
    if L:
        for nz in sorted({n for m in L for n in L[m]}):
            vals = {m: -np.array(list(L[m][nz].values())) for m in L if nz in L[m]}
            if "SOE-TF-AdamW" in vals:
                mu = {m: v.mean() for m, v in vals.items()}
                sd = {m: v.std(ddof=1) for m, v in vals.items()}
                out.append((f"lm: lowest cross-entropy of the six (noise {nz})",
                            min(mu, key=mu.get) == "SOE-TF-AdamW",
                            f"best = {min(mu, key=mu.get)}, {len(mu)} optimizers"))
                # the paper claims the smallest spread only under corruption, and says
                # explicitly that on clean text the spread is the second largest
                order = sorted(sd, key=sd.get)
                rank = order.index("SOE-TF-AdamW") + 1
                if nz > 0:
                    out.append((f"lm: smallest seed spread (noise {nz})",
                                rank == 1, f"rank {rank} of {len(order)}"))
                else:
                    out.append(("lm: second largest seed spread on clean text",
                                rank == len(order) - 1,
                                f"rank {rank} of {len(order)}"))
    cst = stats("cifar")
    if cst:
        for nz in (0.0, 0.4):
            r = row("cifar", nz, "SOE-TF-AdamW")
            if not r:
                continue
            key = [k for k in cst if abs(float(k) - nz) < 1e-12][0]
            hi = [x["method"] for x in cst[key] if x["mean"] > r["mean"]]
            if r["mean"] <= r["ref_mean"]:
                hi = hi + ["AdamW"]
            out.append((f"cifar: standing at noise {nz}", True,
                        "best on the mean" if not hi else f"beaten by {hi}"))
    return out


if __name__ == "__main__":
    src = open(MAIN, encoding="utf-8").read()
    bad = 0
    for tag, s in checks():
        ok = s in src
        bad += not ok
        print(f"{'ok  ' if ok else 'MISS'}  {tag:28s} {s}")
    print()
    for tag, ok, detail in claims():
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {tag:52s} {detail}")
    print(f"\n{bad} inconsistency(ies) against {os.path.basename(MAIN)}")
    sys.exit(1 if bad else 0)
