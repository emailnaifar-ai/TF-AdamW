"""Unified deep-learning harness: datasets, models, optimizer factory, training loop.

Every experiment in the revision is expressed as a JOB dict and executed by `run_job`,
so that the tuning protocol (separate tuning seeds / evaluation seeds) and the paired
label-noise realizations are identical across optimizers by construction.
"""
import os, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kernels as K
import opt_zoo as Z

WORK = os.environ.get("TFADAMW_DATA", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data"))
_CACHE = {}


# ------------------------------------------------------------------ data
def load_fmnist(n_train=12000, n_val=3000, n_test=5000, split_seed=0):
    key = ("fm", n_train, n_val, n_test, split_seed)
    if key in _CACHE:
        return _CACHE[key]
    d = np.load(os.path.join(WORK, "fmnist.npz"))
    n_all = n_train + n_val + n_test
    rng = np.random.default_rng(split_seed)
    # permute INDICES and slice before materializing floats: keeps the working set to
    # exactly the samples this job needs instead of the whole 70k array
    idx = rng.permutation(len(d["y"]))[:n_all]
    X = d["X"][idx].astype("float32") / 255.0
    y = d["y"][idx].astype("int64")
    mu, sd = X[:n_train].mean(), X[:n_train].std() + 1e-6
    X -= mu; X /= sd
    a, b, c = n_train, n_train + n_val, n_all
    out = ((X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:c], y[b:c]))
    _CACHE[key] = out
    return out


def _check_cifar_layout(X):
    """Fail loudly if the flat rows are not channel-major.

    An earlier version of this loader read the rows as height-width-channel and transposed
    them, which interleaved the channels and scrambled every image. Nothing downstream
    complained: a ResNet still reaches a plausible accuracy on a fixed pixel permutation,
    so the mistake survived a full experimental campaign. These two invariants catch it in
    a fraction of a second.
    """
    S = X[:2000] * 255.0 if X.max() <= 1.0 + 1e-6 else X[:2000]
    chan = S.reshape(-1, 3, 1024).mean((0, 2))
    ref = np.array([125.3, 123.0, 113.9])          # CIFAR-10's published channel means
    if np.abs(chan - ref).max() > 4.0:
        raise RuntimeError(
            f"CIFAR-10 channel means {np.round(chan, 1)} do not match the published "
            f"{ref}; the rows are probably not in channel-major order.")
    img = S.reshape(-1, 3, 32, 32)
    adj = np.abs(np.diff(img, axis=-1)).mean()
    if adj > 20.0:                                  # ~14 when decoded correctly, ~29 when not
        raise RuntimeError(
            f"CIFAR-10 images have mean adjacent-pixel difference {adj:.1f}; natural "
            "images give about 14, so the pixel layout is wrong.")


def load_cifar(n_train=45000, n_val=5000, n_test=10000, split_seed=0):
    key = ("c10", n_train, n_val, n_test, split_seed)
    if key in _CACHE:
        return _CACHE[key]
    d = np.load(os.path.join(WORK, "cifar10.npz"))
    n_all = len(d["ytr"])
    if n_train + n_val > n_all:            # CIFAR-10 ships 50k train images; the
        n_train = n_all - n_val            # validation split is carved out of them
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(n_all)[:n_train + n_val]
    Xtr = d["Xtr"][idx].astype("float32") / 255.0
    ytr = d["ytr"][idx].astype("int64")
    Xte = d["Xte"][:n_test].astype("float32") / 255.0
    yte = d["yte"][:n_test].astype("int64")
    # cifar10.npz stores rows in CIFAR-10's official channel-major layout: 1024 red
    # values, then green, then blue, each row-major. That is already the (3, 32, 32)
    # ordering the model expects, so no transpose is applied -- reshaping as (32, 32, 3)
    # and transposing would interleave the channels and scramble the image.
    _check_cifar_layout(Xtr)
    m = Xtr[:n_train].reshape(-1, 3, 1024).mean((0, 2))
    s = Xtr[:n_train].reshape(-1, 3, 1024).std((0, 2)) + 1e-6
    def norm(A):                            # in place: seven concurrent workers each
        V = A.reshape(-1, 3, 1024)          # hold the array, so avoid a second copy
        V -= m[None, :, None]
        V /= s[None, :, None]
        return A
    Xtr, Xte = norm(Xtr), norm(Xte)
    out = ((Xtr[:n_train], ytr[:n_train]),
           (Xtr[n_train:], ytr[n_train:]), (Xte, yte))
    _CACHE[key] = out
    return out


def load_text():
    """Tiny-Shakespeare-style character corpus for the sequence-modelling benchmark."""
    if "text" in _CACHE:
        return _CACHE["text"]
    path = os.path.join(WORK, "input.txt")
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    chars = sorted(set(s))
    stoi = {c: i for i, c in enumerate(chars)}
    arr = np.array([stoi[c] for c in s], dtype=np.int64)
    n = len(arr)
    out = (arr[: int(0.9 * n)], arr[int(0.9 * n): int(0.95 * n)], arr[int(0.95 * n):], len(chars))
    _CACHE["text"] = out
    return out


def add_label_noise(y, rate, num_classes=10, noise_seed=0):
    """Symmetric label noise.  `noise_seed` is deliberately SEPARATE from the training
    seed so that every optimizer can be evaluated on the IDENTICAL corruption
    realization (paired comparison, Reviewer 2 comment 11)."""
    if rate <= 0:
        return y
    rng = np.random.default_rng(10000 + noise_seed)
    y = y.copy()
    flip = rng.random(len(y)) < rate
    y[flip] = rng.integers(0, num_classes, int(flip.sum()))
    return y


# ------------------------------------------------------------------ models
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


class _Block(nn.Module):
    def __init__(self, i, o, st=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, st, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.sh = (nn.Sequential() if st == 1 and i == o else
                   nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sh(x))


class ResNet20(nn.Module):
    """Standard CIFAR ResNet-20 (He et al., 2016): 3 stages x 3 basic blocks."""

    def __init__(self, w=16, nc=10):
        super().__init__()
        self.c = nn.Conv2d(3, w, 3, 1, 1, bias=False); self.b = nn.BatchNorm2d(w)
        L, ch = [], w
        for st, o in [(1, w), (2, 2 * w), (2, 4 * w)]:
            for i in range(3):
                L.append(_Block(ch, o, st if i == 0 else 1)); ch = o
        self.L = nn.Sequential(*L); self.f = nn.Linear(ch, nc)

    def forward(self, x):
        h = self.L(F.relu(self.b(self.c(x))))
        return self.f(F.adaptive_avg_pool2d(h, 1).flatten(1))


class CharTransformer(nn.Module):
    """Small decoder-only Transformer LM (the non-vision, sequence-modelling benchmark
    the Introduction's motivation calls for)."""

    def __init__(self, V, d=128, nh=4, nl=2, L=128, drop=0.1):
        super().__init__()
        self.e = nn.Embedding(V, d); self.p = nn.Embedding(L, d)
        el = nn.TransformerEncoderLayer(d, nh, 4 * d, drop, batch_first=True,
                                        norm_first=True, activation="gelu")
        self.t = nn.TransformerEncoder(el, nl)
        self.ln = nn.LayerNorm(d); self.o = nn.Linear(d, V); self.L = L

    def forward(self, x):
        n = x.size(1)
        h = self.e(x) + self.p(torch.arange(n))
        msk = torch.triu(torch.full((n, n), float("-inf")), 1)
        return self.o(self.ln(self.t(h, mask=msk)))


class LoRALinear(nn.Module):
    """Frozen base linear + trainable low-rank adapter (for the LoRA experiment that
    demonstrates the state-restriction remedy of Reviewer 2 comment 14)."""

    def __init__(self, base: nn.Linear, r=8, scale=2.0):
        super().__init__()
        self.base = base
        for q in self.base.parameters():
            q.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = scale / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


# ------------------------------------------------------------------ optimizers
WD = 5e-4


def make_opt(name, params, lr, ctx):
    """ctx carries the SOE fit, the matched mean delay, and any method-specific knob."""
    params = list(params)
    fit = ctx["fit"]; mu = ctx["mu"]
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    if name.startswith("AdamW-b1="):
        b1 = float(name.split("=")[1])
        return torch.optim.AdamW(params, lr=lr, betas=(b1, 0.999), weight_decay=WD)
    if name == "AdamW-MM":                       # memory-matched beta1
        b1 = K.beta_for_mean_delay(mu)
        return torch.optim.AdamW(params, lr=lr, betas=(b1, 0.999), weight_decay=WD)
    if name == "AMSGrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=WD, amsgrad=True)
    if name == "RMSProp":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "RAdam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=WD)
    if name == "AdaBelief":
        return Z.AdaBelief(params, lr=lr, weight_decay=WD)
    if name == "Adafactor":
        return Z.Adafactor(params, lr=lr, weight_decay=WD)
    if name == "Lion":
        return Z.Lion(params, lr=lr, weight_decay=WD)
    if name == "MultiEMA-8":
        return Z.MultiEMAAdamW(params, lr=lr, M=8, mu_target=mu, kind="matched",
                               weight_decay=WD)
    if name == "Boxcar-MM":
        W = K.boxcar_window_for_mean_delay(mu)
        return Z.KernelAdamW(params, lr=lr, kernel=np.ones(W), weight_decay=WD)
    if name == "F-Adam":                          # untempered, J=64 (original baseline)
        return Z.KernelAdamW(params, lr=lr, kernel=K.w_alpha(0.7, 63), weight_decay=WD)
    if name == "F-Adam-MM":                       # untempered, MATCHED mean delay
        J = ctx["fadam_J_matched"]
        return Z.KernelAdamW(params, lr=lr, kernel=K.w_alpha(0.7, J - 1), weight_decay=WD)
    if name == "SOE-TF-AdamW":
        return Z.SOETFAdamW(params, lr=lr, fit=fit, weight_decay=WD)
    if name == "SOE-TF-AdamW-AMS":
        return Z.SOETFAdamW(params, lr=lr, fit=fit, weight_decay=WD, amsgrad=True)
    if name.startswith("SOE-TF-AdamW-M="):
        M = int(name.split("=")[1])
        return Z.SOETFAdamW(params, lr=lr, fit=ctx["fits"][M], weight_decay=WD)
    if name.startswith("SOE-TF-AdamW-al="):       # alpha/lambda grid cell
        a, l = name.split("=")[1].split("_")
        return Z.SOETFAdamW(params, lr=lr, fit=ctx["fits_al"][(float(a), float(l))],
                            weight_decay=WD)
    if name == "FracAdam-Shin":
        return Z.FracAdamShin(params, lr=lr, weight_decay=WD)
    if name == "lambda-FAdaMax":
        return Z.LambdaFAdaMax(params, lr=lr, weight_decay=WD)
    if name == "FOSGD-ME":
        return Z.FOSGDME(params, lr=lr, weight_decay=WD)
    if name == "TFGD":
        return Z.TFGD(params, lr=lr, fit=fit, weight_decay=WD)
    if name == "SAM-AdamW":
        base = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
        return Z.SAMWrapper(base, params, rho=0.05)
    raise ValueError(name)


def make_ctx(alpha=0.7, lam=0.05, M=8, extra_M=(), extra_al=()):
    fit = K.soe_fit_nnls(alpha, lam, M)
    mu = K.mean_delay_tf(alpha, lam)
    # J such that the truncated UNTEMPERED power-law kernel has the same mean delay
    J = 2
    while True:
        w = K.w_alpha(alpha, J - 1)
        if (np.arange(J) * w).sum() / w.sum() >= mu or J > 4000:
            break
        J += 1
    ctx = dict(fit=fit, mu=float(mu), alpha=alpha, lam=lam, M=M, fadam_J_matched=int(J))
    if extra_M:
        ctx["fits"] = {m: K.soe_fit_nnls(alpha, lam, m) for m in extra_M}
    if extra_al:
        ctx["fits_al"] = {(a, l): K.soe_fit_nnls(a, l, M) for a, l in extra_al}
    return ctx


# ------------------------------------------------------------------ training
def _eval(model, X, y, bs=1024):
    model.eval()
    with torch.no_grad():
        pr = [model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]
    return float((torch.cat(pr) == y).float().mean())


def train_vision(job, ctx):
    """job: dataset, model, opt, lr, noise, seed, noise_seed, epochs, batch, n_train"""
    ds = job["dataset"]
    torch.manual_seed(job["seed"]); np.random.seed(job["seed"])
    if ds == "fmnist":
        (Xtr, ytr), (Xva, yva), (Xte, yte) = load_fmnist(
            job.get("n_train", 12000), job.get("n_val", 3000), job.get("n_test", 5000))
        shape = (-1, 1, 28, 28); Model = SmallCNN
    else:
        (Xtr, ytr), (Xva, yva), (Xte, yte) = load_cifar(
            job.get("n_train", 45000), job.get("n_val", 5000), job.get("n_test", 10000))
        shape = (-1, 3, 32, 32); Model = ResNet20
    ytr = add_label_noise(ytr, job.get("noise", 0.0), 10, job.get("noise_seed", 0))
    Xtr_t = torch.tensor(Xtr).view(*shape); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva).view(*shape); yva_t = torch.tensor(yva)
    Xte_t = torch.tensor(Xte).view(*shape); yte_t = torch.tensor(yte)
    model = Model()
    opt = make_opt(job["opt"], model.parameters(), job["lr"], ctx)
    sam = isinstance(opt, Z.SAMWrapper)
    n = len(Xtr_t); idx = np.arange(n); rng = np.random.default_rng(job["seed"])
    batch = job.get("batch", 128)
    sched = None
    if job.get("sched") == "cosine" and not sam:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=job["epochs"])
    losses = []
    for ep in range(job["epochs"]):
        model.train(); rng.shuffle(idx); tot = 0.0; nb = 0
        for b in range(0, n, batch):
            bi = idx[b:b + batch]
            xb, yb = Xtr_t[bi], ytr_t[bi]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            if sam:
                opt.first_step(); opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward(); opt.second_step()
            else:
                opt.step()
            tot += float(loss); nb += 1
        if sched is not None:
            sched.step()
        losses.append(tot / nb)
    return dict(val=_eval(model, Xva_t, yva_t), test=_eval(model, Xte_t, yte_t),
                losses=losses)


def _batch_lm(data, bs, L, rng):
    ix = rng.integers(0, len(data) - L - 1, bs)
    x = np.stack([data[i:i + L] for i in ix])
    y = np.stack([data[i + 1:i + L + 1] for i in ix])
    return torch.tensor(x), torch.tensor(y)


def _lm_loss(model, data, L, rng, nb=20, bs=32):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for _ in range(nb):
            x, y = _batch_lm(data, bs, L, rng)
            tot += float(F.cross_entropy(model(x).reshape(-1, model.o.out_features),
                                         y.reshape(-1)))
    return tot / nb


def train_lm(job, ctx):
    """Character-level Transformer language model; reports validation/test loss
    (cross-entropy in nats per character).  Lower is better."""
    torch.manual_seed(job["seed"]); np.random.seed(job["seed"])
    tr, va, te, V = load_text()
    L = job.get("seqlen", 128); bs = job.get("batch", 32)
    model = CharTransformer(V, d=job.get("d", 128), nl=job.get("nl", 2), L=L)
    if job.get("noise", 0.0) > 0:                # input-token corruption
        rate = job["noise"]
        rngn = np.random.default_rng(10000 + job.get("noise_seed", 0))
        tr = tr.copy()
        fl = rngn.random(len(tr)) < rate
        tr[fl] = rngn.integers(0, V, int(fl.sum()))
    opt = make_opt(job["opt"], model.parameters(), job["lr"], ctx)
    rng = np.random.default_rng(job["seed"])
    steps = job.get("steps", 3000)
    curve = []
    for k in range(steps):
        model.train()
        x, y = _batch_lm(tr, bs, L, rng)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (k + 1) % max(1, steps // 20) == 0:
            curve.append(float(loss))
    rv = np.random.default_rng(0)
    return dict(val=-_lm_loss(model, va, L, rv, 20, bs),
                test=-_lm_loss(model, te, L, np.random.default_rng(1), 20, bs),
                losses=curve)


def run_job(job, ctx):
    t0 = time.time()
    if job["dataset"] in ("fmnist", "cifar"):
        r = train_vision(job, ctx)
    elif job["dataset"] == "lm":
        r = train_lm(job, ctx)
    else:
        raise ValueError(job["dataset"])
    r["secs"] = round(time.time() - t0, 1)
    r["job"] = job
    return r
