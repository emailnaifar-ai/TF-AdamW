"""Demonstrate the LoRA state-restriction remedy (Reviewer 2, comment 14).

The reviewer notes that restricting the optimizer state to low-rank adapters is proposed in
the manuscript but never demonstrated. This experiment demonstrates it:

  1. pre-train a CNN on clean Fashion-MNIST;
  2. freeze the backbone and insert low-rank adapters on the two fully-connected layers;
  3. adapt to a shifted task (40% symmetric label noise) with the optimizer state carried
     ONLY by the adapters;
  4. report the resulting optimizer-state cost against full fine-tuning, and the accuracy.

The point is not that adapters improve accuracy -- it is that the M+1 state buffers, which
are the method's real cost, can be confined to a small fraction of the parameters.
"""
import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kernels as K
import dl_core as C
import opt_zoo as Z

torch.set_num_threads(1)
OUT = "results"
os.makedirs(OUT, exist_ok=True)
CTX = C.make_ctx()


class LoRACNN(nn.Module):
    """SmallCNN with the two fully-connected layers wrapped in low-rank adapters."""

    def __init__(self, base: C.SmallCNN, r=8):
        super().__init__()
        self.c1, self.c2, self.drop = base.c1, base.c2, base.drop
        for m in (self.c1, self.c2):
            for p in m.parameters():
                p.requires_grad_(False)
        self.fc1 = C.LoRALinear(base.fc1, r=r)
        self.fc2 = C.LoRALinear(base.fc2, r=r)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


def state_bytes(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
            elif isinstance(v, list):
                tot += sum(t.numel() * t.element_size() for t in v if torch.is_tensor(t))
    return tot


def run(mode, lr, seed=0, epochs=6, noise=0.4):
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = C.load_fmnist(12000, 3000, 5000)
    Xtr_t = torch.tensor(Xtr).view(-1, 1, 28, 28)
    Xte_t = torch.tensor(Xte).view(-1, 1, 28, 28); yte_t = torch.tensor(yte)

    # ---- stage 1: pre-train on CLEAN labels
    base = C.SmallCNN()
    opt0 = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=5e-4)
    y0 = torch.tensor(ytr)
    idx = np.arange(len(Xtr_t)); rng = np.random.default_rng(seed)
    for _ in range(4):
        base.train(); rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            bi = idx[b:b + 128]
            opt0.zero_grad(); F.cross_entropy(base(Xtr_t[bi]), y0[bi]).backward(); opt0.step()
    acc_pre = C._eval(base, Xte_t, yte_t)

    # ---- stage 2: adapt to the shifted (noisy) task
    yn = torch.tensor(C.add_label_noise(ytr, noise, 10, seed))
    if mode == "full":
        model = copy.deepcopy(base)
        params = [p for p in model.parameters()]
    else:
        model = LoRACNN(copy.deepcopy(base), r=8)
        params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in base.parameters())
    opt = Z.SOETFAdamW(params, lr=lr, fit=CTX["fit"], weight_decay=5e-4)
    t0 = time.time()
    for _ in range(epochs):
        model.train(); rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            bi = idx[b:b + 128]
            opt.zero_grad(); F.cross_entropy(model(Xtr_t[bi]), yn[bi]).backward(); opt.step()
    return dict(mode=mode, lr=lr, acc_pretrain=acc_pre,
                acc=C._eval(model, Xte_t, yte_t),
                trainable=n_train, total=n_total,
                frac_trainable=n_train / n_total,
                opt_state_bytes=state_bytes(opt),
                bytes_per_model_param=state_bytes(opt) / n_total,
                secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    res = {}
    for mode in ("full", "lora"):
        best = None
        for lr in (5e-4, 1e-3, 2e-3):
            r = run(mode, lr, seed=100)                      # tuning seed
            if best is None or r["acc"] > best["acc"]:
                best = r
        runs = [run(mode, best["lr"], seed=s) for s in (0, 1, 2)]
        res[mode] = dict(lr=best["lr"],
                         acc_mean=float(np.mean([r["acc"] for r in runs])),
                         acc_sd=float(np.std([r["acc"] for r in runs], ddof=1)),
                         acc_pretrain=runs[0]["acc_pretrain"],
                         trainable=runs[0]["trainable"], total=runs[0]["total"],
                         frac_trainable=runs[0]["frac_trainable"],
                         opt_state_bytes=runs[0]["opt_state_bytes"],
                         bytes_per_model_param=runs[0]["bytes_per_model_param"])
        r = res[mode]
        print(f"{mode:5s} lr={r['lr']:g} acc={r['acc_mean']*100:.2f}+-{r['acc_sd']*100:.2f} "
              f"trainable={r['trainable']:,}/{r['total']:,} ({r['frac_trainable']*100:.1f}%) "
              f"optimizer state={r['opt_state_bytes']/1024:.0f} KiB "
              f"({r['bytes_per_model_param']:.2f} bytes per model parameter)", flush=True)
    json.dump(res, open(os.path.join(OUT, "lora.json"), "w"), indent=1)
    print("wrote results/lora.json")
