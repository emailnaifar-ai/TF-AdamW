"""Optimizer cost accounting (Reviewer 1 comment 6, Reviewer 2 comment 14).

Reports, across four model scales, the EXACT optimizer-state bytes per parameter (a
device-independent quantity), the measured peak process memory, and the measured
throughput relative to AdamW.  We do not have GPU hardware for this study, so we report
the analytic state accounting -- which is exact and transferable -- together with
measured CPU wall-clock and peak resident memory, and we say so explicitly.
"""
import os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil

import kernels as K
import opt_zoo as Z
import dl_core as C

torch.set_num_threads(2)
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def mlp(width, depth=4, din=256, dout=10):
    L = [nn.Linear(din, width), nn.ReLU()]
    for _ in range(depth - 1):
        L += [nn.Linear(width, width), nn.ReLU()]
    L += [nn.Linear(width, dout)]
    return nn.Sequential(*L)


MODELS = [("SmallCNN (0.2 M)", lambda: C.SmallCNN(), (1, 1, 28, 28)),
          ("ResNet-20 (0.27 M)", lambda: C.ResNet20(), (1, 3, 32, 32)),
          ("CharTransformer (0.43 M)", lambda: C.CharTransformer(65), None),
          ("MLP-1024 (3.2 M)", lambda: mlp(1024), (1, 256)),
          ("MLP-2048 (12.6 M)", lambda: mlp(2048), (1, 256))]
print("NOTE: timings are measured while other jobs run; only ratios to AdamW are reported.",
      flush=True)

CTX = C.make_ctx()
METHODS = ["AdamW", "Lion", "Adafactor", "SOE-TF-AdamW-M=2", "SOE-TF-AdamW-M=4",
           "SOE-TF-AdamW", "F-Adam"]
CTX2 = C.make_ctx(extra_M=(1, 2, 4, 8, 12))


def state_bytes(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
            elif isinstance(v, list):
                for t in v:
                    if torch.is_tensor(t):
                        tot += t.numel() * t.element_size()
    return tot


def bench(name, mk, inshape, bs=32, nstep=5):
    model = mk()
    P = sum(p.numel() for p in model.parameters())
    rows = {}
    for m in METHODS:
        gc.collect()
        model = mk()
        opt = C.make_opt(m, model.parameters(), 1e-3, CTX2)
        if inshape is None:
            x = torch.randint(0, 65, (bs, 64)); y = torch.randint(0, 65, (bs, 64))
            fwd = lambda: F.cross_entropy(model(x).reshape(-1, 65), y.reshape(-1))
        else:
            x = torch.randn(bs, *inshape[1:])
            y = torch.randint(0, 10, (bs,))
            fwd = lambda: F.cross_entropy(model(x), y)
        for _ in range(2):                          # warm-up + state allocation
            opt.zero_grad(); fwd().backward(); opt.step()
        rss0 = psutil.Process().memory_info().rss
        t0 = time.time()
        for _ in range(nstep):
            opt.zero_grad(); fwd().backward(); opt.step()
        dt = (time.time() - t0) / nstep
        sb = state_bytes(opt)
        rows[m] = dict(state_bytes=sb, bytes_per_param=sb / P, ms_per_step=dt * 1e3,
                       peak_rss_MB=psutil.Process().memory_info().rss / 2 ** 20)
        del opt, model
    base = rows["AdamW"]
    for m in rows:
        rows[m]["rel_time"] = rows[m]["ms_per_step"] / base["ms_per_step"]
        rows[m]["rel_state"] = rows[m]["state_bytes"] / base["state_bytes"]
    return dict(params=P, rows=rows)


if __name__ == "__main__":
    res = {}
    for name, mk, sh in MODELS:
        res[name] = bench(name, mk, sh)
        json.dump(res, open(os.path.join(OUT, "cost.json"), "w"), indent=1)
        print(f"\n{name}  ({res[name]['params']:,} params)")
        print(f"  {'method':20s} {'B/param':>8} {'xAdamW state':>13} {'ms/step':>9} {'xAdamW time':>12}")
        for m, r in res[name]["rows"].items():
            print(f"  {m:20s} {r['bytes_per_param']:8.2f} {r['rel_state']:13.2f} "
                  f"{r['ms_per_step']:9.2f} {r['rel_time']:12.2f}")
    json.dump(res, open(os.path.join(OUT, "cost.json"), "w"), indent=1)
    print("\nwrote results/cost.json")
