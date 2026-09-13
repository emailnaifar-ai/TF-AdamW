"""Minimum check that the repository is functional.

Fits the sum-of-exponentials kernel, verifies the closed-form identities the paper proves,
and trains a small model for a few steps with SOE-TF-AdamW.  Runs in well under a minute.

    python smoke_test.py
"""
import numpy as np
import kernels as K


def main():
    alpha, lam, M = 0.7, 0.05, 8

    d = K.d_mass(alpha, lam)
    kap = K.tempered_kernel(alpha, lam, 20000)
    assert abs(kap.sum() - d) / d < 1e-9, "normalization identity"
    mu = K.mean_delay_tf(alpha, lam)
    assert abs((np.arange(len(kap)) * kap).sum() / kap.sum() - mu) / mu < 1e-6, "mean delay"
    print(f"kernel identities hold: d = {d:.6f}, mean delay = {mu:.4f}")

    fit = K.soe_fit_nnls(alpha, lam, M)
    print(f"SOE fit at M = {M}: fitted error = {fit['eps_fit']:.3e}, "
          f"certified bound = {fit['eps_certified']:.3e} "
          f"(the paper reports 6.03e-06)")
    assert fit["eps_certified"] < 1e-4, "sum-of-exponentials fit"
    assert (fit["c"] > 0).all() and (fit["rho"] < 1).all(), "nonnegative, stable states"

    ess_tf = K.ess_tf(alpha, lam)
    ess_ex = K.ess_ema(K.beta_for_mean_delay(mu))
    print(f"effective sample size: tempered fractional {ess_tf:.3f}, "
          f"exponential at the same mean delay {ess_ex:.3f} "
          f"(ratio {ess_tf / ess_ex:.3f}, the paper reports 0.810)")
    assert ess_tf < ess_ex, "at matched mean delay the exponential kernel has the larger ESS"

    try:
        import torch
        import torch.nn.functional as F
        import opt_zoo as Z
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 3))
        opt = Z.SOETFAdamW(model.parameters(), lr=1e-3, fit=fit, weight_decay=5e-4)
        x, y = torch.randn(128, 20), torch.randint(0, 3, (128,))
        first = last = None
        for step in range(50):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss)
            last = float(loss)
        print(f"SOE-TF-AdamW trains: loss {first:.4f} -> {last:.4f} over 50 steps")
        assert last < first, "loss should decrease"
    except ImportError:
        print("torch not installed; skipped the optimizer check")

    print("\nOK")


if __name__ == "__main__":
    main()
