"""Regenerate everything downstream of the experiment results, in dependency order.

Run this after any experiment stage finishes.  It re-derives the statistics, the tables,
the figures and the three response documents, rebuilds both PDFs, and then runs the two
consistency gates.  Anything that moved shows up as a named failure rather than as a
silent disagreement between the prose and the tables.

    python refresh.py            # everything
    python refresh.py --no-build # skip the LaTeX builds (fast, for a numbers-only check)
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.environ.get("TFADAMW_REV", HERE)
PY = sys.executable
# (experiment, reference optimizer for the paired comparisons).  The heat map compares
# every (alpha, lambda) cell against the recommended default cell; the rank sweep compares
# every M against the recommended M=8.
EXPS = [("fmnist", "AdamW"), ("heat", "SOE-TF-AdamW-al=0.7_0.05"),
        ("rank", "SOE-TF-AdamW-M=8"), ("lm", "AdamW"), ("cifar", "AdamW")]


def run(cwd, *a, quiet=True):
    print(">>>", " ".join(str(x) for x in a), flush=True)
    r = subprocess.run([PY] + [str(x) for x in a], cwd=cwd,
                       capture_output=quiet, text=True)
    if quiet and r.returncode:
        print((r.stdout or "")[-1500:], (r.stderr or "")[-800:])
    elif not quiet:
        pass
    return r


if __name__ == "__main__":
    build = "--no-build" not in sys.argv
    for exp, ref in EXPS:
        if os.path.isdir(os.path.join(HERE, f"out_{exp}_eval")):
            run(HERE, "analyze.py", exp, ref)
    run(HERE, "make_tables.py")
    run(HERE, "make_figures.py")
    run(HERE, "makedocs.py")
    if build:
        r = subprocess.run([PY, "build.py"], cwd=REV, capture_output=True, text=True)
        print(r.stdout.strip())
    print("\n=== prose vs results ===", flush=True)
    subprocess.run([PY, "check_numbers.py"], cwd=HERE)
    print("\n=== publication readiness ===", flush=True)
    subprocess.run([PY, "final_check.py"], cwd=REV)
