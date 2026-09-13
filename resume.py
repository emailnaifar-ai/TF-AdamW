"""Restart the experiment pipeline exactly where it stopped.

Safe to run after a reboot, or at any time.  Every job already completed is recorded in
out_<exp>_<tune|eval>/shard*.jsonl and is skipped, so nothing is recomputed; at most the
handful of jobs that were in flight when the machine went down are repeated.

    python resume.py            # continue the whole programme
    python resume.py fmnist lm  # continue only these experiments
"""
import os, sys, json, glob, subprocess


def status():
    print(f"{'experiment':<14}{'stage':<7}{'done':>7} {'planned':>8}")
    print("-" * 40)
    for e in ("fmnist", "heat", "rank", "lm", "cifar", "fmnistfull"):
        for stage in ("tune", "eval"):
            d = f"out_{e}_{stage}"
            jf = f"jobs_{e}_{stage}.json"
            if not os.path.isdir(d) and not os.path.exists(jf):
                continue
            done = sum(1 for f in glob.glob(os.path.join(d, "shard*.jsonl"))
                       for line in open(f)
                       if '"error"' not in line) if os.path.isdir(d) else 0
            planned = len(json.load(open(jf))["jobs"]) if os.path.exists(jf) else "?"
            print(f"{e:<14}{stage:<7}{done:>7} {str(planned):>8}")
    for f in ("results/synth_instances.json", "results/synth_shape.json",
              "results/geometry.json", "results/soe_error.json", "results/cost.json",
              "results/gradstats.json"):
        print(f"  {'OK ' if os.path.exists(f) else '-- '} {f}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=== state before resuming ===")
    status()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--status" in sys.argv:
        sys.exit(0)
    print("\n=== resuming ===", flush=True)
    # finish.py, not run_all.py: it also widens the learning-rate grid until no method
    # sits on a boundary, which is the protocol the manuscript states.
    subprocess.run([sys.executable, "finish.py"] + args,
                   env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
