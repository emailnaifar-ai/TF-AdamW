"""Complete the outstanding experimental stages under the stated tuning rule.

For every experiment: tune, then widen the learning-rate grid until no (method, noise)
cell selects a boundary value, then evaluate.  Every stage resumes: a job already
recorded without an error in the target directory is skipped, so this script can be
re-run after an interruption.

    python finish.py                 # all stages, in the order below
    python finish.py cifar fmnist    # a subset
"""
import os, sys, subprocess, time, glob, json
import keepawake

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finish.log")


class _Tee:
    """Write to the console and to finish.log, flushing every line.

    The log used to be produced by `>> finish.log` in the launcher. That is not reliable
    for a run started with no console: an entire relaunch once executed several stages and
    left the log byte-for-byte unchanged, so there was no way to see what had happened.
    Owning the file here means a line is on disk as soon as it is printed, whoever started
    the run and however it was started.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", encoding="utf-8", errors="replace")

    def write(self, s):
        try:
            self.stream.write(s)
            self.stream.flush()
        except Exception:
            pass                                  # no console attached: the file is enough
        self.fh.write(s)
        self.fh.flush()
        os.fsync(self.fh.fileno())                # publish the size to other readers
        return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        self.fh.flush()

PY = sys.executable
# (experiment, workers).  CIFAR holds a 45k x 3072 float32 array per worker, so it
# runs with fewer of them than the small-image stages.
# Cheapest first: the Fashion-MNIST, rank, heat-map and language-model stages settle the
# numbers the manuscript quotes most, and finish in a few hours; CIFAR is the long pole and
# goes last so an interruption costs the least-load-bearing result.
STAGES = [("fmnist", 7), ("rank", 7), ("heat", 7), ("lm", 7), ("cifar", 6)]
MAX_WIDEN_ROUNDS = 6


def sh(*a, fatal=True):
    """Run a pipeline step.  rc 3 means extend_grid found nothing left to widen.

    Any other nonzero return is treated as fatal by default: when the environment breaks
    (a logoff kills every child process, say) the loop must stop rather than carry on and
    select learning rates from a half-measured grid.
    """
    print(">>>", " ".join(str(x) for x in a), flush=True)
    r = subprocess.run([PY] + [str(x) for x in a],
                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    if r.returncode not in (0, 3):
        print("!!! failed:", a, "rc=", r.returncode, flush=True)
        if fatal:
            raise SystemExit(f"aborting: {a[0]} failed with rc={r.returncode}")
    return r.returncode


def n_good(pattern):
    n = 0
    for f in glob.glob(pattern):
        for line in open(f):
            try:
                if "error" not in json.loads(line):
                    n += 1
            except Exception:
                pass
    return n


def stage(exp, W):
    # Always rebuild and re-dispatch the planned tuning grid.  The dispatcher skips every
    # job already recorded without an error, so this is a no-op for a finished stage and
    # completes the remainder of one that was interrupted -- widening a grid that is only
    # half measured would select learning rates from an incomplete search.
    sh("build_jobs.py", "tune", exp)
    sh("runner.py", "dispatch", f"jobs_{exp}_tune.json", f"out_{exp}_tune", W)
    for r in range(MAX_WIDEN_ROUNDS):
        if sh("extend_grid.py", exp) == 3:
            print(f"[{exp}] grid rule satisfied after {r} widening round(s)", flush=True)
            break
        sh("runner.py", "dispatch", f"jobs_{exp}_tune.json", f"out_{exp}_tune", W)
    else:
        print(f"[{exp}] WARNING: still widening after {MAX_WIDEN_ROUNDS} rounds", flush=True)
    if sh("build_jobs.py", "eval", exp, fatal=False):
        return
    sh("runner.py", "dispatch", f"jobs_{exp}_eval.json", f"out_{exp}_eval", W)


if __name__ == "__main__":
    want = sys.argv[1:]
    sys.stdout = _Tee(sys.stdout, LOG)
    sys.stderr = sys.stdout
    print(f"\n===== run started {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(pid {os.getpid()}) =====", flush=True)
    # Without this the machine drops into Modern Standby a minute after the user walks
    # away and the run is suspended and then torn down; see keepawake.py.
    keepawake.hold("the experimental pipeline")
    t0 = time.time()
    for exp, W in STAGES:
        if want and exp not in want:
            continue
        print(f"\n===== {exp} =====  [{(time.time()-t0)/3600:.2f} h elapsed]", flush=True)
        stage(exp, W)
        print(f"===== {exp} DONE [{(time.time()-t0)/3600:.2f} h] =====", flush=True)
    print(f"\nALL STAGES DONE in {(time.time()-t0)/3600:.2f} h", flush=True)
