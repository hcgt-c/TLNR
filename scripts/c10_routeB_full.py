# -*- coding: utf-8 -*-
"""Route B full runner: sequential 100-epoch x 3-seed x 5-arm CIFAR-10 color-robustness benchmark.
Launched as a background job; per (arm,seed) JSON written incrementally so partial credit survives.
Also writes results/c10_routeB/summary.json once all arms finish.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, time, subprocess, argparse

WORK = rp.REPO_ROOT
PY = sys.executable
SCRIPT = os.path.join(WORK, "scripts", "c10_routeB_train.py")
OUT = os.path.join(WORK, "results", "c10_routeB")

ARMS = ["z2", "z2hue", "ocode", "ce8", "lcer8"]
SEEDS = [0, 1, 2]
EPOCHS = 100


def run_one(arm, seed, epochs, jitter=None):
    logf = os.path.join(OUT, f"{arm}_s{seed}.log")
    cmd = [PY, SCRIPT, "--arm", arm, "--seed", str(seed), "--epochs", str(epochs)]
    if jitter is not None:
        cmd += ["--jitter", str(jitter)]
    print(f"[driver] start {arm} s{seed} {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    with open(logf, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[driver] {arm} s{seed} done in {(time.time()-t0)/60:.1f} min rc={r.returncode}", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--jitter", type=float, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",")]
    os.makedirs(OUT, exist_ok=True)
    for arm in arms:
        for seed in seeds:
            fn = os.path.join(OUT, f"{arm}_s{seed}.json")
            if a.skip_existing and os.path.exists(fn):
                print(f"[driver] skip existing {fn}", flush=True)
                continue
            run_one(arm, seed, a.epochs, a.jitter)


if __name__ == "__main__":
    main()
