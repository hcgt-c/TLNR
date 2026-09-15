# -*- coding: utf-8 -*-
"""Aggregate Route B results: per-arm mean/std over seeds of clean acc, mean OOD sweep acc,
full 37-point curve (mean over seeds), and a compact honest summary table.
Usage: python scripts/c10_routeB_summary.py [results_dir]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json
import numpy as np

WORK = rp.REPO_ROOT
OUT = os.path.join(WORK, "results", "c10_routeB")

ARM_ORDER = ["z2", "z2hue", "ce8", "lcer8", "ocode"]
ARM_LABEL = {
    "z2": "plain CNN (ResNet-44)",
    "z2hue": "plain + hue-jitter aug",
    "ce8": "CEConv (rotations=8)",
    "lcer8": "LCER (G_h=8 lifting)",
    "ocode": "ours: + harmonic code aux",
}


def main():
    files = sorted(os.listdir(OUT))
    by_arm = {}
    for f in files:
        if not f.endswith(".json") or f == "summary.json" or f.startswith("_"):
            continue
        d = json.load(open(os.path.join(OUT, f)))
        by_arm.setdefault(d["arm"], []).append(d)
    rows = []
    for arm in ARM_ORDER:
        if arm not in by_arm:
            continue
        ds = sorted(by_arm[arm], key=lambda x: x["seed"])
        clean = [d["clean_acc"] for d in ds]
        msweep = [d["mean_sweep_acc"] for d in ds]
        # factor grid union (use keys of the last d with most points)
        grid = sorted({float(k) for d in ds for k in d["sweep"]})
        curve = []
        for f in grid:
            vals = [d["sweep"][f"{f:.4f}"] for d in ds if f"{f:.4f}" in d["sweep"]]
            if vals:
                curve.append((f, float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0))
        nparam = ds[0]["n_params"]
        rows.append({
            "arm": arm, "label": ARM_LABEL.get(arm, arm), "n_seeds": len(ds),
            "n_params": nparam,
            "clean_mean": float(np.mean(clean)), "clean_std": float(np.std(clean)),
            "ood_mean": float(np.mean(msweep)), "ood_std": float(np.std(msweep)),
            "per_seed_clean": clean, "per_seed_oodmean": msweep,
            "curve": [list(c) for c in curve],
            "wall_seconds": sum(d["wall_seconds"] for d in ds),
        })
    print("=" * 100)
    print(f"{'arm':8} {'seeds':5} {'params(M)':9} {'clean mean±std':>20} {'OODsweep mean±std':>24} {'wall(h)':>9}")
    for r in rows:
        print(f"{r['arm']:8} {r['n_seeds']:<5} {r['n_params']/1e6:<9.3f} "
              f"{r['clean_mean']:.4f}±{r['clean_std']:.4f}     {r['ood_mean']:.4f}±{r['ood_std']:.4f}     {r['wall_seconds']/3600:7.1f}")
    print("=" * 100)
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    # full curve print (seed-mean) at ~10-deg resolution
    for r in rows:
        step = max(1, int(round(len(r["curve"]) / 18)))
        pts = [(round(f * 360), a, sd) for f, a, sd in r["curve"]]
        print(r["arm"], "deg/acc:", " ".join(f"{d:+.0f}:{a:.3f}" for d, a, _ in pts[::step]))
    return rows


if __name__ == "__main__":
    main()
