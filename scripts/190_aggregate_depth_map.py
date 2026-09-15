# -*- coding: utf-8 -*-
"""190_aggregate_depth_map.py — mean ± sd across seeds, and the depth trend test.

The depth grid is run once per seed (three invocations writing into the same per-backbone file). This script
aggregates:

  * per (backbone, family, site, operator): mean and sd of the **downstream transfer**
    $T_F = 1 - d(F(Wz), F(g_\\tau z)) / d(F(z), F(g_\\tau z))$, which is the only absolute faithfulness
    score in the grid (1 = matches the real route, 0 = no better than the no-op, < 0 = worse), together with
    the companion statistics (projection, win rate, top-1 agreement) as mean ± sd;
  * a per (backbone, family) **depth trend test**: Spearman rank correlation between depth and transfer over
    all sites and seeds, with its two-sided p-value, so that the paper can claim a degradation *trend*
    rather than a monotone law;
  * a flag for sites that are causally unreachable from the consumer.

Usage: python scripts/190_aggregate_depth_map.py
Outputs results/depth_map_summary.json and paper/targets/TPAMI/DEPTH_MAP_TABLES.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, glob
import numpy as np
from scipy import stats

WORK = rp.REPO_ROOT
BACKBONES = ["resnet50", "convnext", "vitb16", "dinov2b14"]
LABEL = {"resnet50": "ResNet-50", "convnext": "ConvNeXt-T", "vitb16": "ViT-B/16", "dinov2b14": "DINOv2-B/14"}
OPS = ["O1", "O2", "O8", "O6"]
# both criteria travel together: the operator families are fitted by least squares, i.e. they are optimal
# for transfer_squared, while the reported headline is the norm-ratio transfer (see 198 for the control)
FIELDS = ["transfer", "transfer_squared", "proj_coef_agg", "win_rate", "top1_agree_real"]


def load(b, fam):
    p = os.path.join(WORK, "results", f"depth_map_{b}_{fam}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    summary, lines = {}, []
    for fam in ("hue", "heat"):
        key_prefix = "hue_" if fam == "hue" else "heat_"
        for b in BACKBONES:
            d = load(b, fam)
            if not d:
                continue
            for key, block in d["results"].items():
                if not key.startswith(key_prefix) or "seed" not in block:
                    continue
                sites = d["site_names"]
                seed_keys = sorted(block["seed"])
                agg = {}
                for si, site in enumerate(sites, start=1):
                    recs = [block["seed"][sk]["sites"].get(site) for sk in seed_keys]
                    recs = [r for r in recs if r]
                    if not recs:
                        continue
                    reach = all(r.get("intervention_reaches_consumer", True) for r in recs)
                    entry = {"depth": si, "n_seeds": len(recs), "reachable": reach}
                    # the site-level diagnostics travel with the operator metrics: the canonical spatial
                    # action's residual and the channel-map residual, both as relative residuals
                    for extra in ("intertwining_residual_canonical",
                                  "intertwining_residual_after_global_linear", "effect_size_site"):
                        v = [r.get(extra) for r in recs if r.get(extra) is not None]
                        if v:
                            entry[extra] = float(np.mean(v))
                            entry[extra + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                    for op in OPS:
                        if not all(op in r for r in recs):
                            continue
                        for f_ in FIELDS:
                            v = [r[op].get(f_) for r in recs if r[op].get(f_) is not None]
                            if v:
                                entry[f"{op}_{f_}_mean"] = float(np.mean(v))
                                entry[f"{op}_{f_}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                    agg[site] = entry
                # depth trend: Spearman over (depth, transfer) pooled across sites and seeds
                for op in OPS:
                    xs, ys = [], []
                    for si, site in enumerate(sites, start=1):
                        if site not in agg or not agg[site]["reachable"]:
                            continue
                        for sk in seed_keys:
                            r = block["seed"][sk]["sites"].get(site)
                            if r and op in r and r[op].get("transfer") is not None:
                                xs.append(si); ys.append(r[op]["transfer"])
                    if len(xs) >= 4 and len(set(xs)) > 1:
                        rho, p = stats.spearmanr(xs, ys)
                        agg.setdefault("_trend", {})[op] = {"spearman_rho": float(rho), "p": float(p),
                                                           "n": len(xs)}
                summary[f"{b}|{key}"] = {"backbone": b, "family_key": key, "n_seeds": len(seed_keys),
                                         "sites": {k: v for k, v in agg.items() if k != "_trend"},
                                         "depth_trend": agg.get("_trend", {})}
                lines.append((b, key, seed_keys, agg))

    json.dump(summary, open(os.path.join(WORK, "results", "depth_map_summary.json"), "w"), indent=1)

    L = ["# TPAMI depth map: aggregated over seeds", "",
         "**Downstream transfer** $T_F = 1 - d(F(Wz),F(g_\\tau z))/d(F(z),F(g_\\tau z))$ is the absolute",
         "faithfulness score: 1 = the intervention matches the real route, 0 = no better than doing nothing,",
         "< 0 = worse than nothing. Projection and win rate are companion statistics. `unreachable` marks a",
         "site no intervention can reach (class-token head at the final block).", ""]
    for b, key, seed_keys, agg in lines:
        L += [f"## {LABEL[b]} — {key} ({len(seed_keys)} seeds)", "",
              "| site | depth | transfer (O1/O2/O8/O6) | transfer$^2$ (O2) | projection (O2) | win (O2) | reachable |",
              "|---|---|---|---|---|---|---|"]
        for site, e in agg.items():
            if site == "_trend":
                continue
            t = " / ".join(f"{e.get(f'{op}_transfer_mean', float('nan')):.2f}" for op in OPS)
            L.append(f"| {site} | {e['depth']} | {t} | "
                     f"{e.get('O2_transfer_squared_mean', float('nan')):.3f} | "
                     f"{e.get('O2_proj_coef_agg_mean', float('nan')):.3f} ± "
                     f"{e.get('O2_proj_coef_agg_sd', 0):.3f} | "
                     f"{e.get('O2_win_rate_mean', float('nan')):.2f} | {'yes' if e['reachable'] else 'NO'} |")
        tr = agg.get("_trend", {})
        if tr:
            L += ["", "depth trend (Spearman over sites × seeds): " +
                  "; ".join(f"{op} ρ={v['spearman_rho']:+.2f} (p={v['p']:.3g}, n={v['n']})"
                            for op, v in tr.items()), ""]
    open(os.path.join(WORK, "paper", "targets", "TPAMI", "DEPTH_MAP_TABLES.md"), "w",
         encoding="utf-8").write("\n".join(L) + "\n")
    print(f"depth-map aggregate: PASS (0 issues)")
    print(f"  {len(summary)} (backbone, family) blocks aggregated")
    for k, v in summary.items():
        tr = v["depth_trend"].get("O2", {})
        print(f"  {k}: seeds {v['n_seeds']}, O2 transfer trend rho={tr.get('spearman_rho', float('nan')):+.2f} "
              f"(p={tr.get('p', float('nan')):.3g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
