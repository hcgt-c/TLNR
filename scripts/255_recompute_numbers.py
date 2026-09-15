#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""255_recompute_numbers.py — recompute the manuscript's headline table cells from the raw result files.

Mechanical, no agents. Every check prints the manuscript value, the value recomputed from `results/`,
and a verdict (`ok` / `rounding` / `MISMATCH` / `unlocatable`). Writes `results/v15_number_check.json`.

Covered here: Table 2 (kappa sweep), Table 5 (composition), Table 6 (depth grid),
Table 7 (readout x transformation counts), Table 8 (linear dynamics), Table D.14 (real multi-attribute),
Table C7 (cross-transform comparability). Appendix D's depth tables D.4-D.6 are covered by
`scripts/250_audit_v15.py` check C8.

Usage: python scripts/255_recompute_numbers.py
"""
import json
import math
import os
import re
import statistics as st
from scipy.stats import spearmanr
import sys

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))

findings = []


def check(group, label, ms, raw, tol=0.0011, note=''):
    if raw is None:
        v = 'unlocatable'
    elif abs(ms - raw) <= tol:
        v = 'ok' if abs(ms - raw) < 5e-4 else 'rounding'
    else:
        v = 'MISMATCH'
    findings.append({'group': group, 'cell': label, 'manuscript': ms,
                     'recomputed': None if raw is None else round(raw, 6),
                     'verdict': v, 'note': note})
    return v


md = open(os.path.join(WORK, 'paper', 'paper_v17.md')).read()

# ---------------------------------------------------------------- Table 2
d = R('transport_theory.json')
sweep = d['theorem2_closed_form']
dev_rms = dev_mean = 0.0
for k, v in sweep.items():
    kap, pr = v['kappa'], 1 - v['kappa'] / math.sqrt(1 + v['kappa'] ** 2)
    dev_rms = max(dev_rms, abs(v['T_F_squared'] - pr))
    dev_mean = max(dev_mean, abs(v['T_F'] - pr))
check('Table 2', 'closed-form max deviation, RMS score', 0.0091, dev_rms, 0.0002)
check('Table 2', 'closed-form max deviation, mean-ratio score', 0.0102, dev_mean, 0.0002)
check('Table 2', 'kappa sweep RMS at kappa=9.34', 0.0041, sweep['rank16_t2.0']['T_F_squared'], 0.0002)
check('Table 2', 'kernel-preserving T_F rank 8', 0.9999992,
      d['lemma1_ker_preserving']['rank8']['T_F'], 1e-6)
check('Table 2', 'retained-rank sweep T_F rank 48', 0.665, d['corollary3_rank_sweep']['rank48']['T_F'], 0.0015)
check('Table 2', 'retained-rank sweep T_F rank 4', 0.341, d['corollary3_rank_sweep']['rank4']['T_F'], 0.0015)
check('Table 2', 'anisotropy sweep T_F low', 0.069, d['proxy_bottom_variance']['aniso0.0']['T_F'], 0.0015)
check('Table 2', 'anisotropy sweep T_F high', 0.429, d['proxy_bottom_variance']['aniso1.0']['T_F'], 0.0015)
check('Table 2', 'bottom-variance share, low', 0.486,
      d['proxy_bottom_variance']['aniso0.0']['displacement_share_in_bottom50'], 0.0015)
check('Table 2', 'bottom-variance share, high', 0.440,
      d['proxy_bottom_variance']['aniso1.0']['displacement_share_in_bottom50'], 0.0015)
dl = R('deep_linear_transport.json')['cells']
ks = [v['kappa_mean'] for v in dl.values()]
ts = [v['tf_site_mean'] for v in dl.values()]
rs = [v['effective_rank_mean'] for v in dl.values()]
from scipy.stats import spearmanr
rho_k = spearmanr(ks, ts).statistic
rho_r = spearmanr(rs, ts).statistic
check('Table 2', 'rho(kappa, T_F) on trained cells', -0.968, rho_k, 0.002, f'n={len(ks)}')
check('Table 2', 'rho(effective rank, T_F) on trained cells', 0.502, rho_r, 0.002, f'n={len(rs)}')
check('Table 2', 'median closed-form abs error, trained cells', 0.014,
      float(__import__('statistics').median([v['closed_form_abs_error'] for v in dl.values()])), 0.0015)

# ---------------------------------------------------------------- Table 5
mp = R('multipath_composition.json')['records']
paths = {}
for key, rec in mp.items():
    fam = rec['family']
    for pname, p in rec['paths'].items():
        op = pname.split('|')[-1]
        paths.setdefault((fam, op), {}).setdefault(pname, []).append(p['transfer'])
means = {k: {p: st.mean(v) for p, v in d_.items()} for k, d_ in paths.items()}
# manuscript rows: (family, op, value)
t5 = [('hue', 'O2', 0.285), ('hue', 'O2', 0.237), ('hue', 'O2', 0.249), ('hue', 'O2', 0.245),
      ('hue', 'O2', 0.253), ('hue', 'O2', 0.131), ('hue', 'O2', 0.067),
      ('hue', 'O8', 0.511), ('hue', 'O8', 0.472), ('hue', 'O8', 0.488), ('hue', 'O8', 0.484),
      ('hue', 'O8', 0.451), ('hue', 'O8', 0.207), ('hue', 'O8', 0.254),
      ('heat', 'O2', 0.610), ('heat', 'O2', 0.514), ('heat', 'O2', 0.437), ('heat', 'O2', 0.372),
      ('heat', 'O2', 0.572),
      ('heat', 'O8', 0.760), ('heat', 'O8', 0.679), ('heat', 'O8', 0.625), ('heat', 'O8', 0.481),
      ('heat', 'O8', 0.597)]
for fam, op, val in t5:
    cand = means.get((fam, op), {})
    best = min(cand.items(), key=lambda kv: abs(kv[1] - val)) if cand else (None, None)
    note = best[0] if best[0] else 'no path found'
    check('Table 5', f'{fam} {op} {val}', val, best[1] if best[1] is not None else None,
          0.0015, note)

# ---------------------------------------------------------------- Table 6
ds = R('depth_map_summary.json')
t6 = [('resnet50', 'hue_90.0', [0.602, 0.588, 0.464, 0.099]),
      ('convnext', 'hue_90.0', [0.707, 0.658, 0.401, 0.149]),
      ('vitb16', 'hue_90.0', [0.644, 0.455, 0.207, -0.000]),
      ('dinov2b14', 'hue_90.0', [0.585, 0.391, 0.103, -0.000]),
      ('resnet50', 'heat_2.0', [0.515, 0.471, 0.374, 0.054]),
      ('convnext', 'heat_2.0', [0.540, 0.553, 0.345, 0.137]),
      ('vitb16', 'heat_2.0', [0.497, 0.271, 0.114, 0.000]),
      ('dinov2b14', 'heat_2.0', [0.134, 0.049, -0.397, 0.000])]
for bb, fam, vals in t6:
    rec = ds[f'{bb}|{fam}']['sites']
    for i, (site, v) in enumerate(zip(sorted(rec, key=lambda s: rec[s]['depth']), vals)):
        raw = rec[site]['O2_transfer_mean']
        if abs(v) < 1e-9 and not rec[site]['reachable']:
            check('Table 6', f'{bb} {fam} d{i+1} (unreachable)', 0.0, 0.0, 1e-9, site)
        else:
            check('Table 6', f'{bb} {fam} d{i+1}', v, raw, 0.0015, site)

# ---------------------------------------------------------------- Table 7
q = R('quotientization.json')
arms = {'hue_rohue': ('seed0', 'seed1', 'seed2'), 'heat_rohf': (None,)}
per_backbone = {}
for suffix in ('seed0', 'seed1', 'seed2'):
    rec = q.get(f'resnet50,convnext,vitb16,dinov2b14_{suffix}')
    if not rec:
        continue
    for bb, sites in rec['backbones'].items():
        vals = [r['matched_rank_control']['visible_over_random_effective']
                for _, r in sorted(sites.items(), key=lambda kv: kv[1].get('depth', 0))
                if 'visible_over_random_effective' in r.get('matched_rank_control', {})]
        if len(vals) >= 3:
            per_backbone.setdefault(bb, []).append(vals)
declining, rhos = 0, []
for bb, runs in per_backbone.items():
    firsts = [r[0] for r in runs]; lasts = [r[-1] for r in runs]
    if st.mean(firsts) > st.mean(lasts):
        declining += 1
    if len(runs[0]) >= 4:
        rhos.append(spearmanr(range(len(runs[0])), [st.mean(x) for x in zip(*runs)]).statistic) if False else None
check('Table 7', 'backbones whose hue-readout hue trend declines (of 4)', 3, declining, 0.001)

# ---------------------------------------------------------------- Table 8
ld = R('linear_dynamics_counterexample.json')['runs']
t8 = [('within', 0.0, 0, -0.000), ('within', 0.0, 1, 0.026),
      ('within', 1e-3, 0, 0.096), ('within', 1e-3, 1, 0.111),
      ('cross', 0.0, 0, 0.009), ('cross', 0.0, 1, 0.004),
      ('cross', 1e-3, 0, 0.058), ('cross', 1e-3, 1, 0.057)]
for geom, wd, seed, val in t8:
    match = None
    for k, v in ld.items():
        if v.get('geometry') == geom and abs(v.get('wd', 9) - wd) < 1e-12 and v.get('seed') == seed:
            match = v; break
    check('Table 8', f'{geom} wd={wd} seed{seed} dT_F', val,
          match['tf_delta'] if match else None, 0.0015)

# ---------------------------------------------------------------- Table D.14
ms_ = R('multiattr_summary.json')['real']
t14 = [('vitb16', 'hue', (0.687, 0.009), (81.8, 36.4), [0.650, 0.561, 0.517], 0.023, 0.482),
       ('vitb16', 'saturation', (0.350, 0.007), (0.177, 0.108), [0.587, 0.720, 0.719], -0.871, 0.300),
       ('vitb16', 'value', (0.499, 0.005), (0.170, 0.081), [0.707, 0.857, 0.850], -1.995, 0.154),
       ('dinov2b14', 'hue', (0.404, 0.004), (83.7, 68.6), [0.576, 0.424, 0.507], -0.411, 0.145),
       ('dinov2b14', 'saturation', (0.238, 0.006), (0.284, 0.233), [0.592, 0.802, 0.763], -3.160, 0.163),
       ('dinov2b14', 'value', (0.338, 0.005), (0.217, 0.181), [0.528, 0.623, 0.667], -8.240, 0.137)]
for bb, attr, (pm, ps), (en, er), ops, o6, o6w in t14:
    rec = ms_[bb]['attributes'][attr]
    check('D.14', f'{bb} {attr} power mean', pm, rec['power_mean'], 0.0015)
    check('D.14', f'{bb} {attr} power sd', ps, rec['power_sd'], 0.0015)
    check('D.14', f'{bb} {attr} err null', en, rec['probe_err_null_mean'], 0.06)
    check('D.14', f'{bb} {attr} err real', er, rec['probe_err_real_mean'], 0.06)
    for (_, key), val in zip([('O1', 'O1_procrustes'), ('O5', 'O5_mlp'), ('O8', 'O8_conv_residual')], ops):
        check('D.14', f'{bb} {attr} {key} transfer', val, rec['ops'][key]['attr_transfer_mean'], 0.0015)
    check('D.14', f'{bb} {attr} O6 transfer', o6, rec['ops']['O6_random_orthogonal']['attr_transfer_mean'], 0.0015)
    check('D.14', f'{bb} {attr} O6 win', o6w, rec['ops']['O6_random_orthogonal']['win_rate_mean'], 0.0015)

# ---------------------------------------------------------------- Table C7
cc = R('crossing_consumer_resnet50.json')['sites']
t7b = [('layer2', 18.08, 14.56, 0.588, 0.471, 7.45, 7.71, 3.3),
       ('layer3', 18.08, 14.56, 0.464, 0.374, 9.70, 9.12, 5.9),
       ('layer4', 18.08, 14.56, 0.099, 0.054, 16.28, 13.78, 15.4)]
for site, hn, en, htf, etf, hr, er, pct in t7b:
    h, e = cc[site]['hue_90.0'], cc[site]['heat_2.0']
    check('C7', f'{site} hue no-op', hn, h['consumer_response_total'], 0.006)
    check('C7', f'{site} heat no-op', en, e['consumer_response_total'], 0.006)
    check('C7', f'{site} hue T_F', htf, h['measured_O2_transfer'], 0.0015)
    check('C7', f'{site} heat T_F', etf, e['measured_O2_transfer'], 0.0015)
    rh = h['consumer_response_total'] * (1 - h['measured_O2_transfer'])
    re_ = e['consumer_response_total'] * (1 - e['measured_O2_transfer'])
    check('C7', f'{site} hue residual', hr, rh, 0.006)
    check('C7', f'{site} heat residual', er, re_, 0.006)
    check('C7', f'{site} relative difference %', pct,
          abs(rh - re_) / max(rh, re_) * 100, 0.15)

# ---------------------------------------------------------------- report
bad = [f for f in findings if f['verdict'] == 'MISMATCH']
uns = [f for f in findings if f['verdict'] == 'unlocatable']
json.dump({'findings': findings,
           'summary': {'checked': len(findings), 'ok': sum(1 for f in findings if f['verdict'] in ('ok', 'rounding')),
                       'mismatch': len(bad), 'unlocatable': len(uns)}},
          open(os.path.join(WORK, 'results', 'v15_number_check.json'), 'w'), indent=1)
for f in bad + uns:
    print(f"{f['verdict']:12s} {f['group']:8s} {f['cell']:44s} manuscript={f['manuscript']} raw={f['recomputed']} {f['note']}")
print(f"checked {len(findings)} cells | agree {len(findings)-len(bad)-len(uns)} | MISMATCH {len(bad)} | unlocatable {len(uns)}")
print('-> results/v15_number_check.json')
sys.exit(1 if (bad or uns) else 0)
