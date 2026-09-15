#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""261_real_site_closure.py — instantiate the closure criterion on measured features.

z = B c + r with c the harmonic carrier of the *known* input hue, B the synthesis estimated from the
features, r the reconstruction tail.  Reports, per site: the carrier second moment, the relative tail,
the closure defect of B against A_Delta = diag(1, R(D), R(2D)), and the held-out global linear fit
error, with the closure defect *engineered* by dropping the sin 2h coordinate (the pair rule).
Outputs results/real_site_closure_<site>.json.
"""
import json, os
import numpy as np

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {'yolo11n_y3': 'features/dense5_y3.npz', 'resnet18_l2': 'features/dense5_l2.npz'}
D = np.deg2rad(37.0)


def carrier(h):
    return np.stack([np.ones_like(h), np.cos(h), np.sin(h), np.cos(2 * h), np.sin(2 * h)])


def A_of(m, D):
    A = np.eye(m)
    for k in range(1, m // 2 + 1):
        R = np.array([[np.cos(k * D), -np.sin(k * D)], [np.sin(k * D), np.cos(k * D)]])
        A[2 * k - 1:2 * k + 1, 2 * k - 1:2 * k + 1] = R
    return A


def analysis(feats, hues, keep, delta_deg=37.0):
    """keep: indices of carrier coordinates retained. Returns the report dict."""
    m = 5
    C = carrier(np.deg2rad(hues))
    A = A_of(m, np.deg2rad(delta_deg))
    out = {}
    # (a) full carrier, pooled and per-shape
    tail, cd = [], []
    hold_err_single, hold_err_direct = [], []
    for sh in np.unique(feats['shape']):
        sel = feats['shape'] == sh
        Z = feats['feats'][sel]
        Cc = C[:, sel]
        B = Z.T @ np.linalg.pinv(Cc)                      # synthesis: carrier -> feature
        Rres = Z.T - B @ Cc
        tail.append(np.linalg.norm(Rres) / np.linalg.norm(Z.T))
        # shift the hue by delta and ask whether B reproduces the shifted features from A c
        idx = np.argsort(hues[sel])
        # build the shifted counterpart by re-indexing the hue grid
        hgrid = np.sort(hues[sel])
        shifted = np.stack([np.interp((hgrid + delta_deg) % 360, hgrid, Z[:, i]) for i in range(Z.shape[1])])
        # closure defect: can a single K map B c(h) to B A c(h)?
        Zc = B @ Cc
        Zcs = B @ (A @ Cc)
        K, *_ = np.linalg.lstsq(Zc.T, Zcs.T, rcond=None)
        cd.append(float(np.abs(K.T @ Zc - Zcs).max() / max(np.abs(Zcs).max(), 1e-9)))
        # held-out global fit: fit on half the hues, evaluate on the other half
        half = Cc.shape[1] // 2
        Kh, *_ = np.linalg.lstsq(Zc[:, :half].T, shifted[:, :half].T, rcond=None)
        hold_err_single.append(float(np.linalg.norm(Kh.T @ Zc[:, half:] - shifted[:, half:]) / np.linalg.norm(shifted[:, half:])))
    out['full_carrier'] = {'rel_tail_median': float(np.median(tail)),
                           'closure_defect_median': float(np.median(cd)),
                           'heldout_global_fit_rel_err_median': float(np.median(hold_err_single))}
    # (b) engineered non-closure: drop sin 2h (carrier index 4)
    droph = carrier(np.deg2rad(hues))
    Atr = A[:4, :4]
    tail2, cd2, ho2 = [], [], []
    for sh in np.unique(feats['shape']):
        sel = feats['shape'] == sh
        Z = feats['feats'][sel]
        Cc = droph[:4, sel]
        B = Z.T @ np.linalg.pinv(Cc)
        Rres = Z.T - B @ Cc
        tail2.append(np.linalg.norm(Rres) / np.linalg.norm(Z.T))
        Zc = B @ Cc; Zcs = B @ (Atr @ Cc)
        K, *_ = np.linalg.lstsq(Zc.T, Zcs.T, rcond=None)
        cd2.append(float(np.abs(K.T @ Zc - Zcs).max() / max(np.abs(Zcs).max(), 1e-9)))
        hgrid = np.sort(hues[sel])
        shifted = np.stack([np.interp((hgrid + delta_deg) % 360, hgrid, Z[:, i]) for i in range(Z.shape[1])])
        half = Cc.shape[1] // 2
        Kh, *_ = np.linalg.lstsq(Zc[:, :half].T, shifted[:, :half].T, rcond=None)
        ho2.append(float(np.linalg.norm(Kh.T @ Zc[:, half:] - shifted[:, half:]) / np.linalg.norm(shifted[:, half:])))
    out['dropped_sin2h'] = {'rel_tail_median': float(np.median(tail2)),
                            'closure_defect_median': float(np.median(cd2)),
                            'heldout_global_fit_rel_err_median': float(np.median(ho2))}
    out['carrier_second_moment'] = np.round((carrier(np.deg2rad(hues)) @ carrier(np.deg2rad(hues)).T) /
                                            carrier(np.deg2rad(hues)).shape[1], 3).tolist()
    return out


def main():
    res = {}
    for site, path in FILES.items():
        d = np.load(os.path.join(WORK, path), allow_pickle=True)
        hues = d['hue'] if 'hue' in d else d['h']
        feats = {'feats': d['feats'], 'shape': d['shape']}
        res[site] = analysis(feats, hues, keep=None)
        print(f"{site}: full carrier  tail {res[site]['full_carrier']['rel_tail_median']:.3f} "
              f"defect {res[site]['full_carrier']['closure_defect_median']:.2e} "
              f"heldout {res[site]['full_carrier']['heldout_global_fit_rel_err_median']:.3f} | "
              f"drop sin2h  tail {res[site]['dropped_sin2h']['rel_tail_median']:.3f} "
              f"defect {res[site]['dropped_sin2h']['closure_defect_median']:.3f} "
              f"heldout {res[site]['dropped_sin2h']['heldout_global_fit_rel_err_median']:.3f}")
    json.dump(res, open(os.path.join(WORK, 'results', 'real_site_closure.json'), 'w'), indent=1)
    print('-> results/real_site_closure.json')


if __name__ == '__main__':
    main()
