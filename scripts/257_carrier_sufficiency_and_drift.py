#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""257_carrier_sufficiency_and_drift.py — (2) exact carrier sufficiency, (4) the price of structure,
and (1) the rule-drift law for continuous deformations.

PART 2.  Carrier sufficiency with exact constants (replaces the vague tail/misalignment premises).
  Carrier c in R^m with action A_D (block rotations); site features z = B c + w, ||w|| <= eps.
  Read-in E, fixed block action A_D applied in the carrier, decode by B.
    (i)  the fixed-interface error is bounded by  ||B A||(||E B - I|| ||c|| + ||E|| eps) + eps;
    (ii) the best *any* linear K can do is the invariance defect of (B, A_D) of the closure
         criterion, and it is zero exactly when A_D ker B subset ker B;
    (iii) the *price of structure* is the gap between the fixed block-diagonal interface and (ii).
  Misalignment: rotate the interface planes by Theta and measure the exponent in Theta of both the
  fixed-interface error and the invariance defect.

PART 4.  The capacity statement in its honest dual form: given an orbit, the reachable set of a
  family is what matters.  We compare, on a closed orbit, (a) a coordinate-wise monotone family,
  (b) a fixed-basis linear family, (c) the block-rotation carrier family — and report, for each, the
  best achievable L2 error and whether an exact realisation exists.  This replaces the unproved
  sigma_3^2 aggregate with a measured, family-by-family comparison on a known orbit.

PART 1.  Rule drift along a continuous deformation.  For a piecewise-affine site and a one-parameter
  family tau_t of input transformations, the pooled optimal operator K*(t) is computed at each t
  together with the region-transition defect D(t).  We test whether the transfer score of a single
  operator fitted at t0 (the quantity T_F) is predicted by the *drift* of K* rather than by the
  crossing share alone.

Outputs results/carrier_sufficiency.json, results/family_capacity.json, results/rule_drift.json.
"""
import json
import os

import numpy as np

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(3)


def rot(k, th):
    return np.array([[np.cos(k * th), -np.sin(k * th)], [np.sin(k * th), np.cos(k * th)]])


def block_action(m, th, planes=None):
    """A_D on R^m: rotations by k*th on the k-th pair; planes may be a list of 2x2 orthonormal bases."""
    A = np.zeros((m, m))
    for k in range(1, m // 2 + 1):
        if planes is None:
            A[2 * (k - 1):2 * k, 2 * (k - 1):2 * k] = rot(k, th)
        else:
            Q = planes[k - 1]
            A[2 * (k - 1):2 * k, 2 * (k - 1):2 * k] = Q @ rot(k, th) @ Q.T
    return A


# ===================================================================================== PART 2
def part2():
    """Carrier sufficiency in two regimes, both exact.

    Regime (a) sufficient feature dimension: B injective  => the closure criterion holds for every
      carrier action, so the achievable error is zero and the only error is the one you choose:
      an interface action A_Theta differing from the truth A costs ||B (A_Theta - A) c||.
    Regime (b) insufficient dimension: ker B != 0 => a fixed linear K exists iff A ker B subset ker B,
      and when it fails the best linear K has error exactly the invariance defect.
    """
    m, N = 4, 4000
    th = np.deg2rad(37.0)
    h = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c = np.stack([np.cos(h), np.sin(h), np.cos(2 * h), np.sin(2 * h)])
    A = block_action(m, th)
    out = {'regime_a_injective_synthesis': {}, 'regime_b_kernel': {}, 'misalignment_sweep': []}

    # ---- regime (a): orthonormal synthesis d = m (injective)
    d = m
    B = np.linalg.qr(rng.normal(size=(d, m)))[0]
    z, zs_true = B @ c, B @ (A @ c)
    out['regime_a_injective_synthesis']['best_linear_error'] = float(
        np.abs((np.linalg.lstsq(z.T, zs_true.T, rcond=None)[0].T @ z - zs_true)).max())

    # misalignment, done properly: the interface plane is TILTED OUT of the true plane by Theta.
    # (a rotation within the same plane changes nothing: 2x2 rotations commute).  Carrier is 6-d
    # (k = 1,2,3); the k=1 interface plane tilts toward the k=3 pair's first axis.
    m6 = 6
    h6 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c6 = np.stack(sum([[np.cos(k * h6), np.sin(k * h6)] for k in (1, 2, 3)], []))
    A6 = np.zeros((m6, m6))
    for k in (1, 2, 3):
        A6[2 * (k - 1):2 * k, 2 * (k - 1):2 * k] = rot(k, th)

    def tilted_action6(T):
        Aint = np.eye(m6)
        for k in (1, 2, 3):
            f1 = np.zeros(m6); f1[2 * (k - 1)] = 1.0
            f2 = np.zeros(m6); f2[2 * (k - 1) + 1] = 1.0
            if k == 1:
                f2 = np.cos(T) * f2 + np.sin(T) * np.eye(m6)[4]     # tilt into the k=3 pair
                f2 = f2 / np.linalg.norm(f2)
            P = np.outer(f1, f1) + np.outer(f2, f2)
            L = np.outer(f2, f1) - np.outer(f1, f2)
            Rot = np.eye(m6) - P + np.cos(k * th) * P + np.sin(k * th) * L
            Aint = Rot @ Aint
        return Aint

    B6 = np.linalg.qr(rng.normal(size=(m6, m6)))[0]
    for deg in (0.5, 1, 2, 5, 10, 15, 20, 30):
        T = np.deg2rad(deg)
        Am = tilted_action6(T)
        err = float(np.abs(B6 @ ((Am - A6) @ c6)).max())
        out['misalignment_sweep'].append({'theta_deg': deg, 'interface_error': err})
    thetas = np.deg2rad([q['theta_deg'] for q in out['misalignment_sweep']])
    errs = [q['interface_error'] for q in out['misalignment_sweep']]
    out['misalignment_exponent'] = float(np.polyfit(np.log(thetas), np.log(errs), 1)[0])
    out['misalignment_error_at_0p5deg'] = errs[0]

    # ---- regime (b): synthesis with a kernel (drop sin 2h)
    Bk = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0]])       # 3 x 4, ker = span(e4)
    zk, zsk = Bk @ c, Bk @ (A @ c)
    Kk, *_ = np.linalg.lstsq(zk.T, zsk.T, rcond=None)
    u, sv, vh = np.linalg.svd(Bk)
    rank = int((sv > 1e-10).sum())
    null = vh[rank:].T
    out['regime_b_kernel'] = {
        'best_linear_error': float(np.abs(Kk.T @ zk - zsk).max()),
        'invariance_defect': float(np.abs(Bk @ (A @ null)).max()),
        'criterion_says_closed': False,
        'note': 'the dropped coordinate (sin 2h) is carried by A into the retained one',
    }
    # a kernel that IS A-invariant: drop both k=2 coordinates -> the retained block is invariant
    Bk2 = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
    A2 = np.zeros((4, 4)); A2[:2, :2] = rot(1, th); A2[2:, 2:] = rot(2, th)
    u2, sv2, vh2 = np.linalg.svd(Bk2)
    rank2 = int((sv2 > 1e-10).sum())
    null2 = vh2[rank2:].T
    z2, z2s = Bk2 @ c, Bk2 @ (A2 @ c)
    K2, *_ = np.linalg.lstsq(z2.T, z2s.T, rcond=None)
    out['regime_b_kernel']['invariant_kernel_case'] = {
        'best_linear_error': float(np.abs(K2.T @ z2 - z2s).max()),
        'invariance_defect': float(np.abs(Bk2 @ (A2 @ null2)).max()),
        'criterion_says_closed': True,
        'note': 'dropping a whole conjugate pair leaves an A-invariant kernel, and closure holds',
    }
    json.dump(out, open(os.path.join(WORK, 'results', 'carrier_sufficiency.json'), 'w'), indent=1)
    print('PART 2  carrier sufficiency')
    print(f"  (a) injective synthesis: best linear error {out['regime_a_injective_synthesis']['best_linear_error']:.1e}"
          f"; misalignment exponent in Theta = {out['misalignment_exponent']:.2f}")
    rb = out['regime_b_kernel']
    print(f"  (b) kernel (drop sin2h): best linear {rb['best_linear_error']:.4f}, invariance defect "
          f"{rb['invariance_defect']:.4f}  -> not closable")
    ik = rb['invariant_kernel_case']
    print(f"      invariant kernel (drop both k=2): best linear {ik['best_linear_error']:.1e}, "
          f"defect {ik['invariance_defect']:.1e}  -> closable")
    return out


# ===================================================================================== PART 4
def part4():
    """Family capacity on a known orbit, every family acting on the SAME feature space.

    Feature space: the 3-d code z(h) = (cos h, sin h, cos 2h).  Families:
      (a) coordinate-wise monotone maps of the family parameter,
      (b) best fixed 3x3 linear map,
      (c) the carrier-derived 3x3 map (lift to the 4-d carrier, rotate, project back),
      (d) the full 4-d carrier code (a different feature space, for contrast).
    """
    N = 2001
    h = np.linspace(0, 2 * np.pi, N, endpoint=False)
    Z = np.stack([np.cos(h), np.sin(h), np.cos(2 * h)])
    D = np.deg2rad(53.0)
    Zs = np.stack([np.cos(h + D), np.sin(h + D), np.cos(2 * (h + D))])
    out = {}

    # (a) coordinate-wise monotone maps of the parameter (PAVA per coordinate)
    def pava(y):
        n = len(y); fit = np.zeros(n); i = 0; idx = []
        while i < n:
            j = i
            while j + 1 < n and y[j + 1] < y[j]:
                j += 1
            fit[i:j + 1] = y[i:j + 1].mean()
            i = j + 1
        return fit
    mono = sum(float(((pava(Zs[i]) - Zs[i]) ** 2).sum()) for i in range(3))
    out['coordinatewise_monotone'] = {'l2_error': mono, 'exact': mono < 1e-12,
                                      'family': 'per-coordinate monotone functions of the parameter'}

    # (b) best fixed 3x3
    K, *_ = np.linalg.lstsq(Z.T, Zs.T, rcond=None)
    rb = float(((K.T @ Z - Zs) ** 2).sum())
    out['fixed_basis_linear_3d'] = {'l2_error': rb, 'exact': rb < 1e-9,
                                    'family': 'best fixed linear map on the 3-d code'}

    # (c) carrier-derived 3x3: lift with the right inverse of P, rotate in the carrier, project back
    P = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0]])       # 3 x 4
    A = np.zeros((4, 4)); A[:2, :2] = rot(1, D); A[2:, 2:] = rot(2, D)
    Pinv = np.linalg.pinv(P)                                          # 4 x 3 (right inverse)
    Kc = P @ A @ Pinv
    rc = float(((Kc @ Z - Zs) ** 2).sum())
    out['carrier_derived_3d'] = {'l2_error': rc, 'exact': rc < 1e-9,
                                 'operator': Kc.tolist(),
                                 'family': 'act in the 4-d carrier, project back to the 3-d code'}

    # (d) full 4-d carrier
    C = np.stack([np.cos(h), np.sin(h), np.cos(2 * h), np.sin(2 * h)])
    Cs = np.stack([np.cos(h + D), np.sin(h + D), np.cos(2 * (h + D)), np.sin(2 * (h + D))])
    rd = float(((A @ C - Cs) ** 2).sum())
    out['full_carrier_4d'] = {'l2_error': rd, 'exact': rd < 1e-9,
                              'family': 'the 4-d carrier code with the block action'}

    # (e) the proposition: for an orthogonal carrier, the least-squares optimal linear map on any
    #     coordinate-selected code is the carrier action projected onto the retained coordinates,
    #     and its residual is zero exactly when the selection is A-invariant.
    rng2 = np.random.default_rng(11)
    checks, gaps, inv_agree = [], [], True
    for _ in range(40):
        m2 = 6                                            # carrier: k = 1, 2, 3 pairs
        h2 = np.linspace(0, 2 * np.pi, 1500, endpoint=False)
        C2 = np.stack(sum([[np.cos(k * h2), np.sin(k * h2)] for k in (1, 2, 3)], []))
        A2b = np.zeros((m2, m2))
        for k in (1, 2, 3):
            A2b[2 * (k - 1):2 * k, 2 * (k - 1):2 * k] = rot(k, D)
        keep = sorted(rng2.choice(m2, size=rng2.integers(1, m2), replace=False))
        P2 = np.eye(m2)[keep]
        Z2, Z2s = P2 @ C2, P2 @ (A2b @ C2)
        Kstar, *_ = np.linalg.lstsq(Z2.T, Z2s.T, rcond=None)
        gaps.append(float(np.abs(Kstar.T - P2 @ A2b @ P2.T).max()))
        # invariance of the dropped set  <=>  residual zero
        dropped = [i for i in range(m2) if i not in keep]
        u2, sv2, vh2 = np.linalg.svd(P2)
        null2 = vh2[len(keep):].T
        defect = float(np.abs(P2 @ (A2b @ null2)).max()) if null2.size else 0.0
        resid = float(np.abs(Kstar.T @ Z2 - Z2s).max())
        inv_agree = inv_agree and ((defect < 1e-9) == (resid < 1e-9))
    out['optimal_linear_is_projected_carrier_action'] = {
        'n_random_selections': 40,
        'max_gap_to_PAP^T': max(gaps),
        'criterion_agrees_with_residual': bool(inv_agree),
        'statement': 'K* = P A P^T whenever the carrier coordinates are orthogonal with equal norms; '
                     'the residual vanishes iff the retained coordinate set is A-invariant',
    }

    json.dump(out, open(os.path.join(WORK, 'results', 'family_capacity.json'), 'w'), indent=1)
    print('PART 4  family capacity on the orbit (cos h, sin h, cos 2h), shift 53 deg')
    for k, v in out.items():
        if 'l2_error' in v:
            print(f"  {k:26s} L2 {v['l2_error']:.4e}  {'exact' if v['exact'] else 'not exact'}")
    pr = out['optimal_linear_is_projected_carrier_action']
    print(f"  optimal operator = P A P^T over 40 random coordinate selections: max gap "
          f"{pr['max_gap_to_PAP^T']:.2e}; criterion agrees with the residual: {pr['criterion_agrees_with_residual']}")
    return out


# ===================================================================================== PART 1
def part1():
    """Rule drift: fit K*(t) and the transition defect along a sweep of the transformation."""
    D, n, d = 3, 10, 5
    W1 = rng.normal(size=(n, D)); W2 = rng.normal(size=(d, n))
    X = rng.normal(size=(4000, D))

    def site(X, R):
        A = X @ W1.T
        G = A > 0
        return np.maximum(A, 0) @ W2.T, G

    def T(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1.0]])

    ts = np.deg2rad(np.arange(10, 101, 10))
    ops, defects, transfers = [], [], []
    U0, _ = site(X, None)
    for t in ts:
        Tt = T(t)
        U = U0
        V, _ = site(X @ Tt.T, None)
        K, *_ = np.linalg.lstsq(U.T, V.T, rcond=None)
        ops.append(K.T)
        # transfer: use the operator fitted at t=0 (the t0 operator) to predict this t
        K0 = ops[0]
        err = ((K0 @ U - V) ** 2).mean()
        denom = ((U - V) ** 2).mean()
        transfers.append(1.0 - np.sqrt(err / denom) if denom > 0 else np.nan)
        defects.append(float(((K.T @ U - V) ** 2).mean()))
    drift = [float(np.linalg.norm(ops[i] - ops[0], 'fro')) for i in range(len(ts))]
    # correlation of transfer with drift, and with the defect at that t
    def corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.corrcoef(a, b)[0, 1])
    out = {'theta_deg': np.rad2deg(ts).tolist(),
           'drift_from_t0': drift, 'defect': defects, 'transfer_of_t0_operator': transfers,
           'corr_transfer_vs_drift': corr(transfers, drift),
           'corr_transfer_vs_defect': corr(transfers, defects)}
    json.dump(out, open(os.path.join(WORK, 'results', 'rule_drift.json'), 'w'), indent=1)
    print('PART 1  rule drift along a hue-like rotation of a synthetic ReLU site')
    print(f"  corr(transfer of the t0 operator, drift of K*)   = {out['corr_transfer_vs_drift']:+.3f}")
    print(f"  corr(transfer of the t0 operator, transition defect) = {out['corr_transfer_vs_defect']:+.3f}")
    return out


if __name__ == '__main__':
    part2()
    part4()
    part1()
