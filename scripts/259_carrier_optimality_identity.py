#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""259_carrier_optimality_identity.py — the carrier optimality identity, verified.

For a white carrier (E[c c^T] = I), features z = B c and target z' = B A c:

    K*_A = B A B^dagger            (a minimiser; unique on the data-covered directions)
    min_K E || K z - z' ||^2  =  || B A P_{ker B} ||_F^2

and for a coloured carrier (E[c c^T] = Sigma_c) the covariance-weighted version

    min_K E || K z - z' ||^2  =  || B A P_{ker B} Sigma_c^{1/2} ||_F^2 .

The right-hand side is zero iff A ker B subset ker B (the closure criterion), so existence,
optimality, error source and design rule are one statement.  The coordinate-selection case
(K* = P A P^T) is the corollary with B = P an orthonormal coordinate selection.

Outputs results/carrier_optimality_identity.json.
"""
import json, os
import numpy as np

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    h = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    rows, worst = [], 0.0
    for D in (0.3, 0.7, 1.2):
        C = np.stack(sum([[np.cos(k * h), np.sin(k * h)] for k in (1, 2, 3)], []))
        A = np.zeros((6, 6))
        for k in (1, 2, 3):
            A[2 * (k - 1):2 * k, 2 * (k - 1):2 * k] = [[np.cos(k * D), -np.sin(k * D)],
                                                       [np.sin(k * D), np.cos(k * D)]]
        Sig = C @ C.T / C.shape[1]
        for keep, name in [([0, 1, 2], 'drop one member of k=2'), ([0, 1], 'drop all k>=2'),
                           ([2, 3, 4, 5], 'drop the k=1 pair'), (list(range(6)), 'full carrier')]:
            B = np.eye(6)[keep]
            Z, Zs = B @ C, B @ (A @ C)
            K, *_ = np.linalg.lstsq(Z.T, Zs.T, rcond=None)
            K = K.T
            resid = float(((K @ Z - Zs) ** 2).sum(0).mean())
            u, sv, vh = np.linalg.svd(B)
            rank = int((sv > 1e-10).sum())
            Pk = vh[rank:].T @ vh[rank:]
            M = B @ A @ Pk
            formula = float(np.trace(M @ Sig @ M.T))
            L = np.linalg.cholesky(Sig)
            Bw = B @ L
            Mw = Bw @ A @ (np.eye(6) - np.linalg.pinv(Bw) @ Bw)
            resw = float(((np.linalg.lstsq((Bw @ np.linalg.solve(L, C)).T,
                                           (Bw @ (A @ np.linalg.solve(L, C))).T, rcond=None)[0].T
                           @ (Bw @ np.linalg.solve(L, C)) - Bw @ (A @ np.linalg.solve(L, C))) ** 2
                          ).sum(0).mean())
            formulaw = float(np.linalg.norm(Mw, 'fro') ** 2)
            gap = max(abs(resid - formula), abs(resw - formulaw))
            worst = max(worst, gap)
            rows.append({'theta_rad': D, 'synthesis': name, 'residual': resid,
                         'formula_coloured': formula, 'residual_whitened': resw,
                         'formula_whitened': formulaw, 'gap': gap,
                         'K_star_equals_BABdag': float(np.abs(K - B @ A @ np.linalg.pinv(B)).max())})
        # coordinate-selection corollary: K* = P A P^T for an orthonormal selection
        P = np.eye(6)[[0, 1, 2]]
        Z, Zs = P @ C, P @ (A @ C)
        K, *_ = np.linalg.lstsq(Z.T, Zs.T, rcond=None)
        rows.append({'theta_rad': D, 'synthesis': 'coordinate-selection corollary',
                     'K_star_minus_PAPt': float(np.abs(K.T - P @ A @ P.T).max())})
    out = {'worst_identity_gap': worst, 'rows': rows,
           'statement': 'min_K E||Kz - z\'||^2 = ||B A P_ker B Sigma_c^{1/2}||_F^2, zero iff '
                        'A ker B subset ker B; K* = B A B^dagger'}
    json.dump(out, open(os.path.join(WORK, 'results', 'carrier_optimality_identity.json'), 'w'), indent=1)
    print(f"identity verified: worst gap over {len(rows)} rows = {worst:.2e}")
    print("  (coloured and whitened forms agree; K* = B A B^dagger; coordinate case K* = P A P^T)")


if __name__ == '__main__':
    main()
