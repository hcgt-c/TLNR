#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""265_interface_action_comparison.py — is the *measured, fixed* action worth its rigidity?

The paper's interface is c = E_theta(z) with the action fixed by measurement,
A_delta = diag(I_2, R(delta), R(2 delta)), never fitted.  §7.2 states that a fitted action of equal
supervision and budget has not been compared.  This script runs that comparison.

Arms (all share the identical read-in E_theta and the identical code targets; the action is what differs):

  fixed  : A_delta = diag(I_2, R(delta), R(2 delta))                     — 0 action parameters (ours)
  gen    : M_delta = exp(delta * G), G in R^{6x6} free                   — 36 action parameters
  lin    : M_delta = I + delta * B, B in R^{6x6} free                    — 36 action parameters
  mlp    : M_delta(c) = c + MLP([c, cos delta, sin delta]), 8-256-256-6  — 69,638 action parameters

Every arm acts on the six-dimensional code and can therefore be iterated, which is what makes the
composition comparison well posed.  All arms see the same code supervision; the three learnable arms
additionally see paired increments drawn from the *same* labels (hue differences) that fix the
structure of the fixed arm, so the fixed arm gets no data advantage.

Protocol (identical to script 79, the source of §7.4's read-out numbers):
  train shapes 0-3, test shapes 4-5 (zero-shot); 1500 epochs x 24 minibatches x 96 code samples;
  AdamW lr 2e-3, weight decay 1e-4; feature caches in features/*.npz.
  Training increments D_tr = {20, 40, 60, 80} degrees; evaluation increments 37 and 90 degrees
  (never fitted), composition (37, 53) against the direct 90 degrees.

Shapes and cost, before writing code: read-in 64-256-256-6 (83,974 parameters); code batch (96, 6);
pair batch (96, 6); per epoch 24 x (96 + 96) = 4,608 forward/backward samples through a network of
~0.17 MFLOP/sample -> ~0.8 MFLOP/epoch per arm, i.e. negligible; 4 arms x 3 seeds x 1500 epochs is
about 15-30 min on the RTX 2060 (6 GB) and never more than a few hundred MB of memory.

Outputs: results/interface_action_comparison.json (per-seed + aggregate), printed table.
Usage:
  python scripts/265_interface_action_comparison.py [--arms fixed,gen,lin,mlp] [--seeds 0,1,2]
                                                    [--epochs 1500] [--quick]
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_SH = [0, 1, 2, 3]
TEST_SH = [4, 5]
H5 = np.arange(0, 360, 5)
DTR = [20, 40, 60, 80]
DTE = [37, 90]
DCOMP = (37, 53)


def circ(a, b):
    return min(abs(a - b), 360 - abs(a - b))


def pool():
    """Identical to script 79's pool(): 6 shapes x hue x (s, v) cells, y3 GAP features."""
    p = []
    d = np.load(os.path.join(WORK, 'features/dense5_y3.npz'))
    F = d['feats'].reshape(6, 72, 4, 64).mean(2)
    for si in range(6):
        for hi, h in enumerate(H5):
            p.append((si, h, 1.0, 1.0, F[si, hi]))
    d = np.load(os.path.join(WORK, 'features/sheet_hv_y3.npz'))
    F = d['feats'].reshape(10, 6, 72, 64)
    v10 = np.round(np.arange(0.1, 1.0001, 0.1), 2)
    for vi in range(10):
        for si in range(6):
            for hi, h in enumerate(H5):
                p.append((si, h, 1.0, float(v10[vi]), F[vi, si, hi]))
    d = np.load(os.path.join(WORK, 'features/dense3d_rings_y3.npz'))
    F = d['feats'].reshape(6, 5, 36, 4, 64).mean(3)
    SV = [(0.3, 1.0), (0.6, 1.0), (1.0, 0.6), (0.6, 0.6), (0.35, 0.35)]
    for si in range(6):
        for svi, (s, v) in enumerate(SV):
            for k, h in enumerate(range(0, 360, 10)):
                p.append((si, h, s, v, F[si, svi, k]))
    d = np.load(os.path.join(WORK, 'features/dense3d_lattice_y3.npz'))
    F = d['feats'].reshape(6, 12, 4, 4, 4, 64).mean(4)
    SV2 = [0.25, 0.5, 0.75, 1.0]
    for si in range(6):
        for i, h in enumerate(range(0, 360, 30)):
            for a, s in enumerate(SV2):
                for b, v in enumerate(SV2):
                    p.append((si, h, s, v, F[si, i, a, b]))
    d = np.load(os.path.join(WORK, 'features/sheet_sv_y3.npz'))
    F = d['feats'].reshape(3, 6, 10, 10, 64)
    sv10 = np.round(np.arange(0.1, 1.0001, 0.1), 2)
    for hi, h in enumerate([0, 120, 240]):
        for si in range(6):
            for a, s in enumerate(sv10):
                for b, v in enumerate(sv10):
                    p.append((si, h, float(s), float(v), F[hi, si, a, b]))
    return p


# ---------------------------------------------------------------- actions
def rot(d):
    c, s = math.cos(d), math.sin(d)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=DEV)


def fixed_action(c, drad):
    """A_delta = diag(I_2, R(delta), R(2 delta)); drad in radians (tensor of shape (B,))."""
    B = c.shape[0]
    out = c.clone()
    for i in range(B):
        d = drad[i]
        r1 = rot(d)
        r2 = rot(2 * d)
        out[i, 2:4] = c[i, 2:4] @ r1.T
        out[i, 4:6] = c[i, 4:6] @ r2.T
    return out


class GenAction(nn.Module):
    """M_delta = exp(delta * G), G free (36 parameters)."""

    def __init__(self):
        super().__init__()
        self.G = nn.Parameter(torch.zeros(6, 6))

    def forward(self, c, drad):
        M = torch.matrix_exp(drad[:, None, None] * self.G)
        return torch.einsum('bi,bij->bj', c, M)

    @property
    def n_action_params(self):
        return self.G.numel()


class LinAction(nn.Module):
    """M_delta = I + delta * B, B free (36 parameters)."""

    def __init__(self):
        super().__init__()
        self.B = nn.Parameter(torch.zeros(6, 6))

    def forward(self, c, drad):
        return c + drad[:, None] * (c @ self.B.T)

    @property
    def n_action_params(self):
        return self.B.numel()


class MLPAction(nn.Module):
    """M_delta(c) = c + MLP([c, cos delta, sin delta]) with a 256-256 hidden MLP (69,638 parameters)."""

    def __init__(self, dh=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, dh), nn.SiLU(), nn.Linear(dh, dh), nn.SiLU(), nn.Linear(dh, 6))

    def forward(self, c, drad):
        x = torch.cat([c, torch.cos(drad)[:, None], torch.sin(drad)[:, None]], 1)
        return c + self.net(x)

    @property
    def n_action_params(self):
        return sum(p.numel() for p in self.parameters())


class Head6(nn.Module):
    """The read-in of §7.4/Box E2: 64-256-256-6."""

    def __init__(self, dh=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, dh), nn.SiLU(), nn.Linear(dh, dh), nn.SiLU(), nn.Linear(dh, 6))

    def forward(self, x):
        o = self.net(x)
        l = F.softplus(o[:, 0:1])
        s = torch.sigmoid(o[:, 1:2])
        return torch.cat([l, s, o[:, 2:6]], 1)


ACTIONS = {'fixed': lambda: None, 'gen': GenAction, 'lin': LinAction, 'mlp': MLPAction}


def apply_action(name, action, c, drad):
    if name == 'fixed':
        return fixed_action(c, drad)
    return action(c, drad)


# ---------------------------------------------------------------- data plumbing
def build_cells(pool_items):
    cell = defaultdict(dict)
    for si, h, s, v, f in pool_items:
        cell[(si, round(s, 2), round(v, 2))][int(h)] = f
    return {k: v for k, v in cell.items() if len(v) >= 2}


def mstar_table(cells):
    mraw = {}
    for key, hm in cells.items():
        Z = np.stack(list(hm.values()))
        Zc = Z - Z.mean(0, keepdims=True)
        mraw[(key[0], key[1], key[2])] = float(np.sqrt(np.mean(Zc ** 2)))
    ref = max(mraw.values())
    return {k: v / ref for k, v in mraw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='fixed,gen,lin,mlp')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--minibatches', type=int, default=24)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    arms = [x for x in a.arms.split(',') if x]
    seeds = [int(x) for x in a.seeds.split(',') if x]
    if a.quick:
        a.epochs, seeds = 60, [0]

    print(f"device {DEV}", flush=True)
    POOL = pool()
    tr_items = [p for p in POOL if p[0] in TRAIN_SH]
    feat_std = float(np.std([p[4] for p in tr_items]))
    tr_cells = build_cells(tr_items)
    MST = mstar_table(tr_cells)
    test_cells = build_cells([p for p in POOL if p[0] in TEST_SH])
    # index by (s, v) for the script-79-compatible m* lookup
    msv = {(round(k[1], 2), round(k[2], 2)): v for k, v in MST.items()}

    def m_star(s, v):
        k = np.array([round(s, 2), round(v, 2)])
        return msv[tuple(k)]

    keys = list(tr_cells.keys())
    pairs = []                                    # (cell, h, delta) with h+delta present in the same cell
    for k, hm in tr_cells.items():
        hs = sorted(hm.keys())
        for h in hs:
            for d in DTR:
                if (h + d) % 360 in hm:
                    pairs.append((k, h, d))
    print(f"train cells {len(tr_cells)}, pair anchors {len(pairs)}, test cells {len(test_cells)}", flush=True)

    results = {}
    t_all = time.time()
    for arm in arms:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            E = Head6().to(DEV)
            act = ACTIONS[arm]().to(DEV) if arm != 'fixed' else None
            params = [{'name': 'read_in', 'n': sum(p.numel() for p in E.parameters())}]
            if act is not None:
                params.append({'name': 'action', 'n': act.n_action_params})
            opt = torch.optim.AdamW(list(E.parameters()) + (list(act.parameters()) if act is not None else []),
                                    lr=2e-3, weight_decay=1e-4)

            def code_batch(bs):
                Xs, Ys = [], []
                for _ in range(bs):
                    k = random.choice(keys)
                    hm = tr_cells[k]
                    h = random.choice(sorted(hm.keys()))
                    m = m_star(k[1], k[2])
                    th = math.radians(h)
                    Xs.append(hm[h] / feat_std)
                    Ys.append([k[2], k[1], m * math.cos(th), m * math.sin(th), m * math.cos(2 * th), m * math.sin(2 * th)])
                return np.array(Xs), np.array(Ys)

            def pair_batch(bs):
                Xs, Ys, Ds = [], [], []
                for _ in range(bs):
                    k, h, d = random.choice(pairs)
                    hm = tr_cells[k]
                    Xs.append(hm[h] / feat_std)
                    Ys.append(hm[(h + d) % 360] / feat_std)
                    Ds.append(math.radians(d))
                return np.array(Xs), np.array(Ys), np.array(Ds, dtype=np.float32)

            for ep in range(a.epochs):
                for _ in range(a.minibatches):
                    X, Y = code_batch(a.bs)
                    x = torch.tensor(X, dtype=torch.float32, device=DEV)
                    y = torch.tensor(Y, dtype=torch.float32, device=DEV)
                    pred = E(x)
                    loss = torch.mean((pred[:, 0] - y[:, 0]) ** 2) + torch.mean((pred[:, 1] - y[:, 1]) ** 2)
                    loss = loss + torch.mean((pred[:, 2:] - y[:, 2:]) ** 2)
                    if act is not None:
                        Xp, Yp, Dp = pair_batch(a.bs)
                        xp = torch.tensor(Xp, dtype=torch.float32, device=DEV)
                        yp = torch.tensor(Yp, dtype=torch.float32, device=DEV)
                        dd = torch.tensor(Dp, dtype=torch.float32, device=DEV)
                        cp = E(xp)
                        with torch.no_grad():
                            cq = E(yp)
                        moved = apply_action(arm, act, cp, dd)
                        loss = loss + torch.mean((moved - cq) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if a.quick and ep % 20 == 0:
                    print(f"  {arm} s{seed} ep{ep} loss {loss.item():.5f}", flush=True)

            # ---------------- evaluation on unseen shapes ----------------
            E.eval()
            if act is not None:
                act.eval()
            e0, e37, e90, comp_defect, comp_truth = [], [], [], [], []
            with torch.no_grad():
                for k, hm in sorted(test_cells.items()):
                    for h, f in hm.items():
                        ft = torch.tensor(f / feat_std, dtype=torch.float32, device=DEV)[None]
                        c = E(ft)

                        def decode(cc):
                            return float(np.degrees(np.arctan2(cc[0, 3].item(), cc[0, 2].item())) % 360)

                        h0 = decode(c)
                        e0.append(circ(h0, h))
                        for d, store in ((DTE[0], e37), (DTE[1], e90)):
                            dr = torch.tensor([math.radians(d)], dtype=torch.float32, device=DEV)
                            cd = apply_action(arm, act, c, dr)
                            store.append(circ((decode(cd) - h0) % 360, d))       # increment accuracy
                        d1 = torch.tensor([math.radians(DCOMP[0])], dtype=torch.float32, device=DEV)
                        d2 = torch.tensor([math.radians(DCOMP[1])], dtype=torch.float32, device=DEV)
                        dt = torch.tensor([math.radians(DCOMP[0] + DCOMP[1])], dtype=torch.float32, device=DEV)
                        c1 = apply_action(arm, act, c, d1)
                        c12 = apply_action(arm, act, c1, d2)
                        cdir = apply_action(arm, act, c, dt)
                        comp_defect.append(circ(decode(c12), decode(cdir)))
                        comp_truth.append(circ((decode(c12) - h0) % 360, DCOMP[0] + DCOMP[1]))
            rec = {
                'arm': arm, 'seed': seed,
                'epochs': a.epochs, 'params': params,
                'n_params_total': sum(p['n'] for p in params),
                'e0_median': float(np.median(e0)), 'e0_mean': float(np.mean(e0)),
                'inc37_median': float(np.median(e37)), 'inc37_mean': float(np.mean(e37)),
                'inc90_median': float(np.median(e90)), 'inc90_mean': float(np.mean(e90)),
                'comp_defect_median': float(np.median(comp_defect)), 'comp_defect_mean': float(np.mean(comp_defect)),
                'comp_truth_median': float(np.median(comp_truth)), 'comp_truth_mean': float(np.mean(comp_truth)),
                'n_test_cells': len(test_cells),
            }
            results.setdefault(arm, []).append(rec)
            print(f"{arm} seed{seed} | read {rec['e0_median']:.2f}° | inc37 {rec['inc37_median']:.2f}° "
                  f"| inc90 {rec['inc90_median']:.2f}° | comp defect {rec['comp_defect_median']:.2f}° "
                  f"| comp vs truth {rec['comp_truth_median']:.2f}° | params {rec['n_params_total']}", flush=True)

    agg = {}
    for arm, recs in results.items():
        agg[arm] = {k: float(np.mean([r[k] for r in recs])) for k in
                    ('e0_median', 'inc37_median', 'inc90_median', 'comp_defect_median', 'comp_truth_median')}
        agg[arm]['n_params_total'] = recs[0]['n_params_total']
        agg[arm]['n_seeds'] = len(recs)
    out = {'protocol': 'train shapes 0-3, test shapes 4-5; identical read-in and code supervision; '
                       f'D_tr={DTR}, D_eval={DTE}, composition {DCOMP}; {a.epochs} epochs x {a.minibatches} minibatches x {a.bs}',
           'device': DEV, 'feat_std': feat_std, 'per_seed': results, 'aggregate': agg,
           'wall_seconds': round(time.time() - t_all, 1)}
    os.makedirs(os.path.join(WORK, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(WORK, 'results', 'interface_action_comparison.json'), 'w'), indent=1)
    print('\nAE | arm | read (deg) | inc37 | inc90 | comp defect | comp vs truth | params')
    for arm, v in agg.items():
        print(f"   | {arm:5s} | {v['e0_median']:.2f} | {v['inc37_median']:.2f} | {v['inc90_median']:.2f} "
              f"| {v['comp_defect_median']:.2f} | {v['comp_truth_median']:.2f} | {v['n_params_total']}")
    print('-> results/interface_action_comparison.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
