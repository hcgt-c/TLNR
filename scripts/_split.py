# -*- coding: utf-8 -*-
"""_split.py — one place for the fit/held-out split convention.

The defect this module exists to prevent: drawing the fit block and the held-out block from **independent**
permutations, so that with n_fit = 300 and n_held = 200 out of 10,000 images about six held-out images also
enter the fit (expected overlap n_fit * n_held / N). Scripts that fit an operator and then report a number
for it must use `disjoint_split`, which takes ONE permutation and slices it, so the blocks are disjoint by
image index by construction; `assert_disjoint` makes the guarantee explicit at run time.

Usage:
    from _split import disjoint_split
    tr_idx, he_idx = disjoint_split(len(x), n_fit, n_held, seed)
"""
import numpy as np


def disjoint_split(n, n_fit, n_held, seed=0, order="head"):
    """Return (fit_idx, held_idx) drawn from a single permutation: disjoint by index by construction.

    order="head" gives the first n_fit to the fit block and the next n_held to held-out; order="tail" puts
    the held-out block first (useful when a legacy protocol gave the fit block the head of the index range).
    """
    if n_fit + n_held > n:
        raise ValueError(f"n_fit + n_held = {n_fit + n_held} exceeds n = {n}")
    perm = np.random.default_rng(seed).permutation(n)
    if order == "tail":
        he = perm[:n_held]
        tr = perm[n_held:n_held + n_fit]
    else:
        tr = perm[:n_fit]
        he = perm[n_fit:n_fit + n_held]
    assert_disjoint(tr, he)
    return tr, he


def assert_disjoint(a, b):
    inter = set(np.asarray(a).tolist()) & set(np.asarray(b).tolist())
    if inter:
        raise AssertionError(f"fit and held-out blocks overlap on {len(inter)} indices, e.g. {sorted(inter)[:5]}")
    return True
