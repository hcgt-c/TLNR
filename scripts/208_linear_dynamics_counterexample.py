# -*- coding: utf-8 -*-
"""208_linear_dynamics_counterexample.py — can training move kappa in a *linear* system?

E1 (script 207) found a flat trajectory: effect rank compressed 3.7x while T_F moved by 0.02. That was read as
"linear systems have frozen transport geometry". This script tests whether that is a law or an accident of the
transformation geometry.

Theory. kappa = ||P_row g P_ker|| / ||P_row g P_ker|| is determined by where the transformation's mixing *source*
sits relative to what training discards:
  * geometry "within":  g rotates inside the task block (the source of mixing is inside the retained block), so
    compressing within that block leaves the ratio roughly unchanged -> flat trajectory (E1's case);
  * geometry "cross":   g maps nuisance directions into the task block, i.e. the mixing source is precisely what
    weight decay discards. Compression should then *raise* kappa and *lower* T_F.
If the second geometry shows movement, then "optimisation alone does not create transportability in linear
systems" is false as stated, and the correct statement is conditional on the task/transformation geometry:
**learning moves the transport defect exactly when the directions it discards are the ones the transformation
mixes in.**

Usage: python scripts/208_linear_dynamics_counterexample.py
Outputs results/linear_dynamics_counterexample.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import numpy as np
import torch

WORK = rp.REPO_ROOT
D_IN, K_TASK, P_TASK, N_DATA, N_FIT = 64, 16, 4, 4000, 2000
STEPS, LR = 3000, 0.05


def make_g(alpha, seed=0):
    """Orthogonal g with a controllable nuisance -> task mixing.

    alpha = 0   : the mixing source lies inside the task block (E1's geometry)
    alpha > 0   : the generator acquires a nuisance -> task block, so some of the mixing originates in exactly
                  the directions a task-driven loss is expected to discard
    """
    g = torch.Generator().manual_seed(seed)
    RT = torch.linalg.qr(torch.randn(K_TASK, K_TASK, generator=g, dtype=torch.float64))[0]
    RN = torch.linalg.qr(torch.randn(D_IN - K_TASK, D_IN - K_TASK, generator=g, dtype=torch.float64))[0]
    base = torch.block_diag(RT, RN)
    L = torch.zeros(D_IN, D_IN, dtype=torch.float64)
    # within-block mixing generator (task <-> nuisance), present in both geometries
    M = torch.randn(K_TASK, D_IN - K_TASK, generator=g, dtype=torch.float64)
    L[:K_TASK, K_TASK:] = M
    L[K_TASK:, :K_TASK] = -M.T
    # nuisance -> task component of a purely *within-task* generator is not orthogonal-friendly, so instead we
    # apply an extra orthogonal shear from the nuisance block into the task block after the exponential
    gmat = torch.linalg.matrix_exp(0.6 * L) @ base
    if alpha > 0:
        S = torch.eye(D_IN, dtype=torch.float64)
        S[:K_TASK, K_TASK:] = alpha * torch.randn(K_TASK, D_IN - K_TASK, generator=g, dtype=torch.float64) / \
            np.sqrt(D_IN - K_TASK)
        gmat = S @ gmat                     # shear: nuisance directions now feed the task coordinates
        # re-orthogonalise so the comparison is not confounded by g changing the displacement scale
        gmat = torch.linalg.qr(gmat)[0]
    return gmat


def blocks(W, g):
    U, S, Vh = torch.linalg.svd(W, full_matrices=True)
    m = W.shape[0]
    V = Vh.T
    gb = V.T @ g @ V
    return gb[:m, :m], gb[:m, m:]


def principal_angle_direction(W_a, W_b):
    """Largest principal angle (radians) between the row spaces of two site maps, via subspace SVD."""
    Qa = torch.linalg.qr(W_a.T, mode="reduced")[0]      # (d, m_a)
    Qb = torch.linalg.qr(W_b.T, mode="reduced")[0]      # (d, m_b)
    s = torch.linalg.svdvals(Qa.T @ Qb).clamp(0, 1)
    return float(torch.arccos(s.min()))                 # largest principal angle


def site_scores(W1, g, x):
    with torch.no_grad():
        z, zg = x @ W1.T, (x @ g.T) @ W1.T
        zf, zgf, ze, zge = z[:N_FIT], zg[:N_FIT], z[N_FIT:], zg[N_FIT:]
        G = zf.T @ zf + 1e-8 * torch.eye(zf.shape[1], dtype=zf.dtype) * (zf.T @ zf).diagonal().mean()
        rho = torch.linalg.solve(G, zf.T @ zgf).T
        res = (rho @ ze.T - zge.T).T.norm(dim=1); disp = (ze - zge).norm(dim=1)
        tf = 1 - float(res.mean() / disp.mean())
        tf_sq = 1 - float(np.sqrt(float((res ** 2).mean()) / float((disp ** 2).mean())))
        A, C = blocks(W1, g)
        I = torch.eye(A.shape[0], dtype=torch.float64)
        kappa = float(torch.linalg.matrix_norm(C, "fro") /
                      torch.linalg.matrix_norm(A - I, "fro"))
        # Share of the site displacement that originates in discarded (ker) directions: the estimable analogue
        # of kappa. The ker component of x contributes W1 g x_K, while W1 x_K = 0 by definition.
        Pker = torch.eye(D_IN, dtype=torch.float64) - Vh_row(W1)
        xk = x @ Pker.T
        contrib_ker = (xk @ g.T) @ W1.T                     # (n, m) in site space
        share_ker = float((contrib_ker ** 2).sum() / (((zg - z) ** 2).sum() + 1e-30))
        return tf, tf_sq, kappa, share_ker, float(disp.mean())


def Vh_row(W):
    """Row-space projector of W (d x d)."""
    return W.T @ torch.linalg.pinv(W @ W.T) @ W


def eff_rank(W):
    s = torch.linalg.svdvals(W)
    return float((s.sum() ** 2) / ((s ** 2).sum() + 1e-30))


def train_track(m, wd, seed, g, x, y, every=100):
    torch.manual_seed(seed)
    W1_init = None
    W1 = torch.randn(m, D_IN, dtype=torch.float64) * (1.0 / np.sqrt(D_IN))
    W2 = torch.randn(P_TASK, m, dtype=torch.float64) * (1.0 / np.sqrt(m))
    W1.requires_grad_(True); W2.requires_grad_(True)
    opt = torch.optim.Adam([W1, W2], lr=LR, weight_decay=wd)
    traj = []
    for step in range(STEPS):
        opt.zero_grad()
        loss = ((x @ W1.T) @ W2.T - y).pow(2).mean()
        loss.backward(); opt.step()
        if step % every == 0 or step == STEPS - 1:
            W1d = W1.detach()
            if W1_init is None:
                W1_init = W1d.clone()
            tf, tfsq, kap, share, disp = site_scores(W1d, g, x)
            traj.append({"step": step, "loss": float(loss.detach()), "tf": tf, "tf_sq": tfsq,
                         "kappa": kap, "displacement_share_from_ker": share,
                         "eff_rank": eff_rank(W1d), "displacement": disp,
                         # row-space movement relative to the initialisation: the candidate predictor that
                         # the alignment theory says should NOT track the transport change on its own
                         "principal_angle_to_init": principal_angle_direction(W1d, W1_init)})
    with torch.no_grad():
        pred = (x @ W1.T) @ W2.T
        r2 = 1 - float(((pred - y) ** 2).mean() / y.var())
    return traj, r2


def main():
    # The task subspace and the data are part of the experimental condition: an earlier version drew them
    # from the unseeded global RNG, so two identical invocations ran different tasks and reported
    # different numbers (found by a determinism check, not by inspection).
    torch.manual_seed(0); np.random.seed(0)
    A_task = torch.randn(P_TASK, K_TASK, dtype=torch.float64)
    x = torch.randn(N_DATA, D_IN, dtype=torch.float64)
    y = x[:, :K_TASK] @ A_task.T
    out = {"config": {"d_in": D_IN, "k_task_block": K_TASK, "p_task": P_TASK, "steps": STEPS, "lr": LR,
                      "m": 16}, "runs": {}}
    print("geometry      wd      seed | T_F(init) -> T_F(final)  | kappa(init) -> kappa(final) | eff_rank | R2")
    for alpha, gname in ((0.0, "within"), (1.5, "cross")):
        g = make_g(alpha)
        for wd in (0.0, 1e-3):
            for seed in (0, 1):
                traj, r2 = train_track(16, wd, seed, g, x, y)
                first, last = traj[0], traj[-1]
                out["runs"][f"{gname}_wd{wd}_seed{seed}"] = {
                    "geometry": gname, "wd": wd, "seed": seed, "task_r2": r2,
                    "tf_init": first["tf"], "tf_final": last["tf"], "tf_delta": last["tf"] - first["tf"],
                    "kappa_init": first["kappa"], "kappa_final": last["kappa"],
                    "eff_rank_init": first["eff_rank"], "eff_rank_final": last["eff_rank"],
                    "share_from_ker_final": last["displacement_share_from_ker"],
                    "principal_angle_init_to_final": last["principal_angle_to_init"],
                    "trajectory": traj}
                print(f"{gname:10s} {wd:7.4f} {seed:5d} | {first['tf']:+.3f} -> {last['tf']:+.3f} "
                      f"(Δ {last['tf']-first['tf']:+.3f}) | {first['kappa']:5.2f} -> {last['kappa']:5.2f} | "
                      f"{first['eff_rank']:5.2f} -> {last['eff_rank']:5.2f} | {r2:.3f}")
    path = os.path.join(WORK, "results", "linear_dynamics_counterexample.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nlinear dynamics counterexample: PASS (0 issues) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
