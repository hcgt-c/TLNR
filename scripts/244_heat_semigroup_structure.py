# -*- coding: utf-8 -*-
"""244_heat_semigroup_structure.py — before writing an interface, find out what obstruction there is.

The sequence of experiments so far asked whether a GLOBAL LINEAR carrier exists for the heat semigroup,
and kept answering no. That is the wrong question, and the project's own hue result says why: global
families fail and local/content-conditioned ones succeed. The induced action g_t (z -> f(B_t x)) satisfies
the semigroup law EXACTLY, but it is nonlinear, so a common eigenbasis for its linear approximations need
not exist.

This script measures the obstruction directly and cheaply, instead of guessing views:

  1. DO THE LINEAR APPROXIMATIONS COMMUTE?  A common eigenbasis for {A_t} exists iff the family commutes
     (for diagonalisable members). ||A_t A_s - A_s A_t|| / ||A_t A_s|| is the decisive number, and it is
     also a lower bound on how far any single global basis can be from a carrier.
  2. ARE THEY DIAGONALISABLE OVER R?  Real, positive, decreasing spectra are the diagonal-decay signature.
  3. ARE THE DECAY RATES SHARED ACROSS IMAGES?  Project every image's trajectory onto the top-K modes of
     A_1 and fit a per-image rate. Small dispersion = shared rate = a fixed action can serve every input;
     large dispersion = the rate is content-dependent and the interface must READ the rate from the content.

(3) is the one that decides what to build. A shared rate means "learned read-in + fixed decay". A
content-dependent rate means "learned read-in + learned rate read from the same content" -- the exact
mirror of the hue interface, where the phase is shared and only the amplitude is content-dependent.

A synthetic control with a KNOWN shared carrier is run first; the script refuses to report real numbers if
the control does not recover it.

Usage: python scripts/244_heat_semigroup_structure.py --backbone resnet50 --depths 2,3
Outputs results/heat_semigroup_structure.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, argparse, importlib.util
import torch

WORK = rp.REPO_ROOT
RES = os.path.join(WORK, "results")
T_ALL = [0.5, 2.0]          # the two operators whose commutation is tested
T_FIT = [0.5, 1.0]
T_TEST = 2.0


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gram_cap(X0, Xt, r_max):
    X0c = (X0 - X0.mean(0, keepdim=True)).double()
    G = X0c @ X0c.T
    ew, ev = torch.linalg.eigh(G)
    ew, ev = ew.flip(0).clamp_min(0), ev.flip(1)
    r = min(r_max, int((ew > ew.max() * 1e-8).sum()), ew.numel())
    Bd = (ev[:, :r].T @ X0c) / ew[:r].sqrt()[:, None]
    red = {t: ((Y - Y.mean(0, keepdim=True)).double() @ Bd.T).float() for t, Y in Xt.items()}
    return (X0c @ Bd.T).float(), red, float(ew[:r].sum() / ew.sum()), r


def op(X0, Xt, ridge=1e-2):
    n = X0.shape[0]
    C0 = (X0.T @ X0) / max(1, n - 1)
    Ct = (Xt.T @ X0) / max(1, n - 1)
    d = C0.shape[0]
    C0r = C0 + ridge * (torch.diagonal(C0).mean() + 1e-12) * torch.eye(d, dtype=C0.dtype)
    return (torch.linalg.solve(C0r.T, Ct.T).T).double(), C0.double(), C0r.double()


def structure(X0, Xt, K=24):
    A = {t: op(X0, Xt[t])[0] for t in T_ALL}
    comm = {}
    ta, tb = T_ALL
    lhs = A[ta] @ A[tb]
    rhs = A[tb] @ A[ta]
    comm["relative_commutator"] = float((lhs - rhs).norm() / (lhs.norm() + 1e-12))
    comm["relative_commutator_normalised_by_scale"] = float(
        (lhs - rhs).norm() / ((A[ta].norm() * A[tb].norm()) + 1e-12))
    # spectra: real, positive, decreasing?
    specs = {}
    for t in T_ALL:
        w = torch.linalg.eigvals(A[t])
        specs[f"t{t}"] = {"max_abs_imag": float(w.imag.abs().max()),
                          "frac_real_positive": float((w.real > 0).double().mean()),
                          "top5_real": [float(v) for v in w.real.sort(descending=True).values[:5]]}
    # INVARIANCE test — the correct criterion for a shared carrier subspace, and well conditioned unlike
    # "diagonalise with U^{-1}", which is destroyed by the near-null noise subspace (the control caught it:
    # a genuinely shared carrier scored 0.997 off-diagonal there). A carrier subspace must be mapped INTO
    # ITSELF by every A_t, so measure the leakage (I-P) A_t P against the within-subspace part P A_t P.
    def leakage(A_t, V0_, K):
        Q = torch.linalg.qr(V0_[:, :K].double().real).Q          # orthonormal basis of the candidate carrier
        P = Q @ Q.T
        within = P @ A_t.double() @ P
        out = (torch.eye(A_t.shape[0], dtype=torch.float64) - P) @ A_t.double() @ P
        return float(out.norm() / (within.norm() + 1e-12))

    wref, Vref = torch.linalg.eig(A[T_ALL[0]])
    order = torch.argsort(wref.real, descending=True)
    Vref = Vref.real[:, order]
    for tt in T_ALL[1:]:
        comm[f"leakage_top{K}_at_t{tt}"] = leakage(A[tt], Vref, K)
    comm[f"leakage_sanity_same_t{T_ALL[0]}"] = leakage(A[T_ALL[0]], Vref, K)

    # per-image rates in a common basis: project on the top-K modes of A[T_ALL[1]] and fit each image
    w1, V1 = torch.linalg.eig(A[T_ALL[1]])
    order = torch.argsort(w1.real, descending=True)
    U = V1.real[:, order[:K]]
    code = {t: (Xt[t].double() @ U) for t in T_FIT + [T_TEST]}
    c_ref = code[max(T_FIT)]
    tt = torch.tensor(T_FIT, dtype=torch.float64)
    # per-image, per-mode rate from the two fitted t's: log|c| linear in t
    ly = torch.stack([torch.log(code[t].abs() + 1e-12) for t in T_FIT], 0)      # (2,n,K)
    tc = tt - tt.mean()
    slope = ((ly - ly.mean(0, keepdim=True)) * tc[:, None, None]).sum(0) / (tc ** 2).sum()
    lam = -slope                                                                # (n,K)
    lam_mean = lam.mean(0)
    lam_sd = lam.std(0)
    # only modes with a meaningful amplitude carry information about the rate
    amp = c_ref.abs().mean(0)
    keep = amp > amp.median()
    disp = (lam_sd[keep] / lam_mean[keep].abs().clamp_min(1e-9))
    # prediction at the held-out t using the SHARED mean rate (what a fixed action would do)
    pred = c_ref * torch.exp(-lam_mean[None, :] * (T_TEST - max(T_FIT)))
    truth = code[T_TEST]
    err = float((pred - truth).norm() / (truth.norm() + 1e-12))
    copy_err = float((c_ref - truth).norm() / (truth.norm() + 1e-12))
    return {"commutator": comm, "spectra": specs,
            "rate_dispersion": {"median_relative_sd_of_per_image_rate": float(disp.median()),
                                "mean_relative_sd": float(disp.mean()),
                                "frac_modes_below_0.25": float((disp < 0.25).double().mean())},
            "shared_rate_prediction_at_held_out_t": {"err": err, "copy": copy_err,
                                                     "gain": float(copy_err / (err + 1e-12))}}


def synthetic_control():
    """A true shared carrier: A_t = Q diag(exp(-lam t)) Q^T with Q orthogonal -> the family commutes."""
    torch.manual_seed(0)
    n, d, K = 300, 256, 8
    Q, _ = torch.linalg.qr(torch.randn(d, K, dtype=torch.float64))
    lam = torch.linspace(0.05, 1.0, K, dtype=torch.float64)
    a = torch.randn(n, K, dtype=torch.float64)
    X = {0.0: a @ Q.T}
    for t in sorted(set(T_FIT + [T_TEST] + T_ALL)):
        X[t] = (a * torch.exp(-lam * t)[None, :]) @ Q.T
    X = {t: (Y + 0.005 * torch.randn(n, d, dtype=torch.float64)).float() for t, Y in X.items()}
    X0, red, var, r = gram_cap(X[0.0], X, 256)
    s = structure(X0, red)
    # The control's ground truth commutes EXACTLY, so whatever commutator/dispersion it shows is the
    # ESTIMATION NOISE FLOOR of the metric, not a property of the data. Gate only on the prediction gain
    # (which validates that the estimator recovers a real carrier) and carry the floor alongside every
    # real number: a real commutator is only evidence if it is well ABOVE the control's.
    ok = s["shared_rate_prediction_at_held_out_t"]["gain"] > 2.0
    print(f"[CONTROL] commutator {s['commutator']['relative_commutator']:.4f}  "
          f"rate dispersion {s['rate_dispersion']['median_relative_sd_of_per_image_rate']:.3f}  "
          f"gain {s['shared_rate_prediction_at_held_out_t']['gain']:.2f}  -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--depths", default="2,3")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--dim_cap", type=int, default=256)
    a = ap.parse_args()

    ok, ctrl = synthetic_control()
    out = {"script": "244_heat_semigroup_structure.py", "backbone": a.backbone,
           "question": "is the obstruction to a heat interface that (i) the linear approximations do not "
                       "commute, (ii) they are not diagonalisable, or (iii) the decay RATE is content-"
                       "dependent? (iii) decides what to build next.",
           "control": {"recovered_known_carrier": bool(ok), "numbers": ctrl,
            "note": "the control ground truth commutes exactly; its commutator and rate dispersion are the "
                    "ESTIMATION NOISE FLOOR of those two metrics and every real value must be read against them"},
           "noise_floor": {"commutator": ctrl["commutator"]["relative_commutator"],
                           "rate_dispersion": ctrl["rate_dispersion"]["median_relative_sd_of_per_image_rate"]},
           "t_all": T_ALL, "t_fit": T_FIT, "t_test": T_TEST, "sites": {}}
    if not ok:
        print("control FAILED — refusing to report real numbers", flush=True)
        json.dump(out, open(os.path.join(RES, "heat_semigroup_structure.json"), "w"), indent=1)
        return 1

    h = load(os.path.join(WORK, "scripts", "188_depth_map_harness.py"), "h188")
    h.DEV = "cpu"
    model, sites, names, kind = h.build_backbone(a.backbone)
    x01 = h.to01(h.load_crops(a.n, seed=a.seed))
    F = {0.0: h.capture(model, sites, h.norm(x01))}
    for t in sorted(set(T_FIT + [T_TEST] + T_ALL)):
        F[t] = h.capture(model, sites, h.norm(h.heat(x01, t)))
    print(f"[{a.backbone}] captured", flush=True)

    for d in [int(v) for v in a.depths.split(",")]:
        i = d - 1
        site = names[i]
        Z = F[0.0][i]
        X0, Xt, var, r = gram_cap(Z.reshape(Z.shape[0], -1),
                                  {t: F[t][i].reshape(Z.shape[0], -1) for t in set(T_FIT + [T_TEST] + T_ALL)},
                                  a.dim_cap)
        s = structure(X0, Xt, a.K)
        s["var_kept_by_cap"] = var
        out["sites"][site] = s
        c = s["commutator"]["relative_commutator"]
        rd = s["rate_dispersion"]["median_relative_sd_of_per_image_rate"]
        g = s["shared_rate_prediction_at_held_out_t"]["gain"]
        print(f"  {site:9s} commutator {c:.4f} | rate dispersion {rd:.3f} | gain {g:.3f}", flush=True)
        json.dump(out, open(os.path.join(RES, "heat_semigroup_structure.json"), "w"), indent=1)

    json.dump(out, open(os.path.join(RES, "heat_semigroup_structure.json"), "w"), indent=1)
    print("saved results/heat_semigroup_structure.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
