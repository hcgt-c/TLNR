# -*- coding: utf-8 -*-
"""155_vit_intervention.py — activation-level hue intervention on transformer backbones.

Extends the CNN protocol (script 136) to ViT-B/16 (ImageNet-supervised, torchvision) and
DINOv2-B/14 (self-supervised, offline hub). Pipeline:
  * a frozen linear head is trained (CIFAR-10 labels) on the pooled FINAL features, so both routes
    share the same downstream;
  * operators (O1 global Procrustes, O5 per-token MLP, O8 3x3 conv over the patch grid) are fitted on
    TRAIN tokens at block k;
  * real route  y_real = head(pool(full(T_delta x)))
    null route  y_0    = head(pool(full(x)))
    intervention y_int = head(pool(blocks[k+1:] ... with block-k output replaced by W f_k(x)))
Metrics: effect size, rel L2 vs real, direction cosine, projection coefficient, win rate, top-1.

Usage: python scripts/155_vit_intervention.py --backbone vitb16 --block 4 --n_train 1000 --n_held 300 --delta 60
Outputs results/vit_causal_<backbone>_b<block>_s<seed>.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = rp.REPO_ROOT
HUB = rp.DINOV2_HUB
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_rbt():
    spec = importlib.util.spec_from_file_location("rbt", os.path.join(WORK, "scripts", "c10_routeB_train.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def build_backbone(name, dev):
    if name == "vitb16":
        import torchvision
        m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        return m.to(dev).eval(), "torchvision"
    if name in ("dinov2b14", "dinov2s14"):
        arch = "dinov2_vitb14" if name == "dinov2b14" else "dinov2_vits14"
        try:
            m = torch.hub.load(HUB, arch, source="local")
        except Exception as e:
            print("hub load failed:", str(e)[:160])
            sys.path.insert(0, HUB)
            from dinov2.models.vision_transformer import DinoVisionTransformer
            from dinov2.configs import dinov2_default_config as cfgmod
            cfg = cfgmod.get_config(arch)
            m = DinoVisionTransformer(**cfg)
            ck = torch.load(os.path.join(rp.TORCH_HUB, "hub", "checkpoints", f"{arch}_pretrain.pth"), map_location="cpu")
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ck["model"].items()}
            m.load_state_dict(sd, strict=False)
        return m.to(dev).eval(), "dinov2"
    raise ValueError(name)


def blocks_of(model, kind):
    if kind == "torchvision":
        return list(model.encoder.layers), model.encoder.ln
    return list(model.blocks), model.norm


def tokens_of(model, kind, x):
    if kind == "torchvision":
        t = model._process_input(x)
        cls = model.class_token.expand(t.shape[0], -1, -1)
        return torch.cat([cls, t], 1)
    return model.prepare_tokens_with_masks(x)


def pool(t):
    return t[:, 1:].mean(1)          # patch tokens only


def run_tail(model, kind, t, blocks, norm, k, op=None):
    for i, blk in enumerate(blocks):
        t = blk(t)
        if op is not None and i == k:
            t = op(t)
    t = norm(t)
    return t


def fit_ops(Fx, Fy, grid, seed, op8_epochs=600):
    B, T, D = Fx.shape
    X = Fx.reshape(-1, D); Y = Fy.reshape(-1, D)
    W1 = None
    M = Y.T @ X
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W1 = U @ Vh                                                   # Procrustes
    # per-token MLP residual
    torch.manual_seed(seed)
    Wr = torch.linalg.solve(X.T @ X + 1e-3 * torch.eye(D, device=X.device), X.T @ Y).T
    net = nn.Sequential(nn.Linear(D, 128), nn.SiLU(), nn.Linear(128, D)).to(X.device)
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(300):
        opt.zero_grad(); F.mse_loss(X @ Wr.T + net(X), Y).backward(); opt.step()
    def o5(t):
        Bt, Tt, Dt = t.shape
        r = t.reshape(-1, Dt)
        with torch.no_grad():
            out = r @ Wr.T + net(r)
        return out.reshape(Bt, Tt, Dt)
    # 3x3 conv residual over the patch grid (CLS token kept unchanged)
    H, Wg = grid
    Fxp, Fyp = Fx[:, 1:], Fy[:, 1:]                     # patch tokens only
    tgt = (Fyp - torch.einsum("btd,ed->bte", Fxp, Wr)).permute(0, 2, 1).reshape(B, D, H, Wg)
    src = Fxp.permute(0, 2, 1).reshape(B, D, H, Wg)
    conv = nn.Sequential(nn.Conv2d(D, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, D, 3, padding=1)).to(X.device)
    nn.init.zeros_(conv[-1].weight); nn.init.zeros_(conv[-1].bias)
    opt2 = torch.optim.Adam(conv.parameters(), lr=1e-3)
    for _ in range(op8_epochs):
        opt2.zero_grad(); F.mse_loss(conv(src), tgt).backward(); opt2.step()
    def o8(t):
        Bt, Tt, Dt = t.shape
        cls, p = t[:, :1], t[:, 1:]
        x = p.permute(0, 2, 1).reshape(Bt, Dt, H, Wg)
        with torch.no_grad():
            out = torch.einsum("bdhw,ed->behw", x, Wr) + conv(x)
        return torch.cat([cls, out.reshape(Bt, Dt, H * Wg).permute(0, 2, 1)], 1)
    return {"O1_procrustes": lambda t: t @ W1.T, "O5_mlp": o5, "O8_conv_residual": o8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitb16")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_held", type=int, default=300)
    ap.add_argument("--n_head", type=int, default=2000)
    ap.add_argument("--delta", type=float, default=60.0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--op8_epochs", type=int, default=150)
    ap.add_argument("--fit_imgs", type=int, default=150, help="images used to fit operators (memory/time)")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rbt = load_rbt()
    model, kind = build_backbone(a.backbone, dev)
    blocks, norm = blocks_of(model, kind)
    patch = model.patch_size if kind == "dinov2" else 16
    grid = (a.size // patch, a.size // patch)
    D = model.embed_dim if kind == "dinov2" else model.hidden_dim

    x_tr_u8, y_tr, x_te_u8, y_te = rbt.load_cifar()
    rng = np.random.default_rng(0)
    tr = rng.permutation(len(x_te_u8))[:a.n_train]
    he = rng.permutation(len(x_te_u8))[:a.n_held]
    hd = rng.permutation(len(x_tr_u8))[:a.n_head]

    mean = MEAN.to(dev); std = STD.to(dev)

    def prep(u8, shift=0.0):
        x = torch.from_numpy(u8).float().permute(0, 3, 1, 2).to(dev) / 255.0
        x = F.interpolate(x, size=(a.size, a.size), mode="bilinear", align_corners=False)
        if shift:
            x = rbt.rgb_hue_shift_t(x, torch.full((x.shape[0],), float(shift), device=dev))
        return (x - mean) / std

    def feats_full(u8, shift=0.0, bs=100):
        outs = []
        with torch.no_grad():
            x = prep(u8, shift)
            for s in range(0, x.shape[0], bs):
                t = tokens_of(model, kind, x[s:s + bs])
                t = run_tail(model, kind, t, blocks, norm, k=-1)
                outs.append(pool(t).cpu().numpy())
        return np.concatenate(outs, 0)

    # frozen linear head on FINAL pooled features
    Xh = feats_full(x_tr_u8[hd]); yh = y_tr[hd]
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xh, yh)
    def head_np(feats):
        return clf.predict_proba(feats)

    xtr = prep(x_tr_u8[tr]); xhe = prep(x_te_u8[he])
    y_he = y_te[he]

    # ---- features at block k for operator fitting (train) and held-out
    def feats_block(x, bs=100):
        outs = []
        with torch.no_grad():
            for s in range(0, x.shape[0], bs):
                t = tokens_of(model, kind, x[s:s + bs])
                for i, blk in enumerate(blocks[:a.block + 1]):
                    t = blk(t)
                outs.append(t.cpu())
        return torch.cat(outs, 0)

    nfit = min(a.fit_imgs, a.n_train)
    Fx = feats_block(xtr[:nfit]).to(dev)
    x_tr_sh = prep(x_tr_u8[tr][:nfit], a.delta)
    Fy = feats_block(x_tr_sh).to(dev)
    ops = fit_ops(Fx, Fy, grid, a.seed, a.op8_epochs)

    # ---- routes on held-out
    def full_route(x):
        with torch.no_grad():
            t = tokens_of(model, kind, x)
            t = run_tail(model, kind, t, blocks, norm, k=-1)
            return head_np(pool(t).cpu().numpy())

    def int_route(x, op):
        with torch.no_grad():
            t = tokens_of(model, kind, x)
            t = run_tail(model, kind, t, blocks, norm, k=a.block, op=op)
            return head_np(pool(t).cpu().numpy())

    x_real = prep(x_te_u8[he], a.delta)
    y_real = full_route(x_real)
    y_null = full_route(xhe)
    out = {"backbone": a.backbone, "block": a.block, "seed": a.seed, "delta": a.delta,
           "n_train": a.n_train, "n_held": a.n_held, "n_head": a.n_head, "grid": list(grid), "dim": D,
           "head_train_acc": float((clf.predict(Xh) == yh).mean()),
           "power": {}, "ops": {}}
    p_real = y_real.argmax(1); p_null = y_null.argmax(1)
    out["power"] = {"top1_flip_rate": float((p_real != p_null).mean()),
                    "acc_null": float((p_null == y_he).mean()), "acc_real": float((p_real == y_he).mean())}
    d_null = np.linalg.norm(y_null - y_real, axis=1) / (np.linalg.norm(y_real, axis=1) + 1e-9)
    out["power"]["rel_l2_real_vs_null"] = float(d_null.mean())
    for name, op in ops.items():
        y_int = int_route(xhe, op)
        d_int = np.linalg.norm(y_int - y_real, axis=1) / (np.linalg.norm(y_real, axis=1) + 1e-9)
        num = ((y_int - y_null) * (y_real - y_null)).sum(1)
        den = ((y_real - y_null) ** 2).sum(1) + 1e-12
        cos = num / (np.linalg.norm(y_int - y_null, axis=1) * np.linalg.norm(y_real - y_null, axis=1) + 1e-12)
        out["ops"][name] = {"rel_l2_vs_real": float(d_int.mean()),
                            "align_cos": float(cos.mean()),
                            "proj_coef_agg": float(num.sum() / den.sum()),
                            "win_rate": float((d_int < d_null).mean()),
                            "top1_agree_real": float((y_int.argmax(1) == p_real).mean()),
                            "acc_int": float((y_int.argmax(1) == y_he).mean())}
    fn = os.path.join(WORK, "results", f"vit_causal_{a.backbone}_b{a.block}_s{a.seed}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)
    print("power:", {k: round(v, 4) for k, v in out["power"].items()})
    for k_, v in out["ops"].items():
        print(f"  {k_:20s} relL2 {v['rel_l2_vs_real']:.3f} cos {v['align_cos']:.2f} "
              f"proj {v['proj_coef_agg']:.2f} win {v['win_rate']:.2f} top1 {v['top1_agree_real']:.2f}")


if __name__ == "__main__":
    main()
