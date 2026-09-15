# -*- coding: utf-8 -*-
"""固化密集连续实验(34-38)关键数值 → results/dense_continuous_v3.json"""
import numpy as np, os, json
import numpy.linalg as la
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE)
d = np.load(os.path.join(OUT, "features", "dense5_y3.npz"))
feats, shape, hue = d["feats"], d["shape"], d["hue"]
TRAIN = ['triangle','rectangle','circle','star']; TEST=['pentagon','hexagon']
ALL = TRAIN+TEST; HOLD=[20,260]; hues=np.sort(np.unique(hue)); NV=4; N=len(hues)
Z = feats.reshape(len(ALL), N, NV, feats.shape[1]).mean(2)
def ctraj(sh): return Z[ALL.index(sh)]-Z[ALL.index(sh)].mean(0)
R = {"protocol": "dense5 y3 GAP 位姿平均(4var) 去DC 5°步; 测试色相{20,260}排除于拟合",
     "spectrum": {}, "mean_winding": {}, "alignment_vs_triangle": {}, "k3_share": {},
     "intrinsic_dim": {}}
for sh in ALL:
    F = np.fft.rfft(ctraj(sh), axis=0); E=(np.abs(F)**2).sum(1); Et=E.sum()
    R["spectrum"][sh] = {f"k{k}": round(float(E[k]/Et),4) for k in range(1,7)}
    R["spectrum"][sh]["k1k2_sum"] = round(float((E[1]+E[2])/Et),4)
    R["spectrum"][sh]["tail_k3plus"] = round(float(1-(E[1]+E[2])/Et),4)
    R["mean_winding"][sh] = round(float(np.sum(np.arange(len(E))*E)/Et),3)
    R["k3_share"][sh] = round(float(E[3]/Et),4)
    C = ctraj(sh).T @ ctraj(sh)/len(hues); ev = la.eigvalsh(C)[::-1]; cum=np.cumsum(ev)/ev.sum()
    R["intrinsic_dim"][sh] = {"d50":int(np.searchsorted(cum,.5)+1),"d80":int(np.searchsorted(cum,.8)+1),"d95":int(np.searchsorted(cum,.95)+1)}
def plane_k(sh,k):
    F=np.fft.rfft(ctraj(sh),axis=0)[k]; q,_=la.qr(np.stack([F.real,F.imag],1)); return q
ref={k:plane_k(TRAIN[0],k) for k in range(1,11)}
R["alignment_vs_triangle"]={sh:{f"k{k}":round(float(la.svd(plane_k(sh,k).T@ref[k],compute_uv=False)[0]),3) for k in range(1,11)} for sh in TRAIN[1:]+TEST}
# 带限算子 gain + 合成律
for K,tag in [(2,"band_op_K2"),(3,"band_op_K3"),(4,"band_op_K4")]:
    rows,ys=[],[]
    for sh in TRAIN:
        Zc=ctraj(sh); z0=Zc[0]
        for hi,h in enumerate(hues):
            if h in HOLD: continue
            rows.append(np.concatenate([np.cos(np.deg2rad(k*h))*z0 for k in range(1,K+1)]+[np.sin(np.deg2rad(k*h))*z0 for k in range(1,K+1)])); ys.append(Zc[hi])
    X,Y=np.array(rows),np.array(ys); W=la.solve(X.T@X+1e-6*np.eye(X.shape[1]),X.T@Y)
    def T(z0,h):
        ph=np.concatenate([np.cos(np.deg2rad(k*h))*z0 for k in range(1,K+1)]+[np.sin(np.deg2rad(k*h))*z0 for k in range(1,K+1)]); return ph@W
    eo=[np.sum((T(ctraj(sh)[0],h)-ctraj(sh)[hi])**2) for sh in TEST for hi,h in enumerate(hues) if h in HOLD]
    ec=[np.sum((ctraj(sh)[0]-ctraj(sh)[hi])**2) for sh in TEST for hi,h in enumerate(hues) if h in HOLD]
    hs=[h for h in hues if h not in HOLD]; comp=[]; sing=[]
    for sh in TEST:
        Zc=ctraj(sh); z0=Zc[0]
        for h1 in hs:
            for h2 in hs:
                h3=(h1+h2)%360
                if h3 in HOLD or h3==0: continue
                tgt=Zc[hues.tolist().index(h3)]
                comp.append(np.sum((T(T(z0,h1),h2)-tgt)**2)/np.sum(tgt**2)); sing.append(np.sum((T(z0,h3)-tgt)**2)/np.sum(tgt**2))
    R[tag]={"cross_gain_vs_copy": round(float(np.mean(ec)/np.mean(eo)),2),
            "cross_single_err": round(float(np.mean(eo)),4),
            "composition_err": round(float(np.mean(comp)),4), "copy_err_comp": round(float(np.mean(sing)),4)}
os.makedirs(os.path.join(OUT,"results"),exist_ok=True)
json.dump(R, open(os.path.join(OUT,"results","dense_continuous_v3.json"),"w"), indent=1, ensure_ascii=False)
print("saved:", os.path.join(OUT,"results","dense_continuous_v3.json"))
