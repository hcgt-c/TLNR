#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""263_two_term_heldout.py — does the two-term decomposition predict HELD-OUT error, at full-vector scale?

Design (fixed before coding):
  split F (fit) / H (held-out), disjoint, asserted.
  per (site, family), ALL reported numbers are evaluated on H:
    R_diag        : per-channel affine law fitted on F, SSR on H, normalised by the no-op SSR on H
    S_cell        : per-cell affine law fitted on F, SSR on H   (= within+sharing realised out of sample)
    S_pool        : pooled affine law fitted on F, SSR on H
    sharing       : S_pool - S_cell
    R_full(k)     : linear map fitted on F in the top-k PCA subspace of F, SSR on H
  Reading: S_pool vs R_diag tests whether the decomposition accounts for the diagonal operator's
  held-out error; R_full(k) vs R_diag gives the cross-channel coupling gain.
Outputs results/two_term_heldout_resnet50.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import argparse, importlib.util, json, os
import numpy as np, torch
from PIL import Image
WORK=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN=rp.COCO_ANNOTATIONS
IMG=rp.COCO_IMAGES; SIZE=224
def load(n,p):
    sp=importlib.util.spec_from_file_location(n,os.path.join(WORK,'scripts',p)); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
def affine_fit(u,v):
    A=np.stack([u,np.ones_like(u)],1); c,*_=np.linalg.lstsq(A,v,rcond=None); return c
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=300); ap.add_argument('--holdout',type=float,default=0.4)
    ap.add_argument('--sites',default='layer1,layer2,layer3,layer4'); ap.add_argument('--ks',default='8,16,32')
    ap.add_argument('--locs',type=int,default=6); a=ap.parse_args()
    h=load('h188','188_depth_map_harness.py')
    ann=json.load(open(ANN)); imgs={im['id']:im for im in ann['images']}
    boxes=[(an['image_id'],*an['bbox']) for an in ann['annotations'] if not an.get('iscrowd') and an['bbox'][2]>=80 and an['bbox'][3]>=80 and an['bbox'][2]*an['bbox'][3]>=20000]
    rng=np.random.default_rng(0); rng.shuffle(boxes); crops=[]
    for imid,x,y,w,hh in boxes:
        try: I=Image.open(os.path.join(IMG,imgs[imid]['file_name'])).convert('RGB')
        except Exception: continue
        crops.append(I.crop((int(x),int(y),int(x+w),int(y+hh))).resize((SIZE,SIZE),Image.BILINEAR))
        if len(crops)>=a.n: break
    X=torch.from_numpy(np.stack([np.asarray(c,np.float32)/255 for c in crops])).permute(0,3,1,2).contiguous()
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    from torchvision.models import resnet50, ResNet50_Weights
    net=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(dev).eval()
    for p in net.parameters(): p.requires_grad_(False)
    mean=torch.tensor([0.485,0.456,0.406],device=dev)[None,:,None,None]; std=torch.tensor([0.229,0.224,0.225],device=dev)[None,:,None,None]
    stem=torch.nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool)
    def feats(x01,site):
        with torch.no_grad():
            hh=stem((x01.to(dev)-mean)/std); hh=net.layer1(hh)
            if site in ('layer2','layer3','layer4'): hh=net.layer2(hh)
            if site in ('layer3','layer4'): hh=net.layer3(hh)
            if site=='layer4': hh=net.layer4(hh)
            return hh.mean(dim=(2,3)).cpu().numpy()
    N=X.shape[0]; nfit=int(N*(1-a.holdout)); idx=np.arange(N); rng.shuffle(idx); F=idx[:nfit]; H=idx[nfit:]
    assert len(set(F)&set(H))==0, 'split overlap'
    assert len(F)+len(H)==N
    ks=[int(k) for k in a.ks.split(',')]
    out={'device':dev,'n_crops':N,'n_fit':len(F),'n_held':len(H),'ks':ks,'sites':{}}
    for site in a.sites.split(','):
        Z=feats(X,site); C=Z.shape[1]; rec={}
        for fam,Xt in (('hue_90.0',h.hue_shift(X,90.0)),('heat_2.0',h.heat(X,2.0))):
            Zt=feats(Xt,site)
            disp=float(((Zt[H]-Z[H])**2).sum(1).mean())                  # no-op displacement on H
            S_cell=S_pool=0.0
            for c in range(C):
                uF,vF=Z[F,c].astype(np.float64),Zt[F,c].astype(np.float64)
                uH,vH=Z[H,c].astype(np.float64),Zt[H,c].astype(np.float64)
                cells={}
                for gu in (0,1):
                    for gv in (0,1):
                        m=((uF>0)==bool(gu))&((vF>0)==bool(gv))
                        if m.sum()>=3: cells[(gu,gv)]=affine_fit(uF[m],vF[m])
                pcell=affine_fit(uF,vF)
                for i,ui in enumerate(uH):
                    key=((ui>0), (vH[i]>0))
                    if key in cells: S_cell+=(vH[i]-(cells[key][0]*ui+cells[key][1]))**2
                    S_pool+=(vH[i]-(pcell[0]*ui+pcell[1]))**2
            S_cell/=len(H); S_pool/=len(H)
            # diagonal operator error on H (per-channel affine, fit on F)
            R_diag=0.0
            for c in range(C):
                co=affine_fit(Z[F,c].astype(np.float64),Zt[F,c].astype(np.float64))
                R_diag+=float(((Z[H,c]*co[0]+co[1]-Zt[H,c])**2).mean())
            full={}
            Zc=Z[F]-Z[F].mean(0); U,s,vt=np.linalg.svd(Zc,full_matrices=False)
            for k in ks:
                P=vt[:k]                                          # fit-subspace PCA directions
                Af=Z[F]@P.T; Ah=Z[H]@P.T
                W=np.linalg.solve(Af.T@Af+1e-3*np.eye(k),Af.T@Zt[F]).T
                full[str(k)]=float(((Ah@W.T-Zt[H])**2).sum(1).mean())/disp
            rec[fam]={'disp':disp,'R_diag':R_diag/disp,'S_cell':S_cell/disp,'S_pool':S_pool/disp,
                      'sharing':(S_pool-S_cell)/disp,'R_full':full,'R_diag_vs_Spool':(R_diag/disp)/(S_pool/disp) if S_pool>0 else None}
        out['sites'][site]=rec
        print(site, {k:{kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items()} for k,v in rec.items()}, flush=True)
    json.dump(out,open(os.path.join(WORK,'results','two_term_heldout_resnet50.json'),'w'),indent=1)
    print('-> results/two_term_heldout_resnet50.json')
if __name__=='__main__': main()
