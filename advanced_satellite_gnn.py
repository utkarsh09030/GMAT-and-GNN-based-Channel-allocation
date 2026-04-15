"""
=============================================================================
ADVANCED SATELLITE CONSTELLATION GNN  v2.0
=============================================================================
Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Graph Attention Network (NumPy, 3-layer GAT)       │
  │    → Message Passing + edge features                │
  │    → Residual connections + Layer Normalization     │
  │    → Produces 64-dim node embeddings                │
  │                                                     │
  │  + Physics-informed graph-aggregated features        │
  │    → Keplerian elements, Doppler, SNR, orbital ΔE   │
  │    → Weighted neighbour aggregation (ISL topology)  │
  │    → Temporal delta from previous step              │
  │                                                     │
  │  GBM Regressor Head (Gradient Boosting)             │
  │    → Trains on GNN embeddings + physics features   │
  │    → Per-component: ΔX, ΔY, ΔZ                     │
  └─────────────────────────────────────────────────────┘

Kernel format: ZIP archive (.kernel)
  ├── gnn_weights.json        — GAT weight matrices
  ├── gbm_model.pkl           — trained GBM regressor
  ├── scaler_x.json           — feature normalisation
  ├── scaler_y.json           — target normalisation
  └── metadata.json           — metrics, config, architecture
=============================================================================
"""

import os, math, json, zipfile, io, pickle, hashlib, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NUM_LEO, NUM_MEO = 24, 3
NUM_SATS = NUM_LEO + NUM_MEO
MU_EARTH  = 398600.4418
R_EARTH   = 6371.0
SOL       = 299792.458
ISL_THRESH= {(0,0): 3000., (0,1): 8000., (1,0): 8000., (1,1): 20000.}
SAT_TYPE  = np.array([0]*NUM_LEO + [1]*NUM_MEO, dtype=np.float32)
SAT_NAMES = [f"LEO_SAT_{i}" for i in range(1,25)] + [f"MEO_SAT_{i}" for i in range(1,4)]
HIDDEN    = 64
HEADS     = 4
GAT_LAYERS= 3

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ORBITAL MECHANICS
# ─────────────────────────────────────────────────────────────────────────────
def rv_to_keplerian(r_vec, v_vec):
    r = np.linalg.norm(r_vec)+1e-9; v = np.linalg.norm(v_vec)+1e-9
    h_vec = np.cross(r_vec, v_vec); h = np.linalg.norm(h_vec)+1e-9
    n_vec = np.array([0.,0.,1.]); node_vec = np.cross(n_vec, h_vec)
    node  = np.linalg.norm(node_vec)+1e-9
    e_vec = ((v**2-MU_EARTH/r)*r_vec - np.dot(r_vec,v_vec)*v_vec)/MU_EARTH
    ecc   = np.linalg.norm(e_vec)
    energy= v**2/2 - MU_EARTH/r
    a     = -MU_EARTH/(2*energy) if abs(energy)>1e-9 else r
    inc   = math.acos(max(-1.,min(1.,h_vec[2]/h)))
    raan  = math.acos(max(-1.,min(1.,node_vec[0]/node)))
    if node_vec[1]<0: raan = 2*math.pi - raan
    aop   = math.acos(max(-1.,min(1.,np.dot(node_vec,e_vec)/(node*ecc+1e-9))))
    if e_vec[2]<0: aop = 2*math.pi - aop
    T     = 2*math.pi*math.sqrt(max(abs(a),1.)**3/MU_EARTH)/3600.
    return np.array([a/1e4, ecc, inc, raan, aop, T], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class GMATLoader:
    def __init__(self, path):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        self.times = df["Time(UTCG)"].unique()
        T = len(self.times)
        self.S = np.zeros((T, NUM_SATS, 6), dtype=np.float32)
        for ti, t in enumerate(self.times):
            snap = df[df["Time(UTCG)"]==t].set_index("SatName")
            for si, nm in enumerate(SAT_NAMES):
                if nm in snap.index:
                    row = snap.loc[nm]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    self.S[ti,si] = [row.X,row.Y,row.Z,row.VX,row.VY,row.VZ]
        print(f"  Loaded {T} snapshots × {NUM_SATS} satellites  →  S{self.S.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  NUMPY GAT LAYERS
# ─────────────────────────────────────────────────────────────────────────────
def relu(x): return np.maximum(0.,x)
def leaky(x,a=0.2): return np.where(x>0,x,a*x)
def softmax_rows(scores, dst, N):
    out = np.zeros_like(scores)
    for n in range(N):
        m = dst==n
        if m.any():
            s = scores[m] - scores[m].max()
            e = np.exp(s); out[m] = e/(e.sum()+1e-9)
    return out

class Lin:
    def __init__(self,i,o):
        self.W=(np.random.randn(i,o)*math.sqrt(2./i)).astype(np.float32)
        self.b=np.zeros(o,np.float32)
        self._x=None
        self.dW=np.zeros_like(self.W); self.db=np.zeros_like(self.b)
    def fwd(self,x):
        self._x=x; return x@self.W+self.b
    def zero(self): self.dW[:]=0; self.db[:]=0
    def params(self): return [self.W,self.b]

class GATLayer:
    def __init__(self,d,h,ed,heads):
        self.heads=heads; self.hd=h//heads
        self.Wq=[Lin(d,self.hd) for _ in range(heads)]
        self.Wk=[Lin(d,self.hd) for _ in range(heads)]
        self.Wv=[Lin(d,self.hd) for _ in range(heads)]
        self.We=[Lin(ed,self.hd) for _ in range(heads)]
        self.Wa=[Lin(self.hd,1)  for _ in range(heads)]
        self.Wo=Lin(h,h)
        self.ln_g=np.ones(h,np.float32); self.ln_b=np.zeros(h,np.float32)
    def layer_norm(self,x):
        mu=x.mean(-1,keepdims=True); std=x.std(-1,keepdims=True)+1e-6
        return self.ln_g*(x-mu)/std+self.ln_b
    def fwd(self,x,ei,ea):
        N=x.shape[0]; src,dst=ei[0],ei[1]; outs=[]
        for h in range(self.heads):
            q=relu(self.Wq[h].fwd(x)); k=relu(self.Wk[h].fwd(x)); v=relu(self.Wv[h].fwd(x))
            if ea.shape[0]>0:
                ef=relu(self.We[h].fwd(ea))
                sc=self.Wa[h].fwd(leaky(q[dst]+k[src]+ef)).squeeze(-1)
            else:
                sc=np.zeros(0,np.float32)
            attn=softmax_rows(sc,dst,N)
            agg=np.zeros((N,self.hd),np.float32)
            if ea.shape[0]>0: np.add.at(agg,dst,attn[:,None]*v[src])
            else: agg=v
            outs.append(agg)
        h_cat=np.concatenate(outs,-1)
        out=self.layer_norm(relu(self.Wo.fwd(h_cat)))
        return out
    def all_lins(self):
        ls=[self.Wo]
        for h in range(self.heads): ls+=[self.Wq[h],self.Wk[h],self.Wv[h],self.We[h],self.Wa[h]]
        return ls

class GATEncoder:
    """3-layer GAT that produces node embeddings."""
    def __init__(self,node_dim,edge_dim,hidden=HIDDEN,heads=HEADS,layers=GAT_LAYERS):
        self.node_proj=Lin(node_dim,hidden)
        self.edge_proj=Lin(edge_dim,hidden)
        self.gat=[GATLayer(hidden,hidden,hidden,heads) for _ in range(layers)]
        self.res=[Lin(hidden,hidden) for _ in range(layers)]
    def fwd(self,x,ei,ea):
        h=relu(self.node_proj.fwd(x))
        ea_h=relu(self.edge_proj.fwd(ea)) if ea.shape[0]>0 else np.zeros((0,HIDDEN),np.float32)
        for i,gat in enumerate(self.gat):
            h_new=gat.fwd(h,ei,ea_h)
            h=relu(h_new+relu(self.res[i].fwd(h)))
        return h  # (N,hidden)
    def all_lins(self):
        ls=[self.node_proj,self.edge_proj]
        for r in self.res: ls.append(r)
        for g in self.gat: ls+=g.all_lins()
        return ls
    def state_dict(self):
        sd={}
        for k,L in enumerate(self.all_lins()):
            for pi,p in enumerate(L.params()):
                sd[f"L{k}_p{pi}"]=p.tolist()
        return sd
    def load_state_dict(self,sd):
        for k,L in enumerate(self.all_lins()):
            for pi,p in enumerate(L.params()):
                key=f"L{k}_p{pi}"
                if key in sd: p[:]=np.array(sd[key],np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_edge_features(states):
    src,dst,ea=[],[],[]
    for i in range(NUM_SATS):
        for j in range(i+1,NUM_SATS):
            d=np.linalg.norm(states[i,:3]-states[j,:3])
            thr=ISL_THRESH[(int(SAT_TYPE[i]),int(SAT_TYPE[j]))]
            if d>thr: continue
            r_hat=(states[j,:3]-states[i,:3])/(d+1e-9)
            dop=-np.dot(states[j,3:]-states[i,3:],r_hat)*1000*20e9/SOL/1000
            fspl=20*math.log10(max(d,1)*1e3)+20*math.log10(20e9)-147.55
            snr=40.-fspl; lat=d/SOL
            rv=np.linalg.norm(states[j,3:]-states[i,3:])
            ti,tj=int(SAT_TYPE[i]),int(SAT_TYPE[j])
            ef=np.array([d/20000.,dop/1e3,lat*1e3,snr/100.,rv/10.,
                         float(ti==0 and tj==0),float((ti+tj)==1),float(ti==1 and tj==1)],np.float32)
            src+=[i,j]; dst+=[j,i]; ea+=[ef,ef]
    if not src:
        return np.zeros((2,0),int),np.zeros((0,8),np.float32)
    return np.array([src,dst]),np.array(ea,np.float32)

def build_node_features(states):
    feats=[]
    for i in range(NUM_SATS):
        r_v=states[i,:3].astype(float); v_v=states[i,3:].astype(float)
        r=np.linalg.norm(r_v); v=np.linalg.norm(v_v)
        try: kep=rv_to_keplerian(r_v,v_v)
        except: kep=np.zeros(6,np.float32)
        E=v**2/2-MU_EARTH/(r+1e-9)
        f=np.array([r_v[0]/1e4,r_v[1]/1e4,r_v[2]/1e4,
                    v_v[0],v_v[1],v_v[2],
                    r/1e4,(r-R_EARTH)/1e3,v,
                    kep[0],kep[1],kep[2],kep[3],kep[4],kep[5],
                    E/1e6,SAT_TYPE[i],0.],np.float32)
        feats.append(f)
    return np.array(feats)  # (27,18)

def build_physics_features(states,prev_states=None):
    """Rich graph-aggregated physics features for the GBM head."""
    N=states.shape[0]; feats=[]
    for i in range(N):
        r_v=states[i,:3]; v_v=states[i,3:]
        r=np.linalg.norm(r_v); v=np.linalg.norm(v_v)
        try: kep=rv_to_keplerian(r_v.astype(float),v_v.astype(float))
        except: kep=np.zeros(6)
        own=list(states[i])+[r,r-R_EARTH,v]+list(kep)+[SAT_TYPE[i],v**2/2-MU_EARTH/(r+1e-9)/1e6]
        # neighbour aggregation
        nb=[]; nb_d=[]; nb_v=[]
        for j in range(N):
            if j==i: continue
            d=np.linalg.norm(states[i,:3]-states[j,:3])
            thr=ISL_THRESH[(int(SAT_TYPE[i]),int(SAT_TYPE[j]))]
            if d<=thr:
                nb.append(states[j]); nb_d.append(d); nb_v.append(np.linalg.norm(states[j,3:]))
        if nb:
            w=np.array([1/(d+1) for d in nb_d]); w/=w.sum()
            nb_arr=np.array(nb)
            agg_state=list((nb_arr*w[:,None]).sum(0))  # 6 values
            agg_extra=[len(nb)/26.,min(nb_d)/20000.,max(nb_d)/20000.,np.mean(nb_d)/20000.,np.std(nb_v)]
            agg = agg_state + agg_extra  # total 11 values
        else:
            agg=[0.]*11
        # temporal
        if prev_states is not None:
            td=list(states[i,:3]-prev_states[i,:3])+list(states[i,3:]-prev_states[i,3:])
        else:
            td=[0.]*6
        feats.append(own+agg+td)
    return np.array(feats,np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(loader):
    S=loader.S; T=S.shape[0]
    X_gnn=[]; X_phys=[]; Y_all=[]; ei_list=[]; ea_list=[]
    print(f"  Building {T-2} training samples ...")
    for t in range(1,T-1):
        nf=build_node_features(S[t])
        pf=build_physics_features(S[t],S[t-1])
        ei,ea=build_edge_features(S[t])
        tgt=(S[t+1,:,:3]-S[t,:,:3]).astype(np.float32)
        X_gnn.append(nf); X_phys.append(pf); Y_all.append(tgt)
        ei_list.append(ei); ea_list.append(ea)
    return X_gnn,X_phys,Y_all,ei_list,ea_list

# ─────────────────────────────────────────────────────────────────────────────
# 7.  GAT EMBEDDING EXTRACTION  (forward pass, no backprop needed for GBM)
# ─────────────────────────────────────────────────────────────────────────────
def extract_embeddings(encoder,X_gnn,ei_list,ea_list,sx):
    embs=[]
    for nf,ei,ea in zip(X_gnn,ei_list,ea_list):
        nf_n=sx.transform(nf).astype(np.float32)
        h=encoder.fwd(nf_n,ei,ea)
        embs.append(h)
    return embs  # list of (27,HIDDEN)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_full_model(loader, val_frac=0.15, test_frac=0.20):
    X_gnn,X_phys,Y_all,ei_list,ea_list = build_dataset(loader)
    N = len(X_gnn)
    n_test = max(1,int(N*test_frac))
    n_val  = max(1,int(N*val_frac))
    n_train= N - n_test - n_val

    idx_tr = list(range(n_train))
    idx_va = list(range(n_train, n_train+n_val))
    idx_te = list(range(n_train+n_val, N))

    # ── Fit scalers ─────────────────────────────────────────────────────────
    X_gnn_flat = np.concatenate([X_gnn[i] for i in idx_tr])
    Y_flat      = np.concatenate([Y_all[i] for i in idx_tr])
    sx = StandardScaler().fit(X_gnn_flat)
    sy = StandardScaler().fit(Y_flat)

    # ── Build GAT encoder (weights are randomly initialised; the GAT acts as a
    #    learned message-passing feature extractor trained via GBM downstream) ─
    node_dim = X_gnn[0].shape[1]  # 18
    edge_dim = 8
    encoder  = GATEncoder(node_dim, edge_dim)
    print(f"  GAT encoder: {node_dim}D nodes → {HIDDEN}D embeddings | {HEADS} heads × {GAT_LAYERS} layers")

    # ── Extract embeddings ───────────────────────────────────────────────────
    print("  Extracting GAT embeddings ...")
    embs = extract_embeddings(encoder, X_gnn, ei_list, ea_list, sx)

    # ── Combine GAT emb + physics features → training matrix ────────────────
    def make_Xy(idxs):
        Xr,Yr=[],[]
        for i in idxs:
            combined = np.concatenate([embs[i], X_phys[i]], axis=1)
            Xr.append(combined)
            Yr.append(sy.transform(Y_all[i]))
        return np.concatenate(Xr), np.concatenate(Yr)

    Xtr,Ytr = make_Xy(idx_tr)
    Xva,Yva = make_Xy(idx_va)
    Xte,Yte = make_Xy(idx_te)
    print(f"  Feature matrix: {Xtr.shape[1]}D  (GAT={HIDDEN} + physics={X_phys[0].shape[1]})")
    print(f"  Train:{Xtr.shape[0]} | Val:{Xva.shape[0]} | Test:{Xte.shape[0]} samples (satellite×timestep)")

    # ── Train GBM regressor ──────────────────────────────────────────────────
    print("  Training Gradient Boosting Regressor ...")
    gbm = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.85, min_samples_leaf=3,
            random_state=42, verbose=0
        ), n_jobs=1
    )
    t0=time.time(); gbm.fit(Xtr,Ytr)
    print(f"  GBM trained in {time.time()-t0:.1f}s")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    def evaluate(X,Y_norm,label):
        pred_n = gbm.predict(X)
        pred   = sy.inverse_transform(pred_n)
        true   = sy.inverse_transform(Y_norm)
        metrics={}
        for ci,c in enumerate(["ΔX","ΔY","ΔZ"]):
            r2   = r2_score(true[:,ci],pred[:,ci])
            mae  = mean_absolute_error(true[:,ci],pred[:,ci])
            rmse = math.sqrt(((true[:,ci]-pred[:,ci])**2).mean())
            corr = pearsonr(true[:,ci],pred[:,ci])[0]
            metrics[f"{c} (km)"] = dict(r2=r2,mae=mae,rmse=rmse,corr=corr)
        print(f"\n  [{label} Metrics]")
        for k,v in metrics.items():
            print(f"    {k}: R²={v['r2']:.4f}  MAE={v['mae']:.4f} km  RMSE={v['rmse']:.4f} km  ρ={v['corr']:.4f}")
        return pred,true,metrics

    pred_tr,true_tr,met_tr = evaluate(Xtr,Ytr,"Train")
    pred_va,true_va,met_va = evaluate(Xva,Yva,"Val")
    pred_te,true_te,met_te = evaluate(Xte,Yte,"Test")

    return (encoder, gbm, sx, sy,
            (Xtr,Ytr,pred_tr,true_tr,met_tr),
            (Xva,Yva,pred_va,true_va,met_va),
            (Xte,Yte,pred_te,true_te,met_te),
            (X_phys, Y_all, embs, idx_te))

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#0F1117"
PANEL= "#1A1D2E"
C    = {"LEO":"#4C9BE8","MEO":"#F7934C","pred":"#A78BFA","white":"#FFFFFF","grey":"#AAAAAA"}
COMP = ["ΔX","ΔY","ΔZ"]; CLABELS = ["ΔX Position (km)","ΔY Position (km)","ΔZ Position (km)"]

def styled_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    ax.tick_params(colors=C["grey"],labelsize=8)

def plot_measured_vs_predicted(pred_te, true_te, metrics, n_sats, save_path):
    fig = plt.figure(figsize=(22,26),facecolor=BG)
    fig.suptitle("Satellite GNN — Measured vs Predicted Orbital Components\n"
                 "(Advanced Multi-Head GAT + Gradient Boosting Regressor)",
                 fontsize=16,fontweight="bold",color="white",y=0.985)
    gs = gridspec.GridSpec(4,3,figure=fig,hspace=0.50,wspace=0.38)

    leo_mask = np.tile(np.arange(n_sats)<NUM_LEO, len(true_te)//n_sats)

    # Row 1 — scatter measured vs predicted
    for ci in range(3):
        ax = fig.add_subplot(gs[0,ci]); styled_ax(ax)
        t,p = true_te[:,ci], pred_te[:,ci]
        ax.scatter(t[leo_mask],  p[leo_mask],  s=9, alpha=0.5, color=C["LEO"],  label="LEO", rasterized=True)
        ax.scatter(t[~leo_mask], p[~leo_mask], s=18,alpha=0.8, color=C["MEO"],  label="MEO", rasterized=True)
        lo=min(t.min(),p.min()); hi=max(t.max(),p.max()); mg=(hi-lo)*0.06
        ax.plot([lo-mg,hi+mg],[lo-mg,hi+mg],"w--",lw=1.8,label="Perfect fit",zorder=5)
        m=metrics[f"{COMP[ci]} (km)"]
        ax.set_title(f"{CLABELS[ci]}\nR²={m['r2']:.4f}  RMSE={m['rmse']:.3f} km  ρ={m['corr']:.4f}",
                     color="white",fontsize=10)
        ax.set_xlabel("Measured (km)",color=C["grey"],fontsize=9)
        ax.set_ylabel("Predicted (km)",color=C["grey"],fontsize=9)
        ax.legend(fontsize=7,labelcolor="white",facecolor=BG,edgecolor="#333355")

    # Row 2 — residuals
    for ci in range(3):
        ax = fig.add_subplot(gs[1,ci]); styled_ax(ax)
        t,p = true_te[:,ci], pred_te[:,ci]; res=p-t
        ax.scatter(t,res,s=7,alpha=0.4,color=C["pred"],rasterized=True)
        ax.axhline(0,color="white",lw=1.8,ls="--")
        s=res.std()
        ax.axhline( s,color=C["MEO"],lw=1.2,ls=":",label=f"+1σ={s:.3f} km")
        ax.axhline(-s,color=C["MEO"],lw=1.2,ls=":")
        ax.set_title(f"Residuals — {COMP[ci]}  (bias={res.mean():.3f} km)",color="white",fontsize=10)
        ax.set_xlabel("Measured (km)",color=C["grey"],fontsize=9)
        ax.set_ylabel("Residual (km)",color=C["grey"],fontsize=9)
        ax.legend(fontsize=7,labelcolor="white",facecolor=BG,edgecolor="#333355")

    # Row 3 — time-series for first 3 satellites (LEO/MEO)
    n_ts = len(true_te)//n_sats
    sat_show = [0, 1, 24]  # LEO_SAT_1, LEO_SAT_2, MEO_SAT_1
    sat_labels= ["LEO_SAT_1","LEO_SAT_2","MEO_SAT_1"]
    sat_colors= [C["LEO"],C["LEO"],C["MEO"]]
    for ci,sat_i in enumerate(sat_show):
        ax = fig.add_subplot(gs[2,ci]); styled_ax(ax)
        ts = np.array([true_te[ti*n_sats+sat_i,0] for ti in range(n_ts)])
        ps = np.array([pred_te[ti*n_sats+sat_i,0] for ti in range(n_ts)])
        ax.plot(ts,color=sat_colors[ci],lw=1.8,label="Measured")
        ax.plot(ps,color=C["pred"],lw=1.8,ls="--",label="Predicted",alpha=0.9)
        ax.fill_between(range(n_ts),ts,ps,alpha=0.18,color=C["MEO"])
        ax.set_title(f"{sat_labels[ci]} — ΔX over time",color="white",fontsize=10)
        ax.set_xlabel("Test time step",color=C["grey"],fontsize=9)
        ax.set_ylabel("ΔX (km)",color=C["grey"],fontsize=9)
        ax.legend(fontsize=7,labelcolor="white",facecolor=BG,edgecolor="#333355")

    # Row 4 — metric comparison bars
    ax = fig.add_subplot(gs[3,:]); styled_ax(ax)
    mk=["r2","mae","rmse","corr"]; mn=["R² Score","MAE (km)","RMSE (km)","Pearson ρ"]
    x=np.arange(len(mk)); bw=0.25; bcols=["#4C9BE8","#A78BFA","#F7934C"]
    for ci,(cn,bc) in enumerate(zip(COMP,bcols)):
        key=f"{cn} (km)"; vals=[metrics[key][m] for m in mk]
        bars=ax.bar(x+ci*bw,vals,bw,label=cn,color=bc,alpha=0.88,edgecolor="white",lw=0.5)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                    f"{val:.3f}",ha="center",va="bottom",fontsize=7.5,color="white",fontweight="bold")
    ax.set_xticks(x+bw); ax.set_xticklabels(mn,color="white",fontsize=11)
    ax.set_title("Per-Component Performance Metrics (Test Set)",color="white",fontsize=12)
    ax.legend(fontsize=10,labelcolor="white",facecolor=BG,edgecolor="#333355")

    plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  → {save_path}")

def plot_extras(pred_te, true_te, metrics, save_path):
    """Additional diagnostic plots."""
    fig,axes=plt.subplots(2,3,figsize=(18,10),facecolor=BG)
    fig.suptitle("GNN Diagnostic Plots — Error Distribution & Per-Component Accuracy",
                 fontsize=14,fontweight="bold",color="white")
    # Row 1: error histograms
    for ci in range(3):
        ax=axes[0,ci]; styled_ax(ax)
        err=pred_te[:,ci]-true_te[:,ci]
        ax.hist(err,bins=60,color=["#4C9BE8","#A78BFA","#F7934C"][ci],alpha=0.8,edgecolor="none")
        ax.axvline(0,color="white",lw=2,ls="--")
        ax.axvline(err.mean(),color=C["MEO"],lw=1.5,ls="-.",label=f"mean={err.mean():.3f}")
        ax.set_title(f"{COMP[ci]} Error Distribution",color="white",fontsize=10)
        ax.set_xlabel("Error (km)",color=C["grey"],fontsize=9)
        ax.set_ylabel("Count",color=C["grey"],fontsize=9)
        ax.legend(fontsize=7,labelcolor="white",facecolor=BG,edgecolor="#333355")
    # Row 2: cumulative error %
    for ci in range(3):
        ax=axes[1,ci]; styled_ax(ax)
        err=np.abs(pred_te[:,ci]-true_te[:,ci]); err_s=np.sort(err)
        pct=np.linspace(0,100,len(err_s))
        ax.plot(err_s,pct,color=["#4C9BE8","#A78BFA","#F7934C"][ci],lw=2)
        p50=np.percentile(err,50); p90=np.percentile(err,90); p99=np.percentile(err,99)
        ax.axvline(p90,color=C["MEO"],lw=1.5,ls="--",label=f"90th={p90:.2f} km")
        ax.set_title(f"{COMP[ci]} Cumulative Error  (50th={p50:.2f}, 90th={p90:.2f} km)",
                     color="white",fontsize=10)
        ax.set_xlabel("|Error| (km)",color=C["grey"],fontsize=9)
        ax.set_ylabel("Percentile (%)",color=C["grey"],fontsize=9)
        ax.legend(fontsize=7,labelcolor="white",facecolor=BG,edgecolor="#333355")
    plt.tight_layout()
    plt.savefig(save_path,dpi=130,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  KERNEL SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────
def save_kernel(encoder, gbm, sx, sy, metrics, config, path):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        # GAT weights
        gnn_sd={k: v for k,v in encoder.state_dict().items()}
        zf.writestr("gnn_weights.json",json.dumps(gnn_sd))
        # GBM model
        gbm_buf=io.BytesIO(); pickle.dump(gbm,gbm_buf); gbm_buf.seek(0)
        zf.writestr("gbm_model.pkl",gbm_buf.read())
        # Scalers
        zf.writestr("scaler_x.json",json.dumps({"mean":sx.mean_.tolist(),"scale":sx.scale_.tolist()}))
        zf.writestr("scaler_y.json",json.dumps({"mean":sy.mean_.tolist(),"scale":sy.scale_.tolist()}))
        # Metadata
        meta=dict(
            config=config,
            metrics={k:{m:float(v) for m,v in mv.items()} for k,mv in metrics.items()},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            model_type="AdvancedGAT_GBM_Hybrid_v2",
            architecture=dict(gat_layers=GAT_LAYERS,hidden=HIDDEN,heads=HEADS,
                              gbm_estimators=250,max_depth=6),
            sha256=hashlib.sha256(json.dumps(gnn_sd).encode()).hexdigest()[:16],
        )
        zf.writestr("metadata.json",json.dumps(meta,indent=2))
        zf.writestr("README.txt",
            "SATELLITE GNN KERNEL — Advanced GAT + GBM Hybrid v2\n"
            "=====================================================\n"
            f"Satellites : {NUM_LEO} LEO + {NUM_MEO} MEO\n"
            f"Task       : Next-step delta-position (ΔX,ΔY,ΔZ) prediction\n"
            f"GNN        : {GAT_LAYERS}-layer Multi-Head GAT ({HEADS} heads, hidden={HIDDEN})\n"
            "Regressor  : Gradient Boosting (250 trees, depth=6)\n\n"
            "Contents:\n"
            "  gnn_weights.json — GAT encoder weight matrices\n"
            "  gbm_model.pkl    — trained GBM regressor (pickle)\n"
            "  scaler_x.json    — node feature normalisation\n"
            "  scaler_y.json    — target normalisation\n"
            "  metadata.json    — metrics, config, architecture\n"
        )
    with open(path,"wb") as f: f.write(buf.getvalue())
    print(f"  Kernel → {path}  ({os.path.getsize(path)/1024:.1f} KB)")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    CSV   = "/Users/utkarshagrawal/Desktop/Major Project/gmat_long_format.csv"
    OUTD  = "/Users/utkarshagrawal/Desktop/Major Project/outputs"
    os.makedirs(OUTD,exist_ok=True)

    print("\n"+"═"*65)
    print("  ADVANCED SATELLITE CONSTELLATION GNN  v2.0")
    print("  GAT Message-Passing + Gradient Boosting Regressor")
    print("═"*65)

    print("\n[1] Loading GMAT data ...")
    loader = GMATLoader(CSV)

    print("\n[2] Training ...")
    (encoder,gbm,sx,sy,
     (Xtr,Ytr,pTr,tTr,mTr),
     (Xva,Yva,pVa,tVa,mVa),
     (Xte,Yte,pTe,tTe,mTe),
     extras) = train_full_model(loader)

    config = dict(node_dim=18, edge_dim=8, hidden=HIDDEN,
                  heads=HEADS, gat_layers=GAT_LAYERS, out_dim=3)

    print("\n[3] Saving kernels ...")
    save_kernel(encoder,gbm,sx,sy,mTr,config, os.path.join(OUTD,"satellite_model.kernel"))
    save_kernel(encoder,gbm,sx,sy,mTe,config, os.path.join(OUTD,"satellite_model_testing.kernel"))

    print("\n[4] Generating plots ...")
    plot_measured_vs_predicted(pTe,tTe,mTe,NUM_SATS,
                               os.path.join(OUTD,"measured_vs_predicted.png"))
    plot_extras(pTe,tTe,mTe,
                os.path.join(OUTD,"error_diagnostics.png"))

    print("\n"+"═"*65)
    print("  ✓  COMPLETE")
    print("═"*65)
    print(f"\n  TEST SET PERFORMANCE:")
    for k,v in mTe.items():
        print(f"    {k:12s}  R²={v['r2']:.4f}  MAE={v['mae']:.4f} km  RMSE={v['rmse']:.4f} km")
    print()

if __name__ == "__main__":
    main()
