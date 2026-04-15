"""
================================================================================
visualization_8charts.py  —  Satellite Constellation GNN  |  8 Clean Charts
================================================================================
CHARTS (one per output file, white background, black text):
  01. Hybrid Satellite Constellation Analysis     — 3D orbits
  02. Inter-Satellite Link Availability           — ISL heatmap
  03. Bandwidth Variation Analysis                — BW heatmap (single chart)
  04. Latency and Distance Relationship           — scatter only
  05. GNN-Based Satellite Position Prediction     — pred vs actual (single axis)
  06. Network Congestion Analysis                 — link-count heatmap only
  07. Channel Allocation Priority (CAP) Evaluation— ranked bar chart only
  08. Overall Performance Analysis                — summary radar / metrics

RUN:
    python3 visualization_8charts.py

OUTPUT: ./chart_outputs/  (8 PNG files)
================================================================================
"""

import os, math, json, zipfile, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.ndimage import uniform_filter1d

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# ▶  EDIT THESE PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "gmat_long_format.csv")
KERNEL_TR = os.path.join(BASE_DIR, "satellite_model.kernel")
KERNEL_TE = os.path.join(BASE_DIR, "satellite_model_testing.kernel")
OUT_DIR   = os.path.join(BASE_DIR, "chart_outputs")

# Fallback to upload paths if local files not found
if not os.path.exists(CSV_PATH):
    CSV_PATH  = "/mnt/user-data/uploads/gmat_long_format.csv"
    KERNEL_TR = "/mnt/user-data/uploads/1775926370537_satellite_model.kernel"
    KERNEL_TE = "/mnt/user-data/uploads/1775926370537_satellite_model_testing.kernel"

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
NUM_LEO, NUM_MEO = 24, 3
NUM_SATS  = NUM_LEO + NUM_MEO          # 27
MU_EARTH  = 398600.4418
R_EARTH   = 6371.0
SOL       = 299792.458                 # km/s
ISL_THRESH = {(0,0): 6000., (0,1): 25000., (1,0): 25000., (1,1): 50000.}
SAT_TYPE  = np.array([0]*NUM_LEO + [1]*NUM_MEO)
SAT_NAMES = [f"LEO_SAT_{i}" for i in range(1, 25)] + \
            [f"MEO_SAT_{i}" for i in range(1,  4)]
SHORT_NAMES = [f"L{i}" for i in range(1,25)] + [f"M{i}" for i in range(1,4)]

# ── White-theme palette ───────────────────────────────────────────────────────
BG      = "white"
PANEL   = "white"
GRID    = "#DDDDDD"
TEXT    = "black"
LEO_C   = "#1E6FD9"   # blue
MEO_C   = "#E05C00"   # orange
PRED_C  = "#8A2BE2"   # purple
TRUE_C  = "#1A7A4A"   # green
ERR_C   = "#C0392B"   # red
GREY    = "#555555"
GOOD    = "#1A7A4A"
WARN    = "#B8860B"
BAD     = "#C0392B"

BW_CMAP   = LinearSegmentedColormap.from_list("bw",  ["#f0f8ff","#4A90D9","#1E6FD9","#1A7A4A","#B8860B","#C0392B"])
LINK_CMAP = LinearSegmentedColormap.from_list("lk",  ["#f0f0f0","#6BAED6","#2171B5","#6A3D9A","#E05C00"])

# ── Apply white rcParams globally ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GREY,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "axes.titlesize":    13,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "legend.facecolor":  "white",
    "legend.edgecolor":  GREY,
    "legend.labelcolor": TEXT,
})

def _sax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GREY)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(True, color=GRID, alpha=0.6, lw=0.6)

def _sax3(ax):
    ax.set_facecolor(BG)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {name}")
    return path

def _cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=TEXT, labelsize=8)
    cb.set_label(label, color=TEXT, fontsize=9)
    return cb

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_states(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    times = df["Time(UTCG)"].unique()
    T = len(times)
    S = np.zeros((T, NUM_SATS, 6), np.float32)
    for ti, t in enumerate(times):
        snap = df[df["Time(UTCG)"] == t].set_index("SatName")
        for si, nm in enumerate(SAT_NAMES):
            if nm in snap.index:
                row = snap.loc[nm]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                S[ti, si] = [row.X, row.Y, row.Z, row.VX, row.VY, row.VZ]
    print(f"  States loaded: {T} × {NUM_SATS} × 6")
    return S, times

def load_kernel(path):
    with zipfile.ZipFile(path, "r") as z:
        meta = json.loads(z.read("metadata.json"))
        sx_d = json.loads(z.read("scaler_x.json"))
        sy_d = json.loads(z.read("scaler_y.json"))
        gbm  = pickle.loads(z.read("gbm_model.pkl"))
    sx = StandardScaler(); sx.mean_ = np.array(sx_d["mean"]); sx.scale_ = np.array(sx_d["scale"])
    sy = StandardScaler(); sy.mean_ = np.array(sy_d["mean"]); sy.scale_ = np.array(sy_d["scale"])
    return gbm, sx, sy, meta

# ══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_isl_metrics(S):
    T, N = S.shape[:2]
    link_counts = np.zeros((T, N), np.float32)
    snr_rel     = np.zeros((T, N), np.float32)
    bw_est      = np.zeros((T, N), np.float32)
    latency_avg = np.zeros((T, N), np.float32)
    conn_matrix = np.zeros((T, N, N), np.float32)

    for ti in range(T):
        for i in range(N):
            nb_snr, nb_lat, nb_bw = [], [], []
            for j in range(N):
                if i == j: continue
                d   = np.linalg.norm(S[ti, i, :3] - S[ti, j, :3])
                thr = ISL_THRESH[(int(SAT_TYPE[i]), int(SAT_TYPE[j]))]
                if d <= thr:
                    link_counts[ti, i] += 1
                    conn_matrix[ti, i, j] = 1.
                    lat = d / SOL * 1000
                    snr = 20 * math.log10(max(thr / d, 1.))
                    bw  = 500 * math.log2(1 + 10 ** (snr / 10.))
                    nb_snr.append(snr); nb_lat.append(lat); nb_bw.append(bw)
            if nb_snr:
                snr_rel[ti, i]    = float(np.mean(nb_snr))
                latency_avg[ti, i]= float(np.mean(nb_lat))
                bw_est[ti, i]     = float(np.sum(nb_bw))
    return link_counts, snr_rel, bw_est, latency_avg, conn_matrix

def _rv_to_kep6(r_v, v_v):
    r=np.linalg.norm(r_v)+1e-9; v=np.linalg.norm(v_v)+1e-9
    h_v=np.cross(r_v,v_v); h=np.linalg.norm(h_v)+1e-9
    n_v=np.cross(np.array([0.,0.,1.]),h_v); node=np.linalg.norm(n_v)+1e-9
    e_v=((v**2-MU_EARTH/r)*r_v-np.dot(r_v,v_v)*v_v)/MU_EARTH
    ecc=np.linalg.norm(e_v)
    energy=v**2/2-MU_EARTH/r
    a=-MU_EARTH/(2*energy) if abs(energy)>1e-9 else r
    inc=math.acos(max(-1.,min(1.,h_v[2]/h)))
    raan=math.acos(max(-1.,min(1.,n_v[0]/node)))
    if n_v[1]<0: raan=2*math.pi-raan
    aop=0.
    if ecc>1e-9:
        aop=math.acos(max(-1.,min(1.,np.dot(n_v,e_v)/(node*ecc))))
        if e_v[2]<0: aop=2*math.pi-aop
    T_h=2*math.pi*math.sqrt(max(abs(a),1.)**3/MU_EARTH)/3600.
    return np.array([a/1e4, ecc, inc, raan, aop, T_h], np.float32)

def _build_phys_row(states, i, prev_states=None):
    r_v=states[i,:3].astype(float); v_v=states[i,3:].astype(float)
    r=np.linalg.norm(r_v); v=np.linalg.norm(v_v)
    try: kep=_rv_to_kep6(r_v,v_v)
    except: kep=np.zeros(6,np.float32)
    E=v**2/2-MU_EARTH/(r+1e-9)
    own=list(states[i])+[r,r-R_EARTH,v]+list(kep)+[SAT_TYPE[i],E/1e6]
    nb=[]; nb_d=[]; nb_v_=[]
    for j in range(len(states)):
        if j==i: continue
        d=np.linalg.norm(states[i,:3]-states[j,:3])
        thr=ISL_THRESH[(int(SAT_TYPE[i]),int(SAT_TYPE[j]))]
        if d<=thr:
            nb.append(states[j]); nb_d.append(d); nb_v_.append(np.linalg.norm(states[j,3:]))
    if nb:
        w=np.array([1/(d+1) for d in nb_d]); w/=w.sum()
        agg=list((np.array(nb)*w[:,None]).sum(0))
        agg+=[len(nb)/26.,min(nb_d)/20000.,max(nb_d)/20000.,np.mean(nb_d)/20000.,np.std(nb_v_)]
    else:
        agg=[0.]*11
    td=list(states[i,:3]-prev_states[i,:3])+list(states[i,3:]-prev_states[i,3:]) \
        if prev_states is not None else [0.]*6
    return own+agg+td

def run_gnn_predictions(gbm, sx, sy, S):
    T = S.shape[0]; all_pred, all_true = [], []
    for t in range(1, T-1):
        phys = np.array([_build_phys_row(S[t], i, S[t-1]) for i in range(NUM_SATS)], np.float32)
        np.random.seed(0)
        H = np.random.randn(NUM_SATS, 64).astype(np.float32) * 0.1
        combined = np.concatenate([H, phys], axis=1)
        pred_n = gbm.predict(combined)
        pred   = sy.inverse_transform(pred_n)
        true   = (S[t+1,:,:3] - S[t,:,:3]).astype(np.float32)
        all_pred.append(pred); all_true.append(true)
    return np.array(all_pred), np.array(all_true)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 01 — HYBRID SATELLITE CONSTELLATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def chart_01_constellation(S):
    """3D orbital paths — one clean plot, white background."""
    fig = plt.figure(figsize=(12, 10), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    _sax3(ax)
    ax.set_facecolor(BG)

    # Earth sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    ax.plot_surface(R_EARTH*np.cos(u)*np.sin(v),
                    R_EARTH*np.sin(u)*np.sin(v),
                    R_EARTH*np.cos(v),
                    color="#AECDE8", alpha=0.40, linewidth=0, zorder=0)

    for si in range(NUM_SATS):
        xs, ys, zs = S[:,si,0], S[:,si,1], S[:,si,2]
        c  = LEO_C if SAT_TYPE[si]==0 else MEO_C
        lw = 0.7   if SAT_TYPE[si]==0 else 1.8
        a  = 0.55  if SAT_TYPE[si]==0 else 0.90
        ax.plot(xs, ys, zs, color=c, lw=lw, alpha=a)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                   s=35 if SAT_TYPE[si]==0 else 100,
                   color=c, edgecolors="black", linewidths=0.5, zorder=5)

    ax.set_xlabel("X (km)", labelpad=6, color=TEXT)
    ax.set_ylabel("Y (km)", labelpad=6, color=TEXT)
    ax.set_zlabel("Z (km)", labelpad=6, color=TEXT)
    ax.set_title(
        "Hybrid Satellite Constellation Analysis\n"
        "LEO (~550 km orbit)  |  MEO (~19,800 km orbit)",
        color=TEXT, fontsize=13, fontweight="bold", pad=14)
    leo_p = mpatches.Patch(color=LEO_C, label=f"LEO  ({NUM_LEO} satellites, ~550 km)")
    meo_p = mpatches.Patch(color=MEO_C, label=f"MEO  ({NUM_MEO} satellites, ~19,800 km)")
    ax.legend(handles=[leo_p, meo_p], loc="upper left",
              facecolor="white", edgecolor=GREY)
    return _save(fig, "chart_01_constellation_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 02 — INTER-SATELLITE LINK AVAILABILITY
# ══════════════════════════════════════════════════════════════════════════════
def chart_02_isl_availability(conn_matrix):
    """27×27 ISL availability heatmap — single chart."""
    frac = conn_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    _sax(ax)

    im = ax.imshow(frac, cmap=LINK_CMAP, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    _cbar(fig, ax, im, "Fraction of time ISL is active  (0 = never, 1 = always)")

    ticks = np.arange(NUM_SATS)
    ax.set_xticks(ticks); ax.set_xticklabels(SHORT_NAMES, fontsize=7, rotation=90)
    ax.set_yticks(ticks); ax.set_yticklabels(SHORT_NAMES, fontsize=7)

    # Annotate high-value cells
    for i in range(NUM_SATS):
        for j in range(NUM_SATS):
            if frac[i,j] > 0.6:
                ax.text(j, i, f"{frac[i,j]:.2f}", ha="center", va="center",
                        fontsize=4.5, color="white", fontweight="bold")

    # Block separators
    ax.axhline(NUM_LEO-0.5, color=MEO_C, lw=2,   ls="--", alpha=0.8)
    ax.axvline(NUM_LEO-0.5, color=MEO_C, lw=2,   ls="--", alpha=0.8)
    ax.text(NUM_LEO/2-0.5, -1.8, "LEO Block", ha="center",
            color=LEO_C, fontsize=9, fontweight="bold")
    ax.text(NUM_LEO+0.8,   -1.8, "MEO",        ha="center",
            color=MEO_C, fontsize=9, fontweight="bold")

    ax.set_title("Inter-Satellite Link Availability\n"
                 "Fraction of simulation time each satellite pair maintains an active ISL",
                 color=TEXT, fontsize=13, fontweight="bold")
    ax.set_xlabel("Satellite", color=TEXT)
    ax.set_ylabel("Satellite", color=TEXT)
    return _save(fig, "chart_02_isl_availability.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 03 — BANDWIDTH VARIATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def chart_03_bandwidth_variation(bw_est, times):
    """Single bandwidth heatmap (satellite × time), no sub-panels."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    _sax(ax)

    im = ax.imshow(bw_est.T, aspect="auto", cmap=BW_CMAP,
                   interpolation="nearest", origin="lower")
    _cbar(fig, ax, im, "Estimated ISL Bandwidth (Mbps)")

    ax.set_yticks(np.arange(NUM_SATS))
    ax.set_yticklabels(SHORT_NAMES, fontsize=7)
    ax.axhline(NUM_LEO-0.5, color=MEO_C, lw=1.8, ls="--", alpha=0.8,
               label="LEO / MEO boundary")
    ax.legend(loc="upper right", facecolor="white", edgecolor=GREY)

    ax.set_xlabel("Time Step (minutes)", color=TEXT)
    ax.set_ylabel("Satellite",           color=TEXT)
    ax.set_title("Bandwidth Variation Analysis\n"
                 "Available ISL bandwidth per satellite over the full simulation period",
                 color=TEXT, fontsize=13, fontweight="bold")
    return _save(fig, "chart_03_bandwidth_variation.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 04 — LATENCY AND DISTANCE RELATIONSHIP
# ══════════════════════════════════════════════════════════════════════════════
def chart_04_latency_distance(S):
    """Scatter: ISL distance vs one-way latency — single plot."""
    dists_ll, dists_lm, lats_ll, lats_lm = [], [], [], []
    for ti in range(0, S.shape[0], 3):
        for i in range(NUM_SATS):
            for j in range(i+1, NUM_SATS):
                d   = np.linalg.norm(S[ti,i,:3] - S[ti,j,:3])
                ti_ = int(SAT_TYPE[i]); tj_ = int(SAT_TYPE[j])
                thr = ISL_THRESH[(ti_, tj_)]
                if d <= thr:
                    lat = d / SOL * 1000
                    if ti_==0 and tj_==0:
                        dists_ll.append(d); lats_ll.append(lat)
                    else:
                        dists_lm.append(d); lats_lm.append(lat)

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    _sax(ax)

    if dists_ll:
        ax.scatter(dists_ll, lats_ll, s=8, alpha=0.35, color=LEO_C,
                   label=f"LEO–LEO  (n={len(dists_ll):,})", rasterized=True)
    if dists_lm:
        ax.scatter(dists_lm, lats_lm, s=12, alpha=0.55, color=MEO_C,
                   label=f"LEO–MEO  (n={len(dists_lm):,})", rasterized=True)

    # Add median annotation lines
    if lats_ll:
        med_ll = np.median(lats_ll)
        ax.axhline(med_ll, color=LEO_C, lw=1.5, ls="--",
                   label=f"LEO–LEO median: {med_ll:.1f} ms")
    if lats_lm:
        med_lm = np.median(lats_lm)
        ax.axhline(med_lm, color=MEO_C, lw=1.5, ls="--",
                   label=f"LEO–MEO median: {med_lm:.1f} ms")

    ax.set_xlabel("ISL Distance (km)",       color=TEXT)
    ax.set_ylabel("One-way Latency (ms)",    color=TEXT)
    ax.set_title("Latency and Distance Relationship\n"
                 "Each point = one active ISL event   |   Latency = distance / speed-of-light",
                 color=TEXT, fontsize=13, fontweight="bold")
    ax.legend(facecolor="white", edgecolor=GREY)
    ax.text(0.02, 0.97, "← Best allocation zone (low latency)",
            transform=ax.transAxes, color=GOOD, fontsize=9, va="top", fontweight="bold")
    return _save(fig, "chart_04_latency_distance.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 05 — GNN-BASED SATELLITE POSITION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def chart_05_gnn_prediction(all_pred, all_true):
    """
    Predicted vs actual position change — combined single scatter using
    all three axes (ΔX, ΔY, ΔZ) pooled together for a single clear chart.
    """
    P = all_pred.reshape(-1, 3)
    T_ = all_true.reshape(-1, 3)
    leo_mask = np.tile(SAT_TYPE == 0, all_pred.shape[0])

    # Pool all components into one 1-D comparison
    p_all = P.ravel()
    t_all = T_.ravel()
    leo_mask_3 = np.repeat(leo_mask, 3)

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
    _sax(ax)

    ax.scatter(t_all[leo_mask_3],  p_all[leo_mask_3],
               s=4, alpha=0.25, color=LEO_C, label="LEO", rasterized=True)
    ax.scatter(t_all[~leo_mask_3], p_all[~leo_mask_3],
               s=14, alpha=0.7,  color=MEO_C, label="MEO", rasterized=True)

    lo = min(t_all.min(), p_all.min()); hi = max(t_all.max(), p_all.max())
    mg = (hi - lo) * 0.06
    ax.plot([lo-mg, hi+mg], [lo-mg, hi+mg], "k--", lw=2,
            label="Perfect prediction (diagonal)", zorder=5)

    r2   = r2_score(t_all, p_all)
    rmse = math.sqrt(((t_all - p_all)**2).mean())
    corr = np.corrcoef(t_all, p_all)[0, 1]

    # Confidence band
    xs = np.linspace(lo-mg, hi+mg, 200)
    ax.fill_between(xs, xs-rmse, xs+rmse, alpha=0.10, color="blue",
                    label=f"±1 RMSE band ({rmse:.1f} km)")

    ax.set_xlabel("Measured Position Change (km)",  color=TEXT)
    ax.set_ylabel("Predicted Position Change (km)", color=TEXT)
    color_r2 = GOOD if r2 > 0.95 else (WARN if r2 > 0.80 else BAD)
    ax.set_title(
        f"GNN-Based Satellite Position Prediction\n"
        f"R² = {r2:.4f}   |   RMSE = {rmse:.2f} km   |   ρ = {corr:.4f}",
        color=TEXT, fontsize=13, fontweight="bold")
    ax.legend(facecolor="white", edgecolor=GREY)

    # Annotate R² on the plot
    ax.text(0.04, 0.93,
            f"R² = {r2:.4f}\nRMSE = {rmse:.1f} km\nρ = {corr:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color_r2, linewidth=2),
            color=color_r2, fontweight="bold")
    return _save(fig, "chart_05_gnn_prediction.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 06 — NETWORK CONGESTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def chart_06_congestion(link_counts):
    """Single heatmap: active ISL links per satellite per timestep."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    _sax(ax)

    im = ax.imshow(link_counts.T, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", origin="lower", vmin=0)
    _cbar(fig, ax, im, "Active ISL Count per Satellite")

    ax.set_yticks(np.arange(NUM_SATS))
    ax.set_yticklabels(SHORT_NAMES, fontsize=7)
    ax.axhline(NUM_LEO-0.5, color=MEO_C, lw=1.8, ls="--", alpha=0.8,
               label="LEO / MEO boundary")
    ax.legend(loc="upper right", facecolor="white", edgecolor=GREY)

    ax.set_xlabel("Time Step",  color=TEXT)
    ax.set_ylabel("Satellite",  color=TEXT)
    ax.set_title("Network Congestion Analysis\n"
                 "Hot (red) = network hub with many links   |   Cold (yellow) = isolated satellite",
                 color=TEXT, fontsize=13, fontweight="bold")

    # Annotate MEO row labels
    for mi in range(NUM_MEO):
        ax.text(link_counts.shape[0]+1, NUM_LEO+mi,
                SHORT_NAMES[NUM_LEO+mi], fontsize=7, va="center",
                color=MEO_C, fontweight="bold")
    return _save(fig, "chart_06_congestion_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 07 — CHANNEL ALLOCATION PRIORITY (CAP) EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def chart_07_cap_score(bw_est, link_counts, all_pred, all_true):
    """
    Single horizontal bar chart: all 27 satellites ranked by CAP score.
    CAP = 0.40×BW + 0.35×Connectivity + 0.25×GNN accuracy
    """
    avg_bw   = bw_est.mean(axis=0)
    avg_lc   = link_counts.mean(axis=0)
    per_sat_rmse = np.sqrt(((all_pred - all_true)**2).sum(axis=2)).mean(axis=0)

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    bw_n  = norm01(avg_bw)
    lc_n  = norm01(avg_lc)
    acc_n = 1 - norm01(per_sat_rmse)

    cap   = 0.40*bw_n + 0.35*lc_n + 0.25*acc_n
    order = np.argsort(cap)   # ascending for barh (lowest at bottom)

    fig, ax = plt.subplots(figsize=(10, 11), facecolor=BG)
    _sax(ax)

    bar_colors = [LEO_C if SAT_TYPE[i]==0 else MEO_C for i in order]
    bars = ax.barh(range(NUM_SATS), cap[order], color=bar_colors,
                   alpha=0.85, edgecolor="white", linewidth=0.5)

    # Score labels
    for bi, (bar, idx) in enumerate(zip(bars, order)):
        score = cap[idx]
        lbl_color = GOOD if score>0.7 else (WARN if score>0.4 else BAD)
        ax.text(score+0.012, bi, f"{score:.3f}",
                va="center", fontsize=8, color=lbl_color, fontweight="bold")

    ax.set_yticks(range(NUM_SATS))
    ax.set_yticklabels([SHORT_NAMES[i] for i in order], fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("CAP Score  (0 = lowest priority, 1 = highest)", color=TEXT)
    ax.axvline(0.70, color=GOOD, lw=1.8, ls="--", alpha=0.8, label="Primary relay  (>0.70)")
    ax.axvline(0.40, color=WARN, lw=1.8, ls="--", alpha=0.8, label="Secondary relay (>0.40)")
    ax.legend(facecolor="white", edgecolor=GREY)

    leo_p = mpatches.Patch(color=LEO_C, label="LEO satellite")
    meo_p = mpatches.Patch(color=MEO_C, label="MEO satellite")
    ax.legend(handles=[leo_p, meo_p] +
              [plt.Line2D([0],[0], color=GOOD, lw=2, ls="--", label="Primary relay (>0.70)"),
               plt.Line2D([0],[0], color=WARN, lw=2, ls="--", label="Secondary relay (>0.40)")],
              facecolor="white", edgecolor=GREY)

    ax.set_title(
        "Channel Allocation Priority (CAP) Evaluation\n"
        "Score = BW capacity (40%) + Connectivity (35%) + GNN accuracy (25%)",
        color=TEXT, fontsize=13, fontweight="bold")
    return _save(fig, "chart_07_cap_evaluation.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 08 — OVERALL PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def chart_08_overall_performance(bw_est, link_counts, all_pred, all_true, conn_matrix):
    """
    Radar / spider chart showing 5 key performance dimensions
    aggregated across the full simulation — one clean figure.
    """
    # ── Compute 5 KPIs (0–1 normalised) ─────────────────────────────────────
    # 1. Mean ISL Availability (fraction of pairs ever connected)
    frac = conn_matrix.mean(axis=0)
    mask = np.triu(np.ones_like(frac, dtype=bool), k=1)
    isl_avail = frac[mask].mean()

    # 2. Bandwidth Utilisation (mean occupied vs theoretical max)
    bw_util = (bw_est > 0).mean()

    # 3. LEO BW Capacity (normalised mean LEO bandwidth)
    leo_bw_mean = bw_est[:, :NUM_LEO][bw_est[:, :NUM_LEO] > 0].mean()
    leo_bw_norm = min(leo_bw_mean / 12000., 1.0)   # 12 Gbps reference

    # 4. GNN Prediction Accuracy (1 − normalised RMSE)
    rmse_all = math.sqrt(((all_pred - all_true)**2).mean())
    max_range = np.abs(all_true).max()
    gnn_acc = max(0., 1. - rmse_all / (max_range + 1e-9))

    # 5. Network Resilience (fraction of timesteps with >50 total links)
    total_links = link_counts.sum(axis=1) / 2
    resilience  = (total_links > 50).mean()

    labels   = ["ISL\nAvailability", "Bandwidth\nUtilisation",
                "LEO BW\nCapacity", "GNN\nAccuracy", "Network\nResilience"]
    values   = [isl_avail, bw_util, leo_bw_norm, gnn_acc, resilience]
    N        = len(labels)
    angles   = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values_c = values + values[:1]
    angles_c = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True}, facecolor=BG)
    ax.set_facecolor(BG)
    ax.spines["polar"].set_color(GREY)

    # Draw background rings
    for r_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles_c, [r_val]*(N+1), color=GRID, lw=0.8, ls=":")
        ax.text(np.pi/2, r_val+0.03, f"{r_val:.1f}", ha="center",
                fontsize=7, color=GREY)

    # Fill area
    ax.fill(angles_c, values_c, color=LEO_C, alpha=0.25)
    ax.plot(angles_c, values_c, color=LEO_C, lw=2.5, marker="o",
            markersize=8, markerfacecolor=MEO_C, markeredgecolor="white",
            markeredgewidth=1.5)

    # Labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, color=TEXT, fontweight="bold")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.1)

    # Value annotations
    for ang, val, lbl in zip(angles, values, labels):
        ax.text(ang, val+0.08, f"{val:.2f}", ha="center", va="center",
                fontsize=10, color=MEO_C, fontweight="bold")

    ax.set_title("Overall Performance Analysis\n"
                 "Five key metrics aggregated across the full simulation",
                 color=TEXT, fontsize=13, fontweight="bold", pad=26)

    # Summary table below
    summary_lines = [
        f"ISL Availability  : {isl_avail*100:.1f}%  (fraction of time pairs are connected)",
        f"BW Utilisation    : {bw_util*100:.1f}%  (fraction of sat×time slots with active BW)",
        f"LEO BW Capacity   : {leo_bw_norm*100:.1f}%  (normalised to 12 Gbps reference)",
        f"GNN Accuracy      : {gnn_acc*100:.1f}%  (1 − RMSE/range)",
        f"Network Resilience: {resilience*100:.1f}%  (time steps with >50 total ISLs)",
    ]
    fig.text(0.5, 0.01, "\n".join(summary_lines),
             ha="center", va="bottom", fontsize=8.5, color=TEXT,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor=GREY, linewidth=1))

    return _save(fig, "chart_08_overall_performance.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*65)
    print("  SATELLITE GNN — 8 CLEAN SINGLE-PANEL CHARTS")
    print("  White background  |  Black text  |  Output →", OUT_DIR)
    print("═"*65)

    print("\n[1] Loading GMAT states ...")
    S, times = load_states(CSV_PATH)

    print("  Loading kernels ...")
    gbm_tr, sx_tr, sy_tr, meta_tr = load_kernel(KERNEL_TR)
    gbm_te, sx_te, sy_te, meta_te = load_kernel(KERNEL_TE)

    print("\n[2] Computing ISL metrics ...")
    link_counts, snr_rel, bw_est, latency_avg, conn_matrix = compute_isl_metrics(S)

    print("\n[3] Running GNN predictions ...")
    all_pred, all_true = run_gnn_predictions(gbm_te, sx_te, sy_te, S)

    print("\n[4] Generating 8 charts ...")
    chart_01_constellation(S)
    chart_02_isl_availability(conn_matrix)
    chart_03_bandwidth_variation(bw_est, times)
    chart_04_latency_distance(S)
    chart_05_gnn_prediction(all_pred, all_true)
    chart_06_congestion(link_counts)
    chart_07_cap_score(bw_est, link_counts, all_pred, all_true)
    chart_08_overall_performance(bw_est, link_counts, all_pred, all_true, conn_matrix)

    print("\n" + "═"*65)
    print("  ✓  ALL 8 CHARTS SAVED TO:", OUT_DIR)
    print("═"*65)
    print("""
CHART SUMMARY
─────────────────────────────────────────────────────────────
 01 · Constellation Analysis   → 3-D LEO+MEO orbit paths
 02 · ISL Availability         → 27×27 heatmap, fraction connected
 03 · Bandwidth Variation      → BW heatmap (satellite × time)
 04 · Latency & Distance       → Scatter, distance vs latency
 05 · GNN Prediction           → Pooled ΔPos predicted vs actual
 06 · Congestion Analysis      → ISL count heatmap (sat × time)
 07 · CAP Evaluation           → Ranked bar chart, priority score
 08 · Overall Performance      → Radar chart, 5 KPI dimensions
─────────────────────────────────────────────────────────────
""")

if __name__ == "__main__":
    main()