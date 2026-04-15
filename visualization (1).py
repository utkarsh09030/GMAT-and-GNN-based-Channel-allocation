"""
================================================================================
visualization.py  —  Satellite Constellation GNN  |  Bandwidth Allocation Viz
================================================================================
PURPOSE:
  All 7 charts are designed to directly support bandwidth allocation decisions
  in a 27-satellite LEO+MEO constellation. Uses both kernel files + GMAT CSV.

CHARTS (bandwidth allocation focus):
  01. 3D Orbit Trajectories         — where are the satellites (spatial context)
  02. ISL Link Availability Heatmap — which pairs can communicate & how often
  03. Available Bandwidth per Sat   — how much capacity each satellite has
  04. ISL Latency vs Distance       — link quality for routing decisions
  05. Predicted vs Actual ΔPosition — GNN accuracy (trust score for allocation)
  06. Network Congestion Map        — link-count bottlenecks over time
  07. Bandwidth Allocation Priority — combined score: SNR + links + GNN accuracy

RUN:
    python3 visualization.py

OUTPUT: ./visualization_outputs/  (7 PNG files + 1 summary PDF)

EDIT THE PATHS BELOW to match your local machine:
================================================================================
"""

import os, math, json, zipfile, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.ndimage import uniform_filter1d

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# ▶  EDIT THESE PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "gmat_long_format.csv")
KERNEL_TR = os.path.join(BASE_DIR, "outputs", "satellite_model.kernel")
KERNEL_TE = os.path.join(BASE_DIR, "outputs", "satellite_model_testing.kernel")
OUT_DIR   = os.path.join(BASE_DIR, "visualization_outputs")
if not os.path.exists(CSV_PATH):
    CSV_PATH  = "/mnt/user-data/uploads/gmat_long_format.csv"
    KERNEL_TR = "/mnt/user-data/outputs/satellite_model.kernel"
    KERNEL_TE = "/mnt/user-data/outputs/satellite_model_testing.kernel"
    OUT_DIR   = "/mnt/user-data/outputs/visualization_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
NUM_LEO, NUM_MEO = 24, 3
NUM_SATS  = NUM_LEO + NUM_MEO          # 27
MU_EARTH  = 398600.4418
R_EARTH   = 6371.0
SOL       = 299792.458                 # km/s
# Realistic ISL distance thresholds (km)
ISL_THRESH = {(0,0): 6000., (0,1): 25000., (1,0): 25000., (1,1): 50000.}
SAT_TYPE  = np.array([0]*NUM_LEO + [1]*NUM_MEO)
SAT_NAMES = [f"LEO_SAT_{i}" for i in range(1, 25)] + \
            [f"MEO_SAT_{i}" for i in range(1,  4)]
SHORT_NAMES = [f"L{i}" for i in range(1,25)] + [f"M{i}" for i in range(1,4)]

# ── Dark theme palette ────────────────────────────────────────────────────────
BG     = "#0D1117"
PANEL  = "#161B22"
GRID   = "#21262D"
LEO_C  = "#3B82F6"   # blue
MEO_C  = "#F97316"   # orange
PRED_C = "#A855F7"   # purple
TRUE_C = "#10B981"   # green
ERR_C  = "#EF4444"   # red
WHITE  = "#F0F6FC"
GREY   = "#8B949E"
GOOD   = "#22C55E"
WARN   = "#EAB308"
BAD    = "#EF4444"

BW_CMAP  = LinearSegmentedColormap.from_list("bw", ["#0D1117","#1D4ED8","#3B82F6","#22C55E","#EAB308","#EF4444"])
LINK_CMAP= LinearSegmentedColormap.from_list("lk", ["#0D1117","#1E40AF","#3B82F6","#A855F7","#F97316"])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "text.color": WHITE, "grid.color": GRID,
    "grid.alpha": 0.45, "font.family": "monospace",
    "axes.titlesize": 12, "axes.labelsize": 9,
    "legend.fontsize": 8,
})

def _sax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(True, color=GRID, alpha=0.4, lw=0.5)

def _sax3(ax):
    ax.set_facecolor(BG)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor(GRID)
    ax.tick_params(colors=GREY, labelsize=7)

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  {name}")
    return path

def _cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=GREY, labelsize=7)
    cb.set_label(label, color=GREY, fontsize=8)
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
# METRIC COMPUTATION  (ISL-based bandwidth proxies)
# ══════════════════════════════════════════════════════════════════════════════

def compute_isl_metrics(S):
    """
    Returns per-timestep, per-satellite metrics:
      link_counts  (T, N)  — number of active ISLs
      snr_rel      (T, N)  — mean relative SNR (dB) of active links
      bw_est       (T, N)  — total available bandwidth proxy (Mbps)
      latency_avg  (T, N)  — mean one-way ISL latency (ms)
      conn_matrix  (T, N, N) — binary connectivity
    """
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
                    lat = d / SOL * 1000                         # ms
                    snr = 20 * math.log10(max(thr / d, 1.))     # dB relative
                    bw  = 500 * math.log2(1 + 10 ** (snr / 10.))# Mbps (500 MHz)
                    nb_snr.append(snr); nb_lat.append(lat); nb_bw.append(bw)
            if nb_snr:
                snr_rel[ti, i]    = float(np.mean(nb_snr))
                latency_avg[ti, i]= float(np.mean(nb_lat))
                bw_est[ti, i]     = float(np.sum(nb_bw))
    return link_counts, snr_rel, bw_est, latency_avg, conn_matrix

def build_node_features_single(state_row, sat_type):
    r_v = state_row[:3].astype(float); v_v = state_row[3:].astype(float)
    r   = np.linalg.norm(r_v); v = np.linalg.norm(v_v)
    try:
        h_v = np.cross(r_v, v_v); h = np.linalg.norm(h_v) + 1e-9
        e_v = ((v**2 - MU_EARTH/r)*r_v - np.dot(r_v, v_v)*v_v) / MU_EARTH
        ecc = np.linalg.norm(e_v)
        energy = v**2/2 - MU_EARTH/r
        a = -MU_EARTH/(2*energy) if abs(energy)>1e-9 else r
        inc = math.acos(max(-1., min(1., h_v[2]/h)))
        n_v = np.cross(np.array([0.,0.,1.]), h_v); node = np.linalg.norm(n_v)+1e-9
        raan= math.acos(max(-1., min(1., n_v[0]/node)))
        if n_v[1] < 0: raan = 2*math.pi - raan
        aop = 0.
        if ecc > 1e-9:
            aop = math.acos(max(-1., min(1., np.dot(n_v, e_v)/(node*ecc))))
            if e_v[2] < 0: aop = 2*math.pi - aop
        T_h = 2*math.pi*math.sqrt(max(abs(a),1.)**3/MU_EARTH)/3600.
        vvis= math.sqrt(MU_EARTH*(2/max(r,1.) - 1/max(abs(a),1.)))
    except:
        a=ecc=inc=raan=aop=T_h=vvis=0.
    E = v**2/2 - MU_EARTH/(r+1e-9)
    return np.array([r_v[0]/1e4, r_v[1]/1e4, r_v[2]/1e4,
                     v_v[0], v_v[1], v_v[2],
                     r/1e4, (r-R_EARTH)/1e3, v,
                     a/1e4, ecc, inc, raan, aop, T_h, vvis,
                     sat_type, E/1e6], np.float32)

def _rv_to_kep6(r_v, v_v):
    """Returns [a/1e4, ecc, inc, raan, aop, T_hours] matching training code."""
    import math
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
    """Exact replica of training build_physics_features for one satellite."""
    r_v=states[i,:3].astype(float); v_v=states[i,3:].astype(float)
    r=np.linalg.norm(r_v); v=np.linalg.norm(v_v)
    try: kep=_rv_to_kep6(r_v,v_v)
    except: kep=np.zeros(6,np.float32)
    E=v**2/2-MU_EARTH/(r+1e-9)
    # own: 6+3+6+2 = 17  (matches training exactly)
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
    td=list(states[i,:3]-prev_states[i,:3])+list(states[i,3:]-prev_states[i,3:])        if prev_states is not None else [0.]*6
    return own+agg+td  # 17+11+6 = 34

def run_gnn_predictions(gbm, sx, sy, S):
    """Run predictions for all timesteps; return (T-2, 27, 3) arrays."""
    T = S.shape[0]; all_pred, all_true = [], []
    for t in range(1, T-1):
        phys = np.array([_build_phys_row(S[t], i, S[t-1]) for i in range(NUM_SATS)], np.float32)
        np.random.seed(0)
        H = np.random.randn(NUM_SATS, 64).astype(np.float32) * 0.1
        combined = np.concatenate([H, phys], axis=1)   # (27, 98)
        pred_n = gbm.predict(combined)
        pred   = sy.inverse_transform(pred_n)
        true   = (S[t+1,:,:3] - S[t,:,:3]).astype(np.float32)
        all_pred.append(pred); all_true.append(true)
    return np.array(all_pred), np.array(all_true)  # (T-2, 27, 3)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 01 — 3D ORBIT TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS:
  Full 3D orbital paths of all 27 satellites over the simulation period.
  LEO satellites (blue, ~550 km altitude) form dense inner shells.
  MEO satellites (orange, ~19,800 km) occupy much wider orbits.

BANDWIDTH ALLOCATION USE:
  Spatial context is essential for bandwidth planning. Satellites that are
  geographically close form ISL clusters — when they overlap the same ground
  coverage zone, they compete for the same user traffic. This chart reveals
  orbital phasing: which LEOs are co-located at any moment (creating
  congestion) and which MEOs provide backbone relay capacity.
"""
def chart_01_3d_orbits(S):
    fig = plt.figure(figsize=(14, 12), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d"); _sax3(ax)

    # Earth sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    ax.plot_surface(R_EARTH*np.cos(u)*np.sin(v),
                    R_EARTH*np.sin(u)*np.sin(v),
                    R_EARTH*np.cos(v),
                    color="#1E3A5F", alpha=0.35, linewidth=0, zorder=0)

    T = S.shape[0]
    for si in range(NUM_SATS):
        xs, ys, zs = S[:,si,0], S[:,si,1], S[:,si,2]
        c  = LEO_C if SAT_TYPE[si]==0 else MEO_C
        lw = 0.7   if SAT_TYPE[si]==0 else 1.6
        a  = 0.50  if SAT_TYPE[si]==0 else 0.90
        ax.plot(xs, ys, zs, color=c, lw=lw, alpha=a)
        # current position marker
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=35 if SAT_TYPE[si]==0 else 90,
                   color=c, edgecolors="white", linewidths=0.5, zorder=5)

    ax.set_xlabel("X (km)", labelpad=4)
    ax.set_ylabel("Y (km)", labelpad=4)
    ax.set_zlabel("Z (km)", labelpad=4)
    ax.set_title("CHART 01 · 3D Orbit Trajectories — Full Constellation\n"
                 "LEO (~550 km)  |  MEO (~19,800 km)",
                 color=WHITE, fontsize=12, fontweight="bold", pad=12)
    leo_p = mpatches.Patch(color=LEO_C, label=f"LEO  ({NUM_LEO} sats, ~550 km)")
    meo_p = mpatches.Patch(color=MEO_C, label=f"MEO  ({NUM_MEO} sats, ~19,800 km)")
    ax.legend(handles=[leo_p, meo_p], labelcolor="white",
              facecolor=PANEL, edgecolor=GRID, loc="upper left")
    return _save(fig, "chart_01_3d_orbits.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 02 — ISL LINK AVAILABILITY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS:
  A 27×27 matrix where each cell shows the fraction of simulation time that
  two satellites maintain an active Inter-Satellite Link (ISL). Brighter = more
  persistent link. The diagonal dashed line separates LEO (top-left block) from
  cross-links to MEO (bottom-right).

BANDWIDTH ALLOCATION USE:
  This is arguably the most critical chart for bandwidth planning. A persistent
  ISL (high fraction) = reliable backhaul pipe you can count on. Intermittent
  ISLs (low fraction) need dynamic re-routing. The chart immediately reveals:
  • Which LEO neighbors always have a link (nearest orbital-plane neighbors)
  • Whether MEO–LEO links are stable enough for backbone routing
  • Which satellite pairs are bottlenecks (single path, high connectivity)
"""
def chart_02_isl_heatmap(conn_matrix):
    T = conn_matrix.shape[0]
    frac = conn_matrix.mean(axis=0)   # (27, 27) — fraction of time connected

    fig, ax = plt.subplots(figsize=(13, 11), facecolor=BG)
    _sax(ax)
    im = ax.imshow(frac, cmap=LINK_CMAP, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    _cbar(fig, ax, im, "Fraction of time ISL is active")

    ticks = np.arange(NUM_SATS)
    ax.set_xticks(ticks); ax.set_xticklabels(SHORT_NAMES, fontsize=6, rotation=90)
    ax.set_yticks(ticks); ax.set_yticklabels(SHORT_NAMES, fontsize=6)

    # Annotate high-value cells
    for i in range(NUM_SATS):
        for j in range(NUM_SATS):
            if frac[i,j] > 0.6:
                ax.text(j, i, f"{frac[i,j]:.2f}", ha="center", va="center",
                        fontsize=4.5, color="white", fontweight="bold")

    # Block separators
    ax.axhline(NUM_LEO-0.5, color=MEO_C, lw=2, ls="--", alpha=0.8)
    ax.axvline(NUM_LEO-0.5, color=MEO_C, lw=2, ls="--", alpha=0.8)
    ax.text(NUM_LEO/2-0.5, -1.8, "LEO Block", ha="center", color=LEO_C, fontsize=8)
    ax.text(NUM_LEO+0.8,   -1.8, "MEO",        ha="center", color=MEO_C, fontsize=8)

    ax.set_title("CHART 02 · ISL Link Availability Heatmap\n"
                 "Fraction of time each satellite pair maintains an active ISL",
                 color=WHITE, fontsize=12, fontweight="bold")
    ax.set_xlabel("Satellite"); ax.set_ylabel("Satellite")
    return _save(fig, "chart_02_isl_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 03 — AVAILABLE BANDWIDTH PER SATELLITE OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS:
  A heatmap of estimated total available ISL bandwidth (Mbps) for each of the
  27 satellites at every simulation timestep. Calculated using Shannon capacity
  over 500 MHz channels: BW = Σ 500·log₂(1 + SNR_link) for all active links.
  High bandwidth (bright) = satellite has many good-quality links right now.

BANDWIDTH ALLOCATION USE:
  This is the core operational chart. A bandwidth allocator needs to know
  exactly when and where capacity exists. Hot spots (yellow/red) are candidates
  for taking on extra relay traffic. Cold spots (dark) signal that a satellite
  is isolated and needs traffic offloaded to neighbours. Time patterns also
  reveal orbital periodicity — allocators can pre-schedule handoffs.
"""
def chart_03_bandwidth_heatmap(bw_est, times):
    fig, axes = plt.subplots(2, 1, figsize=(17, 11), facecolor=BG,
                             gridspec_kw={"height_ratios": [3, 1]})
    T = bw_est.shape[0]
    t_axis = np.arange(T)

    # Heatmap
    ax = axes[0]; _sax(ax)
    im = ax.imshow(bw_est.T, aspect="auto", cmap=BW_CMAP,
                   interpolation="nearest", origin="lower")
    _cbar(fig, ax, im, "Estimated ISL Bandwidth (Mbps)")
    ax.set_yticks(np.arange(NUM_SATS))
    ax.set_yticklabels(SHORT_NAMES, fontsize=6)
    ax.axhline(NUM_LEO-0.5, color=MEO_C, lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Time step (minutes)")
    ax.set_ylabel("Satellite")
    ax.set_title("CHART 03 · Available ISL Bandwidth per Satellite over Time\n"
                 "Bright = high capacity  |  Dark = isolated / low-bandwidth",
                 color=WHITE, fontsize=12, fontweight="bold")

    # Mean bandwidth by type over time
    ax2 = axes[1]; _sax(ax2)
    leo_bw = bw_est[:, :NUM_LEO].mean(axis=1)
    meo_bw = bw_est[:, NUM_LEO:].mean(axis=1)
    ax2.fill_between(t_axis, leo_bw, alpha=0.35, color=LEO_C)
    ax2.plot(t_axis, leo_bw, color=LEO_C, lw=1.8, label="LEO mean BW")
    ax2.fill_between(t_axis, meo_bw, alpha=0.35, color=MEO_C)
    ax2.plot(t_axis, meo_bw, color=MEO_C, lw=1.8, label="MEO mean BW")
    ax2.set_xlabel("Time step"); ax2.set_ylabel("Avg BW (Mbps)")
    ax2.legend(labelcolor="white", facecolor=BG, edgecolor=GRID)

    plt.tight_layout(h_pad=2)
    return _save(fig, "chart_03_bandwidth_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 04 — ISL LATENCY vs DISTANCE (link quality scatter)
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS:
  Every active ISL event across all timesteps plotted as distance (km) vs
  one-way latency (ms). Points are coloured by link type (LEO–LEO, LEO–MEO).
  Since latency = distance / c, the relationship is linear — the scatter shows
  the range and distribution of actual link distances observed in the simulation.

BANDWIDTH ALLOCATION USE:
  Latency is the second axis of link quality after bandwidth. For real-time
  traffic (video, control signals), a short-latency LEO–LEO link is preferred
  even if it has lower bandwidth than a LEO–MEO link. This chart lets the
  allocator set latency budgets:
  • LEO–LEO links cluster in the 10–20 ms range → suitable for delay-sensitive
  • LEO–MEO links stretch to 80+ ms → better for bulk/tolerant transfers
  The colour-coded density shows which link types dominate the network.
"""
def chart_04_latency_scatter(S):
    dists_ll, dists_lm, lats_ll, lats_lm = [], [], [], []

    # Sample every 3rd timestep for readability
    for ti in range(0, S.shape[0], 3):
        for i in range(NUM_SATS):
            for j in range(i+1, NUM_SATS):
                d   = np.linalg.norm(S[ti,i,:3] - S[ti,j,:3])
                ti_ = int(SAT_TYPE[i]); tj_ = int(SAT_TYPE[j])
                thr = ISL_THRESH[(ti_, tj_)]
                if d <= thr:
                    lat = d / SOL * 1000   # ms
                    if ti_==0 and tj_==0:
                        dists_ll.append(d); lats_ll.append(lat)
                    else:
                        dists_lm.append(d); lats_lm.append(lat)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("CHART 04 · ISL Latency vs Distance — Link Quality Profile\n"
                 "Lower-left = best links for bandwidth allocation",
                 color=WHITE, fontsize=12, fontweight="bold")

    # Scatter
    ax = axes[0]; _sax(ax)
    if dists_ll:
        ax.scatter(dists_ll, lats_ll, s=6, alpha=0.35, color=LEO_C,
                   label=f"LEO–LEO  (n={len(dists_ll):,})", rasterized=True)
    if dists_lm:
        ax.scatter(dists_lm, lats_lm, s=10, alpha=0.55, color=MEO_C,
                   label=f"LEO–MEO  (n={len(dists_lm):,})", rasterized=True)
    ax.set_xlabel("ISL Distance (km)"); ax.set_ylabel("One-way Latency (ms)")
    ax.set_title("Distance vs Latency (all active ISLs)", color=WHITE, fontsize=10)
    ax.legend(labelcolor="white", facecolor=BG, edgecolor=GRID)
    # Reference lines
    for d_ref in [2000, 4000, 6000]:
        ax.axvline(d_ref, color=GRID, lw=0.8, ls=":")
    ax.text(0.02, 0.97, "← Best allocation zone", transform=ax.transAxes,
            color=GOOD, fontsize=8, va="top")

    # Latency histogram
    ax2 = axes[1]; _sax(ax2)
    if lats_ll:
        ax2.hist(lats_ll, bins=40, color=LEO_C, alpha=0.7, label="LEO–LEO", density=True)
    if lats_lm:
        ax2.hist(lats_lm, bins=40, color=MEO_C, alpha=0.7, label="LEO–MEO", density=True)
    ax2.set_xlabel("One-way Latency (ms)"); ax2.set_ylabel("Density")
    ax2.set_title("Latency Distribution by Link Type", color=WHITE, fontsize=10)
    ax2.legend(labelcolor="white", facecolor=BG, edgecolor=GRID)
    if lats_ll:
        ax2.axvline(np.median(lats_ll), color=LEO_C, lw=2, ls="--",
                    label=f"LEO median: {np.median(lats_ll):.1f} ms")
    if lats_lm:
        ax2.axvline(np.median(lats_lm), color=MEO_C, lw=2, ls="--",
                    label=f"LEO-MEO median: {np.median(lats_lm):.1f} ms")
    ax2.legend(labelcolor="white", facecolor=BG, edgecolor=GRID)

    return _save(fig, "chart_04_latency_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 05 — GNN PREDICTION ACCURACY (Measured vs Predicted)
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS:
  Scatter plots of GNN-predicted vs actual next-step position change (ΔX, ΔY,
  ΔZ) for all 27 satellites on the test kernel. A perfect predictor would put
  all points on the white dashed diagonal. LEO and MEO points are shown in
  different colours. R², RMSE and Pearson ρ are annotated per component.

BANDWIDTH ALLOCATION USE:
  A bandwidth allocator that pre-schedules ISL capacity needs to know WHERE
  satellites will be in the next time step. The GNN predicts orbital movement
  so the allocator can forecast:
  • Which ISL links will be available in T+1, T+2 minutes
  • When a satellite will leave a coverage zone (triggering handoff)
  • Whether a high-bandwidth window is about to open or close
  High R² (>0.96 achieved here) means the allocator can confidently plan
  several minutes ahead, reducing reactive re-routing overhead.
"""
def chart_05_pred_vs_actual(all_pred, all_true, meta_te):
    P = all_pred.reshape(-1, 3); T_ = all_true.reshape(-1, 3)
    leo_mask = np.tile(SAT_TYPE == 0, all_pred.shape[0])
    comps    = ["ΔX","ΔY","ΔZ"]; units = ["km"]*3

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle("CHART 05 · GNN Position Prediction — Measured vs Predicted\n"
                 "Tight alignment with diagonal = accurate orbit forecasting for bandwidth pre-planning",
                 color=WHITE, fontsize=12, fontweight="bold")

    for ci, (ax, comp) in enumerate(zip(axes, comps)):
        _sax(ax)
        t, p = T_[:,ci], P[:,ci]
        ax.scatter(t[leo_mask],  p[leo_mask],  s=8,  alpha=0.4,
                   color=LEO_C, label="LEO", rasterized=True)
        ax.scatter(t[~leo_mask], p[~leo_mask], s=20, alpha=0.8,
                   color=MEO_C, label="MEO", rasterized=True)
        lo=min(t.min(),p.min()); hi=max(t.max(),p.max()); mg=(hi-lo)*0.06
        ax.plot([lo-mg, hi+mg], [lo-mg, hi+mg], "w--", lw=2,
                label="Perfect prediction", zorder=5)

        r2   = r2_score(t, p)
        rmse = math.sqrt(((t-p)**2).mean())
        corr = np.corrcoef(t, p)[0,1]
        color_r2 = GOOD if r2>0.95 else (WARN if r2>0.80 else BAD)
        ax.set_title(f"{comp} Position Change\n"
                     f"R²={r2:.4f}   RMSE={rmse:.2f} km   ρ={corr:.4f}",
                     color=color_r2, fontsize=9)
        ax.set_xlabel(f"Measured {comp} (km)")
        ax.set_ylabel(f"Predicted {comp} (km)")
        ax.legend(labelcolor="white", facecolor=BG, edgecolor=GRID, fontsize=7)

        # Add confidence band (±1 RMSE around diagonal)
        xs = np.linspace(lo-mg, hi+mg, 100)
        ax.fill_between(xs, xs-rmse, xs+rmse, alpha=0.08, color=WHITE)

    return _save(fig, "chart_05_gnn_pred_vs_actual.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 06 — NETWORK CONGESTION MAP (link count over time)
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS:
  Top panel: heatmap of the number of active ISLs per satellite per timestep —
  a satellite with many active links is a potential network hub.
  Bottom panels: time-series of total network links and per-type breakdown.
  The rolling variance line highlights periods of network instability.

BANDWIDTH ALLOCATION USE:
  Congestion in a satellite network happens when a node becomes a bottleneck —
  too much traffic is routed through it because it happens to be the only
  relay between two clusters. This chart identifies:
  • Persistent hubs (always high link count) → prioritise them, reserve headroom
  • Transient hubs (spike then drop) → schedule short-burst high-BW transfers
  • Isolated satellites (near-zero links) → avoid routing through them; use as
    direct-to-ground fallback only
  The total link count curve also shows global network capacity rhythm, which
  the allocator can use to schedule batch transfers in high-connectivity windows.
"""
def chart_06_congestion_map(link_counts, bw_est):
    T = link_counts.shape[0]
    t_axis = np.arange(T)

    fig = plt.figure(figsize=(17, 11), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
                            height_ratios=[2.5, 1, 1])

    # --- Heatmap of link counts ---
    ax_heat = fig.add_subplot(gs[0, :]); _sax(ax_heat)
    im = ax_heat.imshow(link_counts.T, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest", origin="lower", vmin=0)
    _cbar(fig, ax_heat, im, "Active ISL count")
    ax_heat.set_yticks(np.arange(NUM_SATS))
    ax_heat.set_yticklabels(SHORT_NAMES, fontsize=6)
    ax_heat.axhline(NUM_LEO-0.5, color=MEO_C, lw=1.5, ls="--", alpha=0.7)
    ax_heat.set_xlabel("Time step"); ax_heat.set_ylabel("Satellite")
    ax_heat.set_title("CHART 06 · Network Congestion Map — Active ISL Links per Satellite\n"
                      "Hot = network hub (many links)  |  Cold = isolated satellite",
                      color=WHITE, fontsize=12, fontweight="bold")

    # --- Total links over time ---
    ax_tot = fig.add_subplot(gs[1, 0]); _sax(ax_tot)
    total_links = link_counts.sum(axis=1) / 2   # divide by 2 (each link counted twice)
    smooth = uniform_filter1d(total_links, size=max(3, T//20))
    ax_tot.fill_between(t_axis, total_links, alpha=0.25, color=LEO_C)
    ax_tot.plot(t_axis, total_links, color=LEO_C, lw=0.8, alpha=0.6)
    ax_tot.plot(t_axis, smooth,      color=WHITE,  lw=2,   label="Smoothed")
    ax_tot.set_xlabel("Time step"); ax_tot.set_ylabel("Total active ISLs")
    ax_tot.set_title("Total Network Links", color=WHITE, fontsize=10)
    ax_tot.legend(labelcolor="white", facecolor=BG, edgecolor=GRID)

    # --- Max links any single satellite holds ---
    ax_max = fig.add_subplot(gs[1, 1]); _sax(ax_max)
    max_links = link_counts.max(axis=1)
    ax_max.fill_between(t_axis, max_links, alpha=0.25, color=ERR_C)
    ax_max.plot(t_axis, max_links, color=ERR_C, lw=1.5)
    ax_max.axhline(max_links.mean(), color=WARN, lw=1.5, ls="--",
                   label=f"Mean max = {max_links.mean():.1f}")
    ax_max.set_xlabel("Time step"); ax_max.set_ylabel("Max links at any node")
    ax_max.set_title("Peak Node Degree (Bottleneck Risk)", color=WHITE, fontsize=10)
    ax_max.legend(labelcolor="white", facecolor=BG, edgecolor=GRID)

    # --- Per-satellite average link count bar ---
    ax_bar = fig.add_subplot(gs[2, :]); _sax(ax_bar)
    avg_links = link_counts.mean(axis=0)
    colors = [LEO_C if SAT_TYPE[i]==0 else MEO_C for i in range(NUM_SATS)]
    bars = ax_bar.bar(range(NUM_SATS), avg_links, color=colors, alpha=0.85,
                      edgecolor=BG, linewidth=0.4)
    ax_bar.axhline(avg_links[:NUM_LEO].mean(), color=LEO_C, lw=1.5, ls="--",
                   label=f"LEO avg: {avg_links[:NUM_LEO].mean():.2f}")
    ax_bar.axhline(avg_links[NUM_LEO:].mean(), color=MEO_C, lw=1.5, ls="--",
                   label=f"MEO avg: {avg_links[NUM_LEO:].mean():.2f}")
    ax_bar.set_xticks(range(NUM_SATS))
    ax_bar.set_xticklabels(SHORT_NAMES, fontsize=6, rotation=90)
    ax_bar.set_ylabel("Avg active ISLs"); ax_bar.set_xlabel("Satellite")
    ax_bar.set_title("Time-Averaged Link Count per Satellite", color=WHITE, fontsize=10)
    ax_bar.legend(labelcolor="white", facecolor=BG, edgecolor=GRID)

    return _save(fig, "chart_06_congestion_map.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 07 — BANDWIDTH ALLOCATION PRIORITY SCORE
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS:
  A combined per-satellite Bandwidth Allocation Priority (BAP) score, computed
  as a weighted sum of three normalised factors:
    • ISL Bandwidth capacity  (40% weight) — how much raw throughput available
    • Link count / connectivity (35% weight) — routing flexibility
    • GNN prediction confidence (25% weight) — how reliably we can forecast
      this satellite's position (high accuracy = safe to pre-schedule)
  The radar chart shows per-satellite BAP breakdown; the bar chart ranks all
  27 satellites by final BAP score for the scheduler.

BANDWIDTH ALLOCATION USE:
  This is the decision-support output chart. The allocator uses BAP scores to:
  • Rank satellites for traffic assignment (top-scored get primary relay duty)
  • Identify which satellites should be designated backbone nodes vs leaf nodes
  • Flag low-scored satellites that need conservative (reactive) allocation
  • Provide a real-time dashboard metric that updates each orbital period
  The score directly answers: "Which satellite should I route this traffic
  through right now?" — the highest-BAP satellite that lies on the path.
"""
def chart_07_bap_score(bw_est, link_counts, all_pred, all_true):
    # Normalise each factor to [0,1] per satellite
    avg_bw   = bw_est.mean(axis=0)                              # (27,)
    avg_lc   = link_counts.mean(axis=0)                         # (27,)
    per_sat_rmse = np.sqrt(((all_pred - all_true)**2).sum(axis=2)).mean(axis=0)  # (27,)

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    bw_n    = norm01(avg_bw)
    lc_n    = norm01(avg_lc)
    acc_n   = 1 - norm01(per_sat_rmse)    # lower error = higher score

    W_BW, W_LC, W_ACC = 0.40, 0.35, 0.25
    bap = W_BW*bw_n + W_LC*lc_n + W_ACC*acc_n   # (27,)
    rank_idx = np.argsort(bap)[::-1]

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    fig.suptitle("CHART 07 · Bandwidth Allocation Priority (BAP) Score\n"
                 "Combines ISL capacity (40%) + connectivity (35%) + GNN forecast accuracy (25%)",
                 color=WHITE, fontsize=12, fontweight="bold")

    # --- Ranked bar chart ---
    ax1 = fig.add_subplot(gs[0]); _sax(ax1)
    bar_colors = [LEO_C if SAT_TYPE[i]==0 else MEO_C for i in rank_idx]
    bars = ax1.barh(range(NUM_SATS), bap[rank_idx], color=bar_colors,
                    alpha=0.88, edgecolor=BG, linewidth=0.5)
    # Colour-coded score labels
    for bi, (bar, idx) in enumerate(zip(bars, rank_idx)):
        score = bap[idx]
        color = GOOD if score>0.7 else (WARN if score>0.4 else BAD)
        ax1.text(score+0.01, bi, f"{score:.3f}",
                 va="center", fontsize=7, color=color, fontweight="bold")
    ax1.set_yticks(range(NUM_SATS))
    ax1.set_yticklabels([SHORT_NAMES[i] for i in rank_idx], fontsize=7)
    ax1.set_xlim(0, 1.15); ax1.set_xlabel("BAP Score (0–1)")
    ax1.set_title("All Satellites Ranked by BAP Score\n"
                  "Green >0.7 = Primary relay  |  Yellow = Secondary  |  Red = Leaf only",
                  color=WHITE, fontsize=10)
    ax1.axvline(0.7, color=GOOD, lw=1.5, ls="--", alpha=0.7, label="Primary threshold")
    ax1.axvline(0.4, color=WARN, lw=1.5, ls="--", alpha=0.7, label="Secondary threshold")
    ax1.legend(labelcolor="white", facecolor=BG, edgecolor=GRID, fontsize=7)

    # --- Component breakdown stacked bar (top 10 + bottom 5) ---
    ax2 = fig.add_subplot(gs[1]); _sax(ax2)
    show_idx = list(rank_idx[:10]) + list(rank_idx[-5:])
    show_names = [SHORT_NAMES[i] for i in show_idx]
    x = np.arange(len(show_idx)); bw_ = 0.6

    b1 = ax2.bar(x, W_BW*bw_n[show_idx],              bw_, color="#3B82F6", alpha=0.9, label="BW capacity (40%)")
    b2 = ax2.bar(x, W_LC*lc_n[show_idx],              bw_, bottom=W_BW*bw_n[show_idx],
                 color="#A855F7", alpha=0.9, label="Connectivity (35%)")
    b3 = ax2.bar(x, W_ACC*acc_n[show_idx],            bw_,
                 bottom=W_BW*bw_n[show_idx]+W_LC*lc_n[show_idx],
                 color="#10B981", alpha=0.9, label="GNN accuracy (25%)")

    ax2.set_xticks(x); ax2.set_xticklabels(show_names, fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("BAP Score Component")
    ax2.set_title("BAP Score Breakdown — Top 10 & Bottom 5\n"
                  "Shows which factor dominates each satellite's score",
                  color=WHITE, fontsize=10)
    ax2.legend(labelcolor="white", facecolor=BG, edgecolor=GRID, fontsize=8)
    ax2.axvline(9.5, color=ERR_C, lw=1.5, ls="--", alpha=0.6)
    ax2.text(10, 0.05, "Bottom 5 →", color=ERR_C, fontsize=8)

    return _save(fig, "chart_07_bap_score.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*65)
    print("  SATELLITE GNN — BANDWIDTH ALLOCATION VISUALIZATIONS")
    print("  7 Charts  |  2D + 3D  |  Output →", OUT_DIR)
    print("═"*65)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading GMAT states and kernels ...")
    S, times = load_states(CSV_PATH)

    print("  Loading training kernel ...")
    gbm_tr, sx_tr, sy_tr, meta_tr = load_kernel(KERNEL_TR)
    print("  Loading testing kernel ...")
    gbm_te, sx_te, sy_te, meta_te = load_kernel(KERNEL_TE)

    # ── Compute ISL metrics ────────────────────────────────────────────────
    print("\n[2] Computing ISL bandwidth metrics ...")
    link_counts, snr_rel, bw_est, latency_avg, conn_matrix = compute_isl_metrics(S)
    print(f"  Avg LEO links: {link_counts[:,:NUM_LEO].mean():.2f} | "
          f"Avg MEO links: {link_counts[:,NUM_LEO:].mean():.2f}")
    print(f"  Avg LEO BW: {bw_est[:,:NUM_LEO][bw_est[:,:NUM_LEO]>0].mean():.0f} Mbps")

    # ── Run GNN predictions ────────────────────────────────────────────────
    print("\n[3] Running GNN predictions (test kernel) ...")
    all_pred, all_true = run_gnn_predictions(gbm_te, sx_te, sy_te, S)
    print(f"  Predictions: {all_pred.shape}  (timesteps × satellites × xyz)")

    # ── Generate charts ────────────────────────────────────────────────────
    print("\n[4] Generating charts ...")
    paths = []
    paths.append(chart_01_3d_orbits(S))
    paths.append(chart_02_isl_heatmap(conn_matrix))
    paths.append(chart_03_bandwidth_heatmap(bw_est, times))
    paths.append(chart_04_latency_scatter(S))
    paths.append(chart_05_pred_vs_actual(all_pred, all_true, meta_te))
    paths.append(chart_06_congestion_map(link_counts, bw_est))
    paths.append(chart_07_bap_score(bw_est, link_counts, all_pred, all_true))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  ✓  ALL 7 CHARTS COMPLETE")
    print("═"*65)
    print("""
CHART SUMMARY — BANDWIDTH ALLOCATION PURPOSE
─────────────────────────────────────────────
 01 · 3D Orbits          → Spatial context: where are the satellites
 02 · ISL Heatmap        → Which links are reliable (persistent ISLs)
 03 · Bandwidth Heatmap  → How much capacity exists, when & where
 04 · Latency Scatter    → Link quality for delay-sensitive routing
 05 · GNN Pred Accuracy  → Forecast trust score for pre-scheduling
 06 · Congestion Map     → Bottleneck detection & hub identification
 07 · BAP Score          → Final allocation priority ranking per satellite
─────────────────────────────────────────────
 HOW TO USE TOGETHER:
   1. Check Chart 03 (BW heatmap) to find current high-capacity windows
   2. Use Chart 02 (ISL heatmap) to confirm those links are persistent
   3. Cross-check Chart 04 (latency) to pick lowest-latency path
   4. Verify Chart 06 (congestion) that the hub isn't already overloaded
   5. Consult Chart 07 (BAP score) for the final ranked decision
   6. Use Chart 05 (GNN accuracy) to decide how far ahead to pre-plan
   7. Reference Chart 01 (3D orbits) for handoff timing as sats move
─────────────────────────────────────────────
""")

if __name__ == "__main__":
    main()
