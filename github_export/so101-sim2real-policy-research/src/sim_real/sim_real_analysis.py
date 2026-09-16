"""
SO101-SimRealGap: Comprehensive Joint-Space & Temporal Gap Analysis
===================================================================
Datasets (LeRobot format, SO-101, 6 joints):
  SIM : ngocthuong2212/so101_pick_red_no_ep100_left
  REAL: vasco281204/pick_red_fixed

Implements Sections 4.2 (Joint-Space Gap) and 4.3 (Temporal Response Gap)
from the SO101-SimRealGap benchmark paper.

Run:
    pip install datasets huggingface_hub scipy numpy pandas matplotlib seaborn
    python sim_real_gap_analysis.py
"""

import os, warnings
import argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy import signal
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.interpolate import interp1d

# ── Try to import optional libs ─────────────────────────────────────────────
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[WARN] `datasets` not installed – will use synthetic data for demo")

# ── Config ───────────────────────────────────────────────────────────────────
SIM_REPO  = "ngocthuong2212/so101_pick_red_no_ep100_left"
REAL_REPO = "vasco281204/pick_red_fixed"

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper"
]
N_JOINTS = 6

# Thresholds
JERK_WINDOW     = 3          # finite difference order for jerk
LAG_MAX_FRAMES  = 30         # max lag to search for cross-correlation
ACTION_KEYS     = None       # auto-detected
STATE_KEYS      = None

OUT_DIR = "sim_real_gap_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE = {"sim": "#4C72B0", "real": "#DD8452"}
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comprehensive joint-space and temporal sim-real gap analysis for SO-101 datasets."
    )
    parser.add_argument("--sim-dataset", default=SIM_REPO)
    parser.add_argument("--real-dataset", default=REAL_REPO)
    parser.add_argument("--output-dir", default=OUT_DIR)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _auto_detect_keys(df: pd.DataFrame):
    """Detect action/state column prefixes in a LeRobot DataFrame."""
    action_cols = [c for c in df.columns if c.startswith("action")]
    state_cols  = [c for c in df.columns if "observation.state" in c or c.startswith("observation_state")]
    if not state_cols:
        state_cols = [c for c in df.columns if c.startswith("state")]
    return action_cols, state_cols


def load_lerobot_dataset(repo_id: str, domain: str) -> pd.DataFrame:
    """Load a LeRobot HuggingFace dataset into a flat DataFrame."""
    print(f"  Loading {domain} dataset from {repo_id} …")
    ds = load_dataset(repo_id, data_dir="data", split="train")
    df = ds.to_pandas()
    df["domain"] = domain

    # Flatten list/array columns (e.g. observation.state → state_0..state_5)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                sample = df[col].iloc[0]
                if hasattr(sample, "__len__") and not isinstance(sample, str):
                    arr = np.stack(df[col].values)
                    for i in range(arr.shape[1]):
                        df[f"{col}_{i}"] = arr[:, i]
                    df.drop(columns=[col], inplace=True)
            except Exception:
                pass
    print(f"    → {len(df):,} frames, columns: {df.shape[1]}")
    return df


def load_or_simulate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real datasets or generate synthetic demo data."""
    if HAS_DATASETS:
        try:
            sim_df  = load_lerobot_dataset(SIM_REPO,  "sim")
            real_df = load_lerobot_dataset(REAL_REPO, "real")
            return sim_df, real_df
        except Exception as e:
            print(f"  [WARN] Could not load from HuggingFace: {e}")
            print("  → Falling back to synthetic demo data")

    return _generate_synthetic_data()


def _generate_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic synthetic LeRobot-style data for demonstration.
    Real joints have: larger jerk, actuator delay, velocity smoothing loss.
    """
    rng = np.random.default_rng(42)
    n_episodes = 40
    frames_per_ep = 150

    def make_episode(ep_idx, domain):
        t = np.arange(frames_per_ep) * (1.0 / 30.0)   # 30 Hz → dt~0.033s
        rows = []
        # Sinusoidal base motion + noise
        phase_offset = rng.uniform(0, np.pi, N_JOINTS)
        amplitude    = rng.uniform(0.3, 1.2, N_JOINTS)

        for f in range(frames_per_ep):
            state  = amplitude * np.sin(2 * np.pi * 0.3 * t[f] + phase_offset)
            action = amplitude * np.sin(2 * np.pi * 0.3 * (t[f] + (1.0 / 30.0))) + phase_offset

            if domain == "real":
                # Simulate: bias offset, extra noise, actuator delay effect
                state  += rng.normal([0.05,-0.08, 0.12,-0.06, 0.03, 0.0], 0.04)
                state  += rng.normal(0, 0.03, N_JOINTS)
                action += rng.normal(0, 0.02, N_JOINTS)
            else:
                state  += rng.normal(0, 0.005, N_JOINTS)
                action += rng.normal(0, 0.003, N_JOINTS)

            row = {
                "episode_index": ep_idx,
                "frame_index":   f,
                "timestamp":     t[f],
                "domain":        domain,
            }
            for j in range(N_JOINTS):
                row[f"observation.state_{j}"] = state[j]
                row[f"action_{j}"]            = action[j]
            rows.append(row)
        return rows

    sim_rows, real_rows = [], []
    for i in range(n_episodes):
        sim_rows  += make_episode(i, "sim")
        real_rows += make_episode(i, "real")

    return pd.DataFrame(sim_rows), pd.DataFrame(real_rows)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_joint_arrays(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Extract (N_frames, N_joints) array for columns matching prefix."""
    cols = sorted([c for c in df.columns if c.startswith(prefix)
                   and c.split("_")[-1].isdigit()])[:N_JOINTS]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'. "
                       f"Available: {df.columns.tolist()[:20]}")
    
    # Lấy dữ liệu thô
    raw_values = df[cols].values.astype(float)
    # Tự động chuyển đổi từ Độ sang Radian để tính toán các thông số chuẩn xác hơn
    return np.deg2rad(raw_values)


def compute_velocity(q: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
    """Finite-difference velocity: q_dot[t] = (q[t+1]-q[t]) / dt"""
    return np.diff(q, axis=0) / dt


def compute_jerk(q: np.ndarray) -> np.ndarray:
    """3rd-order finite difference jerk approximation (Section 4.2 eq.)"""
    # j[t] = q[t+3] - 3q[t+2] + 3q[t+1] - q[t]
    j = (q[3:] - 3*q[2:-1] + 3*q[1:-2] - q[:-3])
    return j


def split_by_episode(df: pd.DataFrame, state_prefix: str, action_prefix: str):
    """Return list of (state_arr, action_arr) per episode."""
    episodes = []
    ep_col = "episode_index" if "episode_index" in df.columns else None
    if ep_col:
        for ep_id, group in df.groupby(ep_col):
            g = group.sort_values("frame_index") if "frame_index" in group.columns else group
            s = get_joint_arrays(g, state_prefix)
            a = get_joint_arrays(g, action_prefix)
            if len(s) > 0:
                episodes.append((s, a))
    else:
        s = get_joint_arrays(df, state_prefix)
        a = get_joint_arrays(df, action_prefix)
        episodes.append((s, a))
    return episodes

def get_kinematics(df: pd.DataFrame, state_prefix: str, action_prefix: str = None):
    """Trích xuất vị trí, tính toán vận tốc và độ giật cho từng episode để tránh nhiễu nối."""
    all_q, all_v, all_j, all_a = [], [], [], []
    ep_col = "episode_index" if "episode_index" in df.columns else None
    if ep_col:
        for ep_id, group in df.groupby(ep_col):
            g = group.sort_values("frame_index") if "frame_index" in group.columns else group
            q = get_joint_arrays(g, state_prefix)
            if len(q) > 0:
                all_q.append(q) # Lưu trữ vị trí khớp (joint position)
                if len(q) > 1: all_v.append(compute_velocity(q)) # Tính toán và lưu trữ vận tốc khớp (joint velocity)
                if len(q) > 3: all_j.append(compute_jerk(q)) # Tính toán và lưu trữ độ giật khớp (joint jerk)
            if action_prefix:
                a = get_joint_arrays(g, action_prefix)
                if len(a) > 0:
                    all_a.append(a[:len(q)])
    else:
        q = get_joint_arrays(df, state_prefix)
        if len(q) > 0: all_q.append(q) # Lưu trữ vị trí khớp (joint position)
        if len(q) > 1: all_v.append(compute_velocity(q)) # Tính toán và lưu trữ vận tốc khớp (joint velocity)
        if len(q) > 3: all_j.append(compute_jerk(q)) # Tính toán và lưu trữ độ giật khớp (joint jerk)
        if action_prefix:
            a = get_joint_arrays(df, action_prefix)
            if len(a) > 0:
                all_a.append(a[:len(q)])
    
    q_cat = np.concatenate(all_q) if all_q else np.array([])
    v_cat = np.concatenate(all_v) if all_v else np.array([])
    j_cat = np.concatenate(all_j) if all_j else np.array([])
    if action_prefix:
        a_cat = np.concatenate(all_a) if all_a else np.array([])
        return q_cat, v_cat, j_cat, a_cat
    return q_cat, v_cat, j_cat

def check_and_export_dataset_info(sim_df: pd.DataFrame, real_df: pd.DataFrame, out_dir: str):
    """
    Kiểm tra dataset đã thực sự được tải (có dữ liệu) hay không,
    đồng thời xuất ra file CSV (chứa 50 dòng đầu) và file TXT (chứa thống kê) để kiểm tra tham số.
    """
    print(f"\n  [+] Exporting dataset info for verification to ./{out_dir}/ ...")
    for name, df in [("sim", sim_df), ("real", real_df)]:
        if df is None or df.empty:
            print(f"    [ERROR] {name.upper()} dataset is empty or failed to load!")
            continue

        # Xuất toàn bộ dữ liệu ra file CSV
        csv_path = os.path.join(out_dir, f"verification_{name}_sample.csv")
        df.to_csv(csv_path, index=False)

        # Xuất thống kê, thông tin cột ra file TXT
        txt_path = os.path.join(out_dir, f"verification_{name}_stats.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== DATASET SUMMARY: {name.upper()} ===\n")
            f.write(f"Total rows: {len(df)}\nTotal columns: {len(df.columns)}\n\n")
            f.write("--- COLUMNS & DATA TYPES ---\n")
            for col in df.columns:
                f.write(f"  - {col}: {df[col].dtype}\n")
            f.write("\n--- DESCRIPTIVE STATISTICS ---\n")
            f.write(df.describe().to_string())
        print(f"    → {name.upper()} info saved: {csv_path} & {txt_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. SECTION 4.2 – JOINT-SPACE GAP
# ═══════════════════════════════════════════════════════════════════════════

def compute_joint_space_gap(sim_df, real_df, state_prefix, action_prefix):
    """
    Compute all joint-space gap metrics (Section 4.2):
      - Mean joint gap  Gμ(qi) = |μ(q_real) - μ(q_sim)|
      - Wasserstein distance on joint position distribution
      - Wasserstein distance on joint velocity distribution
      - Jerk gap  Gjerk = E[||j_real||] - E[||j_sim||]
      - KS statistic for distribution comparison
    Returns a dict of DataFrames/arrays.
    """
    print("\n── Section 4.2: Joint-Space Gap ──")

    sim_q, sim_v, sim_j, sim_a = get_kinematics(sim_df, state_prefix, action_prefix)
    real_q, real_v, real_j, real_a = get_kinematics(real_df, state_prefix, action_prefix)

    def get_delay(q, a, j):
        s = q[:, j]
        act = a[:, j]
        if s.std() < 1e-6 or act.std() < 1e-6:
            return 0
        ds = np.diff(s)
        da = np.diff(act)
        ds_norm = ds - ds.mean()
        da_norm = da - da.mean()
        corr = signal.correlate(ds_norm, da_norm, mode="full")
        lags = signal.correlation_lags(len(ds), len(da), mode="full")
        delay = int(lags[np.argmax(corr)])
        if delay > 30 or delay < -30:
            delay = 0
        return delay

    results = []
    for i, name in enumerate(JOINT_NAMES[:N_JOINTS]):
        sq, rq = sim_q[:, i], real_q[:, i]
        sv, rv = sim_v[:, i], real_v[:, i]
        sj, rj = sim_j[:, i], real_j[:, i]

        mean_gap    = abs(rq.mean() - sq.mean())
        std_gap     = abs(rq.std()  - sq.std())
        w_pos       = wasserstein_distance(sq, rq)
        w_vel       = wasserstein_distance(sv, rv)
        vel_mag_gap = np.mean(np.abs(rv)) - np.mean(np.abs(sv))
        jerk_gap    = np.mean(np.abs(rj)) - np.mean(np.abs(sj))
        ks_stat, ks_p = ks_2samp(sq, rq)
        
        sim_delay = get_delay(sim_q, sim_a, i) if len(sim_a) > 0 else 0
        real_delay = get_delay(real_q, real_a, i) if len(real_a) > 0 else 0

        results.append({
            "joint":          name,
            "mean_gap_Gμ":    round(mean_gap, 5),
            "std_gap":        round(std_gap, 5),
            "W_position_Gq":  round(w_pos, 5),
            "W_velocity_Gq̇":  round(w_vel, 5),
            "vel_mag_gap_Δμv": round(vel_mag_gap, 5),
            "jerk_gap_Gjerk": round(jerk_gap, 5),
            "sim_delay":      sim_delay,
            "real_delay":     real_delay,
            "KS_stat":        round(ks_stat, 5),
            "KS_pvalue":      round(ks_p, 5),
            # raw arrays for plotting
            "_sim_q": sq, "_real_q": rq,
            "_sim_v": sv, "_real_v": rv,
            "_sim_j": sj, "_real_j": rj,
        })
        print(f"  {name:15s}  Gμ={mean_gap:.4f}  W_pos={w_pos:.4f}  "
              f"W_vel={w_vel:.4f}  Δμv={vel_mag_gap:+.4f}  Gjerk={jerk_gap:+.4f}  SimDelay={sim_delay}f  RealDelay={real_delay}f")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. SECTION 4.3 – TEMPORAL RESPONSE GAP
# ═══════════════════════════════════════════════════════════════════════════

def _episode_cross_corr_delay(state_arr, action_arr, max_lag=LAG_MAX_FRAMES):
    """
    Per-episode cross-correlation lag between Δa(t) and Δq(t+τ).
    Returns best lag τ* (in frames).
    """
    dA = np.diff(action_arr, axis=0)  # (T-1, J)
    dQ = np.diff(state_arr,  axis=0)

    # Average across joints
    dA_norm = (dA - dA.mean()) / (dA.std() + 1e-9)
    dQ_norm = (dQ - dQ.mean()) / (dQ.std() + 1e-9)

    T = min(len(dA_norm), len(dQ_norm))
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag >= 0:
            end = T - lag
            c = (dA_norm[:end] * dQ_norm[lag:lag+end]).mean() if end > 0 else 0
        else:
            start = -lag
            c = (dA_norm[start:start+(T+lag)] * dQ_norm[:T+lag]).mean() if T+lag > 0 else 0
        corrs.append(c)

    best = lags[np.argmax(corrs)]
    return int(best), lags, np.array(corrs)


def _episode_gain(state_arr, action_arr, eps=1e-6):
    """gain(t) = ||Δq(t)|| / ||Δa(t)||, averaged over episode."""
    dA = np.diff(action_arr, axis=0)
    dQ = np.diff(state_arr,  axis=0)
    nA = np.linalg.norm(dA, axis=1)
    nQ = np.linalg.norm(dQ, axis=1)
    mask = nA > eps
    if mask.sum() == 0:
        return np.nan
    return float((nQ[mask] / (nA[mask] + eps)).mean())


def _episode_tracking_lag(state_arr, action_arr):
    """Simple per-joint mean abs tracking error (action - state)."""
    n = min(len(state_arr), len(action_arr))
    return float(np.mean(np.abs(action_arr[:n] - state_arr[:n])))


def compute_temporal_gap(sim_df, real_df, state_prefix, action_prefix):
    """
    Compute temporal response gap metrics (Section 4.3):
      - Response delay τ* (cross-correlation based)
      - Action-response gain
      - Tracking lag (mean abs error action vs state)
    Returns summary dict.
    """
    print("\n── Section 4.3: Temporal Response Gap ──")

    sim_eps  = split_by_episode(sim_df,  state_prefix, action_prefix)
    real_eps = split_by_episode(real_df, state_prefix, action_prefix)

    sim_delays,  real_delays  = [], []
    sim_gains,   real_gains   = [], []
    sim_lags,    real_lags    = [], []
    all_lag_curves = {"sim": [], "real": []}

    for s, a in sim_eps:
        d, lags, c = _episode_cross_corr_delay(s, a)
        sim_delays.append(d)
        sim_gains.append(_episode_gain(s, a))
        sim_lags.append(_episode_tracking_lag(s, a))
        all_lag_curves["sim"].append((lags, c))

    for s, a in real_eps:
        d, lags, c = _episode_cross_corr_delay(s, a)
        real_delays.append(d)
        real_gains.append(_episode_gain(s, a))
        real_lags.append(_episode_tracking_lag(s, a))
        all_lag_curves["real"].append((lags, c))

    sim_delays  = np.array(sim_delays)
    real_delays = np.array(real_delays)
    sim_gains   = np.array([x for x in sim_gains  if not np.isnan(x)])
    real_gains  = np.array([x for x in real_gains if not np.isnan(x)])

    Gdelay = float(np.mean(real_delays) - np.mean(sim_delays))
    Ggain  = float(np.mean(real_gains)  - np.mean(sim_gains))
    Glag   = float(np.mean(real_lags)   - np.mean(sim_lags))

    print(f"  Response delay   τ*(sim)={np.mean(sim_delays):.2f}  "
          f"τ*(real)={np.mean(real_delays):.2f}  Gdelay={Gdelay:+.2f} frames")
    print(f"  Action-resp gain  sim={np.mean(sim_gains):.4f}  "
          f"real={np.mean(real_gains):.4f}  Ggain={Ggain:+.4f}")
    print(f"  Tracking lag      sim={np.mean(sim_lags):.4f}  "
          f"real={np.mean(real_lags):.4f}  Glag={Glag:+.4f}")

    if Ggain < 0:
        print("  ⚠ Real robot responds WEAKER than simulation (Ggain < 0)")
    if Gdelay > 0:
        print("  ⚠ Real robot has HIGHER delay than simulation (Gdelay > 0)")

    return {
        "sim_delays":        sim_delays,
        "real_delays":       real_delays,
        "sim_gains":         sim_gains,
        "real_gains":        real_gains,
        "sim_lags":          np.array(sim_lags),
        "real_lags":         np.array(real_lags),
        "Gdelay":            Gdelay,
        "Ggain":             Ggain,
        "Glag":              Glag,
        "all_lag_curves":    all_lag_curves,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

def build_summary_table(joint_results, temporal_results):
    rows = []
    for r in joint_results:
        rows.append({
            "Joint":          r["joint"],
            "Mean Gap Gμ":    r["mean_gap_Gμ"],
            "Std Gap":        r["std_gap"],
            "W Position Gq":  r["W_position_Gq"],
            "W Velocity Gq̇":  r["W_velocity_Gq̇"],
            "Vel Mag Gap Δμv":r["vel_mag_gap_Δμv"],
            "Jerk Gap Gjerk": r["jerk_gap_Gjerk"],
            "Sim Delay (f)":  r["sim_delay"],
            "Real Delay (f)": r["real_delay"],
            "KS Stat":        r["KS_stat"],
            "KS p-value":     r["KS_pvalue"],
        })
    df = pd.DataFrame(rows)

    # Composite gap score = normalised sum of key metrics
    metric_cols = ["Mean Gap Gμ", "W Position Gq", "W Velocity Gq̇", "KS Stat"]
    df["Gap Score"] = df[metric_cols].apply(
        lambda col: (col - col.min()) / (col.max() - col.min() + 1e-9)
    ).mean(axis=1).round(4)

    df["Needs Fix?"] = df["Gap Score"].apply(
        lambda x: "🔴 High" if x > 0.7 else ("🟡 Medium" if x > 0.4 else "🟢 Low")
    )

    temporal_row = {
        "Joint": "── TEMPORAL AVG ──",
        "Mean Gap Gμ": "",
        "Std Gap": "",
        "W Position Gq": "",
        "W Velocity Gq̇": f"Gain={temporal_results['Ggain']:+.4f}",
        "Vel Mag Gap Δμv": f"Lag={temporal_results['Glag']:+.4f}",
        "Jerk Gap Gjerk": "",
        "Sim Delay (f)": round(np.mean(temporal_results["sim_delays"]), 2),
        "Real Delay (f)": round(np.mean(temporal_results["real_delays"]), 2),
        "KS Stat": "",
        "KS p-value": "",
        "Gap Score": "",
        "Needs Fix?": "",
    }
    df = pd.concat([df, pd.DataFrame([temporal_row])], ignore_index=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_joint_distributions(joint_results, save_path):
    """Violin + KDE plot for each joint position distribution."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Joint Position Distributions: Sim vs Real\n(Section 4.2 – Distribution Gap)",
                 fontsize=14, fontweight="bold")

    for ax, r in zip(axes.flat, joint_results):
        sq, rq = r["_sim_q"], r["_real_q"]
        df_sim = pd.DataFrame({"domain": "sim", "value": sq})
        df_real = pd.DataFrame({"domain": "real", "value": rq})
        data = pd.concat([df_sim, df_real], ignore_index=True)
        sns.violinplot(data=data, x="domain", y="value", palette=PALETTE,
                       inner="quartile", ax=ax)
        ax.set_title(f"{r['joint']}\nW={r['W_position_Gq']:.3f}  KS={r['KS_stat']:.3f}",
                     fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("joint position (rad or deg)")

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_velocity_distributions(joint_results, save_path):
    """Velocity distribution comparison per joint."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Joint Velocity Distributions: Sim vs Real\n(Section 4.2 – Velocity Gap)",
                 fontsize=14, fontweight="bold")

    for ax, r in zip(axes.flat, joint_results):
        sv, rv = r["_sim_v"], r["_real_v"]
        ax.hist(sv, bins=50, alpha=0.55, color=PALETTE["sim"],  label="sim",  density=True)
        ax.hist(rv, bins=50, alpha=0.55, color=PALETTE["real"], label="real", density=True)
        ax.set_title(f"{r['joint']}\nW_vel={r['W_velocity_Gq̇']:.3f}", fontsize=10)
        ax.set_xlabel("velocity")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_gap_heatmap(joint_results, save_path):
    """Heatmap of all gap metrics across joints."""
    metric_names = ["Mean Gap Gμ", "W Position Gq", "W Velocity Gq̇", 
                    "|Vel Mag Δμv|", "Jerk Gap |Gjerk|", "KS Stat"]
    matrix = []
    for r in joint_results:
        matrix.append([
            r["mean_gap_Gμ"],
            r["W_position_Gq"],
            r["W_velocity_Gq̇"],
            abs(r["vel_mag_gap_Δμv"]),
            abs(r["jerk_gap_Gjerk"]),
            r["KS_stat"],
        ])
    arr = np.array(matrix)  # (N_joints, N_metrics)
    # Normalise each metric 0→1
    arr_norm = (arr - arr.min(0)) / (arr.max(0) - arr.min(0) + 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Joint-Space Gap Heatmap – Raw & Normalised\n(Section 4.2)",
                 fontsize=14, fontweight="bold")

    sns.heatmap(arr, xticklabels=metric_names,
                yticklabels=JOINT_NAMES[:N_JOINTS],
                annot=True, fmt=".3f", cmap="YlOrRd", ax=ax1)
    ax1.set_title("Raw values")

    sns.heatmap(arr_norm, xticklabels=metric_names,
                yticklabels=JOINT_NAMES[:N_JOINTS],
                annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2)
    ax2.set_title("Normalised (0=best, 1=worst)")

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_gap_bar(joint_results, save_path):
    """Bar chart: Wasserstein position gap per joint."""
    names = [r["joint"] for r in joint_results]
    w_pos = [r["W_position_Gq"]  for r in joint_results]
    w_vel = [r["W_velocity_Gq̇"]  for r in joint_results]
    jerk  = [abs(r["jerk_gap_Gjerk"]) for r in joint_results]
    ks    = [r["KS_stat"]         for r in joint_results]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 1.5*width, w_pos, width, label="W Position Gq",    color="#4C72B0")
    ax.bar(x - 0.5*width, w_vel, width, label="W Velocity Gq̇",    color="#55A868")
    ax.bar(x + 0.5*width, jerk,  width, label="|Jerk Gap Gjerk|", color="#DD8452")
    ax.bar(x + 1.5*width, ks,    width, label="KS Stat",          color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Gap magnitude")
    ax.set_title("Per-Joint Gap Metrics – Higher = Larger Sim-Real Gap\n(Section 4.2)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_temporal_gap(temporal_results, save_path):
    """Temporal gap: delay distribution + gain distribution + lag comparison."""
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle("Temporal Response Gap: Sim vs Real\n(Section 4.3)", 
                 fontsize=14, fontweight="bold")

    # ── 1. Response delay distribution ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    sd, rd = temporal_results["sim_delays"], temporal_results["real_delays"]
    ax1.hist(sd, bins=15, alpha=0.6, color=PALETTE["sim"],  label=f"sim  μ={sd.mean():.1f}")
    ax1.hist(rd, bins=15, alpha=0.6, color=PALETTE["real"], label=f"real μ={rd.mean():.1f}")
    ax1.axvline(sd.mean(), color=PALETTE["sim"],  linestyle="--", linewidth=1.5)
    ax1.axvline(rd.mean(), color=PALETTE["real"], linestyle="--", linewidth=1.5)
    ax1.set_title(f"Response Delay τ*\nGdelay = {temporal_results['Gdelay']:+.2f} frames")
    ax1.set_xlabel("frames")
    ax1.legend(fontsize=9)

    # ── 2. Action-response gain ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    sg, rg = temporal_results["sim_gains"], temporal_results["real_gains"]
    df_sg = pd.DataFrame({"domain": "sim", "value": sg})
    df_rg = pd.DataFrame({"domain": "real", "value": rg})
    data = pd.concat([df_sg, df_rg], ignore_index=True)
    sns.boxplot(data=data, x="domain", y="value", palette=PALETTE, ax=ax2)
    ax2.set_title(f"Action-Response Gain\nGgain = {temporal_results['Ggain']:+.4f}")
    ax2.set_ylabel("gain  ||Δq|| / ||Δa||")

    # ── 3. Tracking lag ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    sl, rl = temporal_results["sim_lags"], temporal_results["real_lags"]
    df_sl = pd.DataFrame({"domain": "sim", "value": sl})
    df_rl = pd.DataFrame({"domain": "real", "value": rl})
    data2 = pd.concat([df_sl, df_rl], ignore_index=True)
    sns.boxplot(data=data2, x="domain", y="value", palette=PALETTE, ax=ax3)
    ax3.set_title(f"Tracking Lag (mean |a-q|)\nGlag = {temporal_results['Glag']:+.4f}")
    ax3.set_ylabel("mean abs error")

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_jerk_comparison(joint_results, save_path):
    """Jerk magnitude per joint – box comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Jerk (Smoothness) Comparison: Sim vs Real\n(Section 4.2 – Jerk Gap)",
                 fontsize=14, fontweight="bold")

    for ax, r in zip(axes.flat, joint_results):
        sj, rj = r["_sim_j"], r["_real_j"]
        df_sim = pd.DataFrame({"domain": "sim", "|jerk|": np.abs(sj)}) if len(sj) > 0 else pd.DataFrame()
        df_real = pd.DataFrame({"domain": "real", "|jerk|": np.abs(rj)}) if len(rj) > 0 else pd.DataFrame()
        data = pd.concat([df_sim, df_real], ignore_index=True)
        sns.boxplot(data=data, x="domain", y="|jerk|", palette=PALETTE,
                    showfliers=False, ax=ax)
        ax.set_title(f"{r['joint']}\nGjerk={r['jerk_gap_Gjerk']:+.3f}", fontsize=10)
        ax.set_xlabel("")

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_mean_trajectory_comparison(sim_df, real_df, state_prefix, save_path):
    """
    Mean ± std trajectory across all episodes for each joint.
    Tells us whether sim and real 'visit' the same joint angles over time.
    """
    def mean_traj(df, prefix, T_ref=150):
        eps_arrays = []
        ep_col = "episode_index" if "episode_index" in df.columns else None
        if ep_col:
            for _, g in df.groupby(ep_col):
                g = g.sort_values("frame_index") if "frame_index" in g.columns else g
                arr = get_joint_arrays(g, prefix)
                T = len(arr)
                if T > 1:
                    t_old = np.linspace(0, 1, T)
                    t_new = np.linspace(0, 1, T_ref)
                    eps_arrays.append(interp1d(t_old, arr, axis=0, kind='linear')(t_new))
        else:
            arr = get_joint_arrays(df, prefix)
            T = len(arr)
            if T > 1:
                t_old = np.linspace(0, 1, T)
                t_new = np.linspace(0, 1, T_ref)
                eps_arrays.append(interp1d(t_old, arr, axis=0, kind='linear')(t_new))
                
        if not eps_arrays:
            return None, None
        stack = np.array(eps_arrays)                          # (E, T, J)
        mu  = np.nanmean(stack, axis=0)                       # (T, J)
        sig = np.nanstd(stack,  axis=0)
        return mu, sig

    sim_mu,  sim_sig  = mean_traj(sim_df,  state_prefix)
    real_mu, real_sig = mean_traj(real_df, state_prefix)

    if sim_mu is None or real_mu is None:
        print("  [SKIP] Not enough episodes for trajectory comparison")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Mean Joint Trajectory: Sim vs Real (mean ± 1σ across episodes)\n"
                 "Shows where in joint-space the two domains diverge",
                 fontsize=13, fontweight="bold")

    for idx, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES[:N_JOINTS])):
        sm, ss = sim_mu[:, idx],  sim_sig[:, idx]
        rm, rs = real_mu[:, idx], real_sig[:, idx]
        # X axis is percentage of episode progress
        t_progress = np.linspace(0, 100, len(sm))

        ax.plot(t_progress, sm, color=PALETTE["sim"],  label="sim",  linewidth=1.8)
        ax.fill_between(t_progress, sm-ss, sm+ss, alpha=0.25, color=PALETTE["sim"])
        ax.plot(t_progress, rm, color=PALETTE["real"], label="real", linewidth=1.8, linestyle="--")
        ax.fill_between(t_progress, rm-rs, rm+rs, alpha=0.25, color=PALETTE["real"])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("episode progress (%)")
        ax.set_ylabel("joint pos")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_composite_radar(joint_results, save_path):
    """Radar (spider) chart of normalised gap score per joint."""
    from matplotlib.patches import FancyArrowPatch

    labels  = ["W-Position", "W-Velocity", "Mean Bias", "|Vel Mag Gap|", "|Jerk Gap|", "KS Stat"]

    data = []
    for r in joint_results:
        data.append([r["W_position_Gq"], r["W_velocity_Gq̇"], r["mean_gap_Gμ"],
                     abs(r["vel_mag_gap_Δμv"]), abs(r["jerk_gap_Gjerk"]), r["KS_stat"]])
    arr = np.array(data)
    arr_n = (arr - arr.min(0)) / (arr.max(0) - arr.min(0) + 1e-9)

    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                              subplot_kw=dict(polar=True))
    fig.suptitle("Gap Radar per Joint (normalised metrics, 0=best)\n(Section 4.2 composite)",
                 fontsize=13, fontweight="bold")

    colors = plt.cm.tab10.colors

    for ax, vals_n, name, col in zip(axes.flat, arr_n, JOINT_NAMES[:N_JOINTS], colors):
        vals = vals_n.tolist() + [vals_n[0]]
        ax.plot(angles, vals, color=col, linewidth=2)
        ax.fill(angles, vals, color=col, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(name, size=11, pad=12)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], size=6)

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 7. INTERPRETABILITY REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_interpretation(joint_results, temporal_results, summary_df):
    print("\n" + "═"*62)
    print("  INTERPRETATION & RECOMMENDATIONS")
    print("═"*62)

    # Most problematic joints
    sorted_jr = sorted(joint_results, key=lambda r: r["W_position_Gq"], reverse=True)
    top = sorted_jr[:3]
    print("\n▸ TOP-3 JOINTS WITH LARGEST POSITION GAP (W distance):")
    for r in top:
        print(f"   {r['joint']:15s}  Gq={r['W_position_Gq']:.4f}  "
              f"KS={r['KS_stat']:.3f}  p={r['KS_pvalue']:.4f}")

    sig = [r for r in joint_results if r["KS_pvalue"] < 0.05]
    print(f"\n▸ {len(sig)}/{len(joint_results)} joints have STATISTICALLY DIFFERENT distributions "
          f"(KS p<0.05)")
    for r in sig:
        print(f"   {r['joint']:15s}  KS={r['KS_stat']:.3f}  p={r['KS_pvalue']:.4g}")

    # Velocity / jerk issues
    noisy = [r for r in joint_results if abs(r["jerk_gap_Gjerk"]) > 0.5 * max(
        abs(x["jerk_gap_Gjerk"]) for x in joint_results)]
    if noisy:
        print("\n▸ JOINTS WITH HIGH JERK GAP (real is rougher / smoother than sim):")
        for r in noisy:
            dir_ = "real jerkier" if r["jerk_gap_Gjerk"] > 0 else "sim jerkier"
            print(f"   {r['joint']:15s}  Gjerk={r['jerk_gap_Gjerk']:+.4f}  ({dir_})")
            
    # Velocity magnitude issues (Real moving slower than Sim)
    slow_joints = [r for r in joint_results if r["vel_mag_gap_Δμv"] < -0.05]
    if slow_joints:
        print("\n▸ JOINTS WITH SIGNIFICANT VELOCITY LOSS (Real moves slower):")
        for r in slow_joints:
            print(f"   {r['joint']:15s}  Δμv={r['vel_mag_gap_Δμv']:+.4f} rad/s")

    # Temporal
    print(f"\n▸ TEMPORAL GAP SUMMARY:")
    Gd = temporal_results["Gdelay"]
    Gg = temporal_results["Ggain"]
    Gl = temporal_results["Glag"]
    print(f"   Gdelay = {Gd:+.2f} frames  → real {'SLOWER' if Gd>0 else 'FASTER'} than sim")
    print(f"   Ggain  = {Gg:+.4f}        → real response {'WEAKER' if Gg<0 else 'STRONGER'}")
    print(f"   Glag   = {Gl:+.4f}        → real tracking {'WORSE' if Gl>0 else 'BETTER'}")

    print("\n▸ RECOMMENDATIONS:")
    if any(r["KS_pvalue"] < 0.05 for r in joint_results):
        print("   1. Joint-space distributions differ significantly.")
        print("      → Apply motor randomisation / system identification in MuJoCo.")
    if Gg < -0.05:
        print("   2. Real robot gain is lower (responds weaker).")
        print("      → Tune actuator gain in MuJoCo or add residual correction.")
    if Gd > 2:
        print("   3. Real robot has notable delay.")
        print("      → Add actuator delay model in simulation.")
    vel_gap_max = max(r["W_velocity_Gq̇"] for r in joint_results)
    if vel_gap_max > 0.1:
        print("   4. Velocity distributions differ – policy may act too fast in real.")
        print("      → Consider velocity smoothing or domain randomisation on speed.")

    print("\n▸ OVERALL TRANSFERABILITY:")
    n_high = sum(1 for r in joint_results if r["W_position_Gq"] > 0.2 or r["KS_stat"] > 0.3)
    if n_high == 0:
        verdict = "✅ Good – sim and real appear similar. Direct transfer likely feasible."
    elif n_high <= 2:
        verdict = ("🟡 Moderate – a few joints need attention. "
                   "Fine-tuning with small real dataset recommended.")
    else:
        verdict = ("🔴 Large gap – significant differences. "
                   "Domain randomisation + system ID strongly recommended before transfer.")
    print(f"   {verdict}\n")
    print("═"*62)


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global SIM_REPO, REAL_REPO, OUT_DIR
    args = parse_args()
    SIM_REPO = args.sim_dataset
    REAL_REPO = args.real_dataset
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 62)
    print("  SO101-SimRealGap: Joint-Space & Temporal Gap Analysis")
    print("  Sections 4.2 & 4.3  |  SO-101, 6 joints")
    print("=" * 62)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading datasets …")
    sim_df, real_df = load_or_simulate()
    
    # Gọi hàm kiểm tra và xuất info ra file
    check_and_export_dataset_info(sim_df, real_df, OUT_DIR)

    # ── Detect column names ────────────────────────────────────────────────
    print("\n[2/6] Detecting column prefixes …")
    for df, name in [(sim_df, "SIM"), (real_df, "REAL")]:
        state_cols  = [c for c in df.columns if "state" in c.lower() and
                       c.split("_")[-1].isdigit()]
        action_cols = [c for c in df.columns if "action" in c.lower() and
                       c.split("_")[-1].isdigit()]
        print(f"  {name}: state_cols={state_cols[:3]}…  action_cols={action_cols[:3]}…")

    # Use first detected prefix
    state_prefix  = next((c.rsplit("_", 1)[0] + "_"
                          for c in sim_df.columns
                          if "state" in c.lower() and c.split("_")[-1].isdigit()), "observation.state_")
    action_prefix = next((c.rsplit("_", 1)[0] + "_"
                          for c in sim_df.columns
                          if "action" in c.lower() and c.split("_")[-1].isdigit()), "action_")

    print(f"  Using state_prefix='{state_prefix}'  action_prefix='{action_prefix}'")

    # ── Section 4.2 ────────────────────────────────────────────────────────
    print("\n[3/6] Computing joint-space gap (Sec 4.2) …")
    joint_results = compute_joint_space_gap(sim_df, real_df, state_prefix, action_prefix)

    # ── Section 4.3 ────────────────────────────────────────────────────────
    print("\n[4/6] Computing temporal response gap (Sec 4.3) …")
    temporal_results = compute_temporal_gap(sim_df, real_df, state_prefix, action_prefix)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n[5/6] Building summary table …")
    summary_df = build_summary_table(joint_results, temporal_results)
    csv_path = os.path.join(OUT_DIR, "gap_summary_table.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print("\n" + summary_df.to_string(index=False))

    # ── Interpretability ───────────────────────────────────────────────────
    print_interpretation(joint_results, temporal_results, summary_df)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[6/6] Generating plots …")
    plot_joint_distributions(
        joint_results,
        os.path.join(OUT_DIR, "01_joint_position_distributions.png"))
    plot_velocity_distributions(
        joint_results,
        os.path.join(OUT_DIR, "02_joint_velocity_distributions.png"))
    plot_gap_heatmap(
        joint_results,
        os.path.join(OUT_DIR, "03_gap_heatmap.png"))
    plot_gap_bar(
        joint_results,
        os.path.join(OUT_DIR, "04_gap_bar_chart.png"))
    plot_jerk_comparison(
        joint_results,
        os.path.join(OUT_DIR, "05_jerk_comparison.png"))
    plot_temporal_gap(
        temporal_results,
        os.path.join(OUT_DIR, "06_temporal_gap.png"))
    plot_mean_trajectory_comparison(
        sim_df, real_df, state_prefix,
        os.path.join(OUT_DIR, "07_mean_trajectory_comparison.png"))
    plot_composite_radar(
        joint_results,
        os.path.join(OUT_DIR, "08_radar_chart.png"))

    print(f"\n✅ All outputs saved to ./{OUT_DIR}/")
    print("   Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"     {f}")


if __name__ == "__main__":
    main()
