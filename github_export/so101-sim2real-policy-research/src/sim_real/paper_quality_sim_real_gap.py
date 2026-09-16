"""
SO-101 Sim-Real Gap Analysis Pipeline
======================================
Phân tích và visualize sim-real gap cho robot SO-101 6-DOF
theo tiêu chí mục 4.2 của paper benchmark.

Datasets (LeRobot v2 format - Hugging Face):
  SIM : ngocthuong2212/so101_pick_red_no_ep100_left
  REAL: vasco281204/pick_red_fixed
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from scipy import signal, stats
from datasets import load_dataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SIM_DATASET  = "ngocthuong2212/so101_pick_red_no_ep100_left"
REAL_DATASET = "vasco281204/pick_red_fixed"

OUTPUT_DIR   = "sim_real_gap_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper"
]
N_JOINTS = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-quality SO-101 sim-real gap figures and tables."
    )
    parser.add_argument("--sim-dataset", default=SIM_DATASET)
    parser.add_argument("--real-dataset", default=REAL_DATASET)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()

# Paper-quality style
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman"],
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

COLORS = {
    "sim":    "#2196F3",   # blue
    "real":   "#E53935",   # red
    "gap":    "#FF9800",   # orange
    "grid":   "#f0f0f0",
    "accent": "#4CAF50",
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_lerobot_v2(repo_id: str, split: str = "train") -> pd.DataFrame:
    """Load LeRobot v2 dataset từ Hugging Face (parquet)."""
    print(f"  Loading: {repo_id} ...")
    ds = load_dataset(repo_id, split=split, data_files="data/**/*.parquet")
    df = ds.to_pandas()
    print(f"    → {len(df):,} rows | columns: {list(df.columns)}")
    return df


def extract_joint_columns(df: pd.DataFrame) -> list:
    """Tìm các cột observation.state hoặc action tương ứng joint."""
    candidates = [c for c in df.columns if "state" in c.lower() or "observation" in c.lower()]
    if not candidates:
        candidates = [c for c in df.columns if "action" in c.lower()]
    return candidates


def unpack_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """Unpack một cột chứa list/array thành ma trận (N, D)."""
    sample = df[col].iloc[0]
    if isinstance(sample, (list, np.ndarray)):
        arr = np.stack(df[col].values)
    else:
        arr = df[col].values.reshape(-1, 1)
    return arr.astype(float)


def get_joint_data(df: pd.DataFrame) -> np.ndarray:
    """Trả về ma trận (N, 6) joint positions."""
    # Ưu tiên observation.state
    for col in ["observation.state", "obs_state", "state"]:
        if col in df.columns:
            arr = unpack_vector_column(df, col)
            if arr.shape[1] >= N_JOINTS:
                return arr[:, :N_JOINTS]
    # Fallback: action
    for col in ["action", "actions"]:
        if col in df.columns:
            arr = unpack_vector_column(df, col)
            if arr.shape[1] >= N_JOINTS:
                return arr[:, :N_JOINTS]
    raise ValueError(f"Không tìm thấy cột joint data. Columns: {list(df.columns)}")


def align_episodes(sim_df: pd.DataFrame, real_df: pd.DataFrame):
    """
    Căn chỉnh sim vs real theo episode.
    Trả về list of (sim_traj, real_traj) đã cắt cùng độ dài.
    """
    sim_ep_col  = next((c for c in sim_df.columns  if "episode" in c.lower()), None)
    real_ep_col = next((c for c in real_df.columns if "episode" in c.lower()), None)

    if sim_ep_col and real_ep_col:
        sim_eps  = sim_df[sim_ep_col].unique()
        real_eps = real_df[real_ep_col].unique()
        n_pair   = min(len(sim_eps), len(real_eps))
        print(f"    Số episodes: sim={len(sim_eps)}, real={len(real_eps)}, paired={n_pair}")
        pairs = []
        for i in range(n_pair):
            s = get_joint_data(sim_df[sim_df[sim_ep_col] == sim_eps[i]])
            r = get_joint_data(real_df[real_df[real_ep_col] == real_eps[i]])
            L = min(len(s), len(r))
            pairs.append((s[:L], r[:L]))
        return pairs
    else:
        # Coi toàn bộ dataset như 1 episode
        s = get_joint_data(sim_df)
        r = get_joint_data(real_df)
        L = min(len(s), len(r))
        return [(s[:L], r[:L])]


# ─────────────────────────────────────────────
# METRICS (Section 4.2)
# ─────────────────────────────────────────────

def compute_joint_metrics(pairs: list) -> dict:
    """
    Joint-space error metrics trên tất cả paired episodes.
    Returns dict với keys: mae, rmse, max_err, std_err  → shape (N_JOINTS,)
    """
    all_diffs = []
    for s, r in pairs:
        all_diffs.append(np.abs(s - r))          # (T, 6)
    diffs = np.concatenate(all_diffs, axis=0)     # (T_total, 6)

    sq_diffs = np.concatenate([((s-r)**2) for s,r in pairs], axis=0)

    return {
        "mae":      diffs.mean(axis=0),
        "rmse":     np.sqrt(sq_diffs.mean(axis=0)),
        "max_err":  diffs.max(axis=0),
        "std_err":  diffs.std(axis=0),
        "per_step": diffs,          # (T_total, 6) for distribution plots
    }


def compute_control_delay(pairs: list) -> np.ndarray:
    """
    Ước lượng control delay giữa sim và real bằng cross-correlation.
    Trả về array delay (frames) cho từng joint, mỗi episode.
    """
    delays = []
    for s, r in pairs:
        ep_delays = []
        for j in range(N_JOINTS):
            xs = s[:, j] - s[:, j].mean()
            xr = r[:, j] - r[:, j].mean()
            if xs.std() < 1e-6 or xr.std() < 1e-6:
                ep_delays.append(0)
                continue
            corr = signal.correlate(xr, xs, mode="full")
            lags  = signal.correlation_lags(len(xr), len(xs), mode="full")
            delay = lags[np.argmax(corr)]
            ep_delays.append(int(delay))
        delays.append(ep_delays)
    return np.array(delays)   # (n_episodes, 6)


def compute_trajectory_smoothness(pairs: list) -> dict:
    """
    Đo độ mượt (jerk proxy) của sim vs real trajectories.
    """
    sim_jerk, real_jerk = [], []
    for s, r in pairs:
        sim_jerk.append(np.diff(s, n=3, axis=0).std(axis=0))
        real_jerk.append(np.diff(r, n=3, axis=0).std(axis=0))
    return {
        "sim":  np.mean(sim_jerk,  axis=0),
        "real": np.mean(real_jerk, axis=0),
    }


def compute_dtw_distance(s: np.ndarray, r: np.ndarray, joint_idx: int) -> float:
    """Simple DTW distance cho 1 joint."""
    xs = s[:, joint_idx]
    xr = r[:, joint_idx]
    n, m = len(xs), len(xr)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(xs[i-1] - xr[j-1])
            dtw[i, j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    # Chuẩn hoá trên cạnh dài nhất để trả về xấp xỉ khoảng cách (Độ)
    return dtw[n, m] / max(n, m)

def compute_all_dtw(pairs: list) -> np.ndarray:
    """Tính DTW Distance cho tất cả các episode và các joint."""
    import sys
    dtw_results = []
    print("    Đang tính toán DTW (có thể mất 1-2 phút do Python loop)...")
    for i, (s, r) in enumerate(pairs):
        sys.stdout.write(f"\r      Đang xử lý Episode {i+1}/{len(pairs)}")
        sys.stdout.flush()
        ep_dtw = []
        for j in range(N_JOINTS):
            ep_dtw.append(compute_dtw_distance(s, r, j))
        dtw_results.append(ep_dtw)
    print()
    return np.array(dtw_results)


# ─────────────────────────────────────────────
# FIGURE 1: Shape Similarity (DTW Distance)
# ─────────────────────────────────────────────

def fig_dtw_overview(dtw_arr: np.ndarray, save_path: str):
    mean_dtw = dtw_arr.mean(axis=0)
    std_dtw = dtw_arr.std(axis=0)

    x = np.arange(N_JOINTS)
    w = 0.45

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Figure 1 — Trajectory Shape Similarity: DTW Distance", fontsize=14, y=1.02)

    bars = ax.bar(x, mean_dtw, w, yerr=std_dtw, capsize=5, color=COLORS["gap"], alpha=0.85, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(JOINT_NAMES, rotation=20, ha="right")
    ax.set_ylabel("DTW Distance (~ degrees)")
    ax.set_title("Lower value = Sim and Real trace similar paths (independent of speed)")
    ax.yaxis.grid(True, color=COLORS["grid"], zorder=0)

    for bar, v in zip(bars, mean_dtw):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02 * mean_dtw.max(),
                f"{v:.2f}°", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 2: Trajectory Comparison (first episode)
# ─────────────────────────────────────────────

def fig_trajectory_comparison(pairs: list, save_path: str, n_show: int = 3):
    s, r = pairs[0]
    t_s = np.arange(len(s))
    t_r = np.arange(len(r))
    show_joints = list(range(min(n_show, N_JOINTS)))

    fig, axes = plt.subplots(len(show_joints), 1, figsize=(12, 3.5 * len(show_joints)), sharex=True)
    if len(show_joints) == 1:
        axes = [axes]

    fig.suptitle("Figure 2 — Unpaired Trajectories: Differences in Speed & Length", fontsize=14, y=1.01)

    for idx, j in enumerate(show_joints):
        ax = axes[idx]
        ax.plot(t_s, s[:, j], color=COLORS["sim"],  lw=2.0, label=f"Simulation ({len(s)} frames)", alpha=0.9)
        ax.plot(t_r, r[:, j], color=COLORS["real"], lw=2.0, label=f"Real ({len(r)} frames)",       alpha=0.9, linestyle="--")

        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(degrees)", fontsize=10)
        ax.yaxis.grid(True, color=COLORS["grid"], zorder=0)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Timestep (frame)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 5: Trajectory Smoothness (Jerk proxy)
# ─────────────────────────────────────────────

def fig_smoothness(smooth: dict, save_path: str):
    sim_jerk  = smooth["sim"]
    real_jerk = smooth["real"]
    x = np.arange(N_JOINTS)
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Figure 5 — Trajectory Smoothness: Jerk Proxy (σ of 3rd difference)",
                 fontsize=14)
    b1 = ax.bar(x - w/2, sim_jerk,  w, label="Simulation", color=COLORS["sim"],  alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, real_jerk, w, label="Real",       color=COLORS["real"], alpha=0.85, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(JOINT_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Jerk Proxy (°/frame²  std)")
    ax.set_title("Higher value = less smooth trajectory")
    ax.yaxis.grid(True, color=COLORS["grid"], zorder=0)
    ax.legend()
    for bars in [b1, b2]:
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h * 1.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 6: Summary Radar Chart
# ─────────────────────────────────────────────

def fig_radar_summary(dtw_arr: np.ndarray, smooth: dict, save_path: str):
    """Radar chart tóm tắt các gap metrics đã normalize."""
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

    categories = JOINT_NAMES
    N = N_JOINTS

    # Normalize mỗi metric về [0, 1]
    def norm(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 1e-9 else arr * 0

    dtw_n   = norm(dtw_arr.mean(axis=0))
    sim_jerk_n = norm(smooth["sim"])
    real_jerk_n = norm(smooth["real"])
    jerk_diff = np.abs(smooth["sim"] - smooth["real"])
    jerk_n  = norm(jerk_diff)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def close(arr): return np.concatenate([arr, arr[:1]])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Figure 6 — Sim-Real Gap Radar Summary (Normalized per Metric)",
                 fontsize=14, y=1.01)

    datasets_radar = [
        (dtw_n,   "DTW Similarity (normalized)", COLORS["gap"],    axes[0][0]),
        (sim_jerk_n,  "Sim Jerk (normalized)",   COLORS["sim"],    axes[0][1]),
        (real_jerk_n, "Real Jerk (normalized)",  COLORS["real"],   axes[1][0]),
        (jerk_n,      "|ΔJerk| (normalized)",    COLORS["accent"], axes[1][1]),
    ]

    for vals, title, color, ax in datasets_radar:
        v = close(vals)
        ax.plot(angles, v, color=color, lw=2.2, zorder=3)
        ax.fill(angles, v, color=color, alpha=0.25, zorder=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25","0.50","0.75","1.0"], fontsize=7, color="gray")
        ax.set_title(title, pad=16, fontsize=11, fontweight="bold", color=color)
        ax.grid(color="#dddddd", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# FIGURE 7: Per-Episode Gap Consistency
# ─────────────────────────────────────────────

def fig_episode_consistency(dtw_arr: np.ndarray, save_path: str):
    """DTW per episode per joint → đánh giá consistency của quỹ đạo qua các lần."""
    n_ep = len(dtw_arr)

    fig, ax = plt.subplots(figsize=(max(8, n_ep * 0.6 + 3), 5))
    fig.suptitle("Figure 7 — Shape Similarity Consistency (DTW) per Episode", fontsize=14)

    x = np.arange(n_ep)
    bottom = np.zeros(n_ep)
    cmap = plt.cm.get_cmap("tab10", N_JOINTS)

    for j in range(N_JOINTS):
        ax.bar(x, dtw_arr[:, j], bottom=bottom,
               label=JOINT_NAMES[j], color=cmap(j), alpha=0.82, zorder=3)
        bottom += dtw_arr[:, j]

    ax.set_xlabel("Episode index")
    ax.set_ylabel("Stacked DTW Distance (degrees)")
    ax.set_title("Lower & consistent bars = more transferable sim policy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ep{i}" for i in x], fontsize=8, rotation=45)
    ax.yaxis.grid(True, color=COLORS["grid"], zorder=0)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# SUMMARY TABLE (CSV + LaTeX snippet)
# ─────────────────────────────────────────────

def save_summary_table(dtw_arr: np.ndarray, smooth: dict, pairs: list, out_dir: str):
    rows = []
    dtw_mean = dtw_arr.mean(axis=0)
    
    for j, name in enumerate(JOINT_NAMES):
        rows.append({
            "Joint":          name,
            "DTW Error (°)":  round(float(dtw_mean[j]),  3),
            "Jerk Sim":       round(float(smooth["sim"][j]), 4),
            "Jerk Real":      round(float(smooth["real"][j]), 4),
            "ΔJerk":          round(float(abs(smooth["sim"][j] - smooth["real"][j])), 4),
        })
    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "sim_real_gap_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV: {csv_path}")

    # LaTeX table snippet
    latex_path = os.path.join(out_dir, "sim_real_gap_table.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated by sim_real_gap_analysis.py\n")
        f.write("% Paste into your paper's table section\n\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Sim-Real Gap Metrics for SO-101 (pick-red-block task)}\n")
        f.write("\\label{tab:sim_real_gap}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Joint & DTW Error (°) & Jerk Sim & Jerk Real & $\\Delta$Jerk \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['Joint']} & {row['DTW Error (°)']} & {row['Jerk Sim']} & "
                f"{row['Jerk Real']} & {row['ΔJerk']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  ✓ Saved LaTeX: {latex_path}")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    global SIM_DATASET, REAL_DATASET, OUTPUT_DIR
    args = parse_args()
    SIM_DATASET = args.sim_dataset
    REAL_DATASET = args.real_dataset
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SO-101 Sim-Real Gap Analysis Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading datasets ...")
    sim_df  = load_lerobot_v2(SIM_DATASET)
    real_df = load_lerobot_v2(REAL_DATASET)

    # 2. Pair episodes
    print("\n[2/5] Aligning episodes ...")
    pairs = align_episodes(sim_df, real_df)
    print(f"    → {len(pairs)} paired task(s)")
    print(f"    → Example Ep 0: Sim length={len(pairs[0][0])} | Real length={len(pairs[0][1])}")

    # 3. Compute metrics
    print("\n[3/5] Computing metrics ...")
    dtw_arr   = compute_all_dtw(pairs)
    smooth    = compute_trajectory_smoothness(pairs)

    print(f"    DTW (°): { {n: f'{v:.3f}' for n,v in zip(JOINT_NAMES, dtw_arr.mean(0))} }")

    # 4. Generate figures
    print(f"\n[4/5] Generating figures → {OUTPUT_DIR}/")
    fig_dtw_overview(dtw_arr,
        os.path.join(OUTPUT_DIR, "fig1_joint_error_overview.png"))
    fig_trajectory_comparison(pairs,
        os.path.join(OUTPUT_DIR, "fig2_trajectory_comparison.png"))
    fig_smoothness(smooth,
        os.path.join(OUTPUT_DIR, "fig5_smoothness.png"))
    fig_radar_summary(dtw_arr, smooth,
        os.path.join(OUTPUT_DIR, "fig6_radar_summary.png"))
    if len(pairs) > 1:
        fig_episode_consistency(dtw_arr,
            os.path.join(OUTPUT_DIR, "fig7_episode_consistency.png"))
    else:
        print("  ⚠ Chỉ có 1 episode, bỏ qua Fig 7 (episode consistency)")

    # 5. Save tables
    print("\n[5/5] Saving summary tables ...")
    df_summary = save_summary_table(dtw_arr, smooth, pairs, OUTPUT_DIR)
    print("\n  Summary Table:")
    print(df_summary.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  ✅ Xong! Tất cả output tại: ./{OUTPUT_DIR}/")
    print(f"     Figures : fig1, 2, 5, 6, 7 (.png, 300 DPI)")
    print(f"     CSV     : sim_real_gap_metrics.csv")
    print(f"     LaTeX   : sim_real_gap_table.tex")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
