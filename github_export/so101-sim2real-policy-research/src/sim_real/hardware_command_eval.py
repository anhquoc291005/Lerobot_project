"""
Comprehensive Sim-to-Real Hardware & Command Evaluation
Đánh giá khách quan 2 mảng:
1. Độ ổn định phần cứng (Hardware Stability): Jerk, Vận tốc, Không gian hoạt động, Phân phối.
2. Chất lượng nhận lệnh (Command Execution): Độ trễ pha, Sai số bám quỹ đạo (Action vs State).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import wasserstein_distance
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import warnings
warnings.filterwarnings("ignore")

# ================= CẤU HÌNH =================
SIM_DATASET = "ngocthuong2212/so101_pick_red_no_ep100_left"
REAL_DATASET = "vasco281204/pick_red_fixed"
OUTPUT_DIR = "hardware_command_eval_results"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
N_JOINTS = 6
# ============================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate hardware stability and command tracking for SO-101 sim/real datasets."
    )
    parser.add_argument("--sim-dataset", default=SIM_DATASET)
    parser.add_argument("--real-dataset", default=REAL_DATASET)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()

def apply_lowpass_filter(data, cutoff_freq, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)

def compute_command_metrics(state, action):
    """Tính delay và tracking error (trước và sau khi bù trễ) cho 1 episode."""
    delays = []
    raw_errors = []
    aligned_errors = []
    
    for j in range(N_JOINTS):
        s = state[:, j]
        a = action[:, j]
        
        raw_err = np.abs(s - a).mean()
        raw_errors.append(raw_err)
        
        # Bỏ qua các episode robot đứng im (flat line) để không làm nhiễu delay
        if s.std() < 0.02 or a.std() < 0.02:
            delays.append(np.nan)
            aligned_errors.append(raw_err)
            continue
            
        s_norm = s - s.mean()
        a_norm = a - a.mean()
        
        corr = signal.correlate(s_norm, a_norm, mode="full")
        lags = signal.correlation_lags(len(s), len(a), mode="full")
        delay = int(lags[np.argmax(corr)])
        
        # Giới hạn delay hợp lý [-30, 30] frames để loại bỏ nhiễu chéo (-17.5)
        if delay > 30 or delay < -30:
            delay = np.nan
            aligned_err = raw_err
        else:
            if delay > 0:
                s_align = s[delay:]
                a_align = a[:-delay]
            elif delay < 0:
                s_align = s[:delay]
                a_align = a[-delay:]
            else:
                s_align = s
                a_align = a
                
            aligned_err = np.abs(s_align - a_align).mean()
            
        delays.append(delay)
        aligned_errors.append(aligned_err)
        
    return delays, raw_errors, aligned_errors

def process_dataset(dataset_id):
    print(f"Loading & Processing: {dataset_id}")
    dataset = LeRobotDataset(dataset_id)
    dt = 1.0 / dataset.fps
    
    res = {
        "q_raw": [], "dq": [], "jerk": [], 
        "ep_delays": [], "ep_raw_err": [], "ep_align_err": [],
        "ep0_q": None, "ep0_a": None
    }

    for ep_idx in range(dataset.num_episodes):
        ep_dict = dataset.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        q = np.array(ep_dict["observation.state"])
        a = np.array(ep_dict["action"])
        
        if np.max(np.abs(q)) > 10.0:
            q, a = np.deg2rad(q), np.deg2rad(a)
            
        if ep_idx == 0:
            res["ep0_q"] = q
            res["ep0_a"] = a
            
        # 1. Chất lượng nhận lệnh (Command Quality) - KHÔNG DÙNG FILTER ĐỂ GIỮ NGUYÊN PHA
        delays, raw_err, align_err = compute_command_metrics(q, a)
        res["ep_delays"].append(delays)
        res["ep_raw_err"].append(raw_err)
        res["ep_align_err"].append(align_err)
        
        # 2. Động học (Hardware Stability) - LỌC NHIỄU CHO CẢ SIM VÀ REAL ĐỂ CÔNG BẰNG
        if len(q) > 15:
            q_filt = apply_lowpass_filter(q, cutoff_freq=4.0, fs=dataset.fps)
        else:
            q_filt = q
            
        dq = np.gradient(q_filt, dt, axis=0)
        ddq = np.gradient(dq, dt, axis=0)
        jerk = np.gradient(ddq, dt, axis=0)
        
        res["q_raw"].append(q)
        res["dq"].append(dq)
        res["jerk"].append(jerk)
        
    return {
        "q_raw": np.concatenate(res["q_raw"], axis=0),
        "dq": np.concatenate(res["dq"], axis=0),
        "jerk": np.concatenate(res["jerk"], axis=0),
        "mean_raw_err": np.nanmean(res["ep_raw_err"], axis=0),
        "mean_align_err": np.nanmean(res["ep_align_err"], axis=0),
        "median_delay": np.nan_to_num(np.nanmedian(res["ep_delays"], axis=0), nan=0.0),
        "ep0_q": res["ep0_q"],
        "ep0_a": res["ep0_a"]
    }

def main():
    global SIM_DATASET, REAL_DATASET, OUTPUT_DIR
    args = parse_args()
    SIM_DATASET = args.sim_dataset
    REAL_DATASET = args.real_dataset
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*60)
    print("HARDWARE & COMMAND EXECUTION EVALUATION")
    print("="*60)
    
    sim = process_dataset(SIM_DATASET)
    real = process_dataset(REAL_DATASET)
    
    metrics = []
    for j in range(N_JOINTS):
        # --- HARDWARE STABILITY ---
        mean_vel_sim = np.abs(sim["dq"][:, j]).mean()
        mean_vel_real = np.abs(real["dq"][:, j]).mean()
        
        msj_sim = np.mean(sim["jerk"][:, j]**2)
        msj_real = np.mean(real["jerk"][:, j]**2)
        norm_jerk_sim = msj_sim / (mean_vel_sim**3 + 1e-6)
        norm_jerk_real = msj_real / (mean_vel_real**3 + 1e-6)
        
        range_sim = sim["q_raw"][:, j].max() - sim["q_raw"][:, j].min()
        range_real = real["q_raw"][:, j].max() - real["q_raw"][:, j].min()
        wd_dist = wasserstein_distance(sim["q_raw"][:, j], real["q_raw"][:, j])
        
        # --- COMMAND QUALITY ---
        track_err_sim = np.rad2deg(sim["mean_raw_err"][j])
        align_err_sim = np.rad2deg(sim["mean_align_err"][j])
        track_err_real = np.rad2deg(real["mean_raw_err"][j])
        align_err_real = np.rad2deg(real["mean_align_err"][j])
        
        delay_sim = sim["median_delay"][j]
        delay_real = real["median_delay"][j]
        
        metrics.append({
            "Joint": JOINT_NAMES[j],
            "WD Dist (rad)": round(wd_dist, 3),
            "Range Sim (rad)": round(range_sim, 2),
            "Range Real (rad)": round(range_real, 2),
            "Norm Jerk Sim": round(norm_jerk_sim, 1),
            "Norm Jerk Real": round(norm_jerk_real, 1),
            "Raw Err Sim (°)": round(track_err_sim, 3),
            "Align Err Sim (°)": round(align_err_sim, 3),
            "Raw Err Real (°)": round(track_err_real, 3),
            "Align Err Real (°)": round(align_err_real, 3),
            "Cmd Delay Sim (f)": round(delay_sim, 1),
            "Cmd Delay Real (f)": round(delay_real, 1),
        })
        
    df = pd.DataFrame(metrics)
    
    print("\n[BẢNG 1: ĐỘ ỔN ĐỊNH PHẦN CỨNG (HARDWARE STABILITY)]")
    df_hw = df[["Joint", "WD Dist (rad)", "Range Sim (rad)", "Range Real (rad)", "Norm Jerk Sim", "Norm Jerk Real"]]
    print(df_hw.to_string(index=False))
    
    print("\n[BẢNG 2: CHẤT LƯỢNG NHẬN LỆNH (COMMAND EXECUTION)]")
    df_cmd = df[["Joint", "Align Err Sim (°)", "Align Err Real (°)", "Cmd Delay Sim (f)", "Cmd Delay Real (f)"]]
    print(df_cmd.to_string(index=False))
    
    df.to_csv(f"{OUTPUT_DIR}/comprehensive_metrics.csv", index=False)
    
    # === VẼ BIỂU ĐỒ ===
    print("\nĐang vẽ biểu đồ...")
    x = np.arange(N_JOINTS)
    width = 0.35
    
    # 1. Plot Command Delay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, df["Cmd Delay Sim (f)"], width, label='Sim Delay', color='#378ADD')
    ax.bar(x + width/2, df["Cmd Delay Real (f)"], width, label='Real Delay', color='#D85A30')
    ax.set_ylabel('Delay Frames (Action vs State)')
    ax.set_title('Command Execution Delay (Phase Shift)')
    ax.set_xticks(x); ax.set_xticklabels(JOINT_NAMES, rotation=15)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/plot_command_delay.png", dpi=300); plt.close()
    
    # 2. Plot Tracking Error (Aligned)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, df["Align Err Sim (°)"], width, label='Sim Error (Aligned)', color='#378ADD')
    ax.bar(x + width/2, df["Align Err Real (°)"], width, label='Real Error (Aligned)', color='#D85A30')
    ax.set_ylabel('Mean Absolute Error (degrees)')
    ax.set_title('True Action Tracking Quality (After Phase Shift Compensation)')
    ax.set_xticks(x); ax.set_xticklabels(JOINT_NAMES, rotation=15)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/plot_tracking_error.png", dpi=300); plt.close()

    # 3. Minh hoạ Action vs State của tập 0 (Real)
    ep0_q = real["ep0_q"][:200] # Lấy 200 frame đầu
    ep0_a = real["ep0_a"][:200]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle("Example: Real Robot Command vs Actual Execution (Episode 0)", fontsize=14)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(np.rad2deg(ep0_a[:, i]), label="Command (Action)", linestyle="--", color="gray", lw=2)
        ax.plot(np.rad2deg(ep0_q[:, i]), label="Actual (State)", color="#D85A30", alpha=0.8, lw=2)
        ax.set_title(JOINT_NAMES[i])
        ax.set_ylabel("Degrees")
        if i == 0: ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_real_action_vs_state_ep0.png", dpi=300); plt.close()

    print(f"Hoàn tất! Các file đã được lưu tại thư mục: ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
