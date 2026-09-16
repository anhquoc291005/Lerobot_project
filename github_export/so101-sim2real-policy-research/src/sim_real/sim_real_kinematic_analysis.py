"""
Sim-to-Real Kinematic Analysis cho LeRobot
Đánh giá độ mượt, vận tốc, gia tốc và phân phối trạng thái khớp.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.signal import butter, filtfilt
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Tên 6 khớp của SO-101 follower
JOINT_NAMES = [
    "shoulder_pan", 
    "shoulder_lift", 
    "elbow_flex", 
    "wrist_flex", 
    "wrist_roll", 
    "gripper"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SO-101 sim-real kinematic metrics: position, velocity, acceleration, jerk."
    )
    parser.add_argument(
        "--sim-dataset",
        default="ngocthuong2212/so101_pick_red_no_ep100_left",
    )
    parser.add_argument("--real-dataset", default="vasco281204/pick_red_fixed")
    parser.add_argument("--output-dir", default="kinematic_analysis_results")
    parser.add_argument("--cutoff-freq", type=float, default=3.0)
    return parser.parse_args()

def apply_lowpass_filter(data, cutoff_freq, fs, order=4):
    """Bộ lọc Butterworth Low-pass để loại bỏ nhiễu tần số cao (bước nhảy kỹ thuật số)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def extract_kinematics(dataset_id, cutoff_freq=3.0):
    """
    Đọc dataset và tính toán Vị trí (q), Vận tốc (dq), Gia tốc (ddq), Jerk (dddq)
    """
    print(f"Loading dataset: {dataset_id}")
    dataset = LeRobotDataset(dataset_id)
    dt = 1.0 / dataset.fps
    
    all_q = []
    all_dq = []
    all_ddq = []
    all_jerk = []

    # Xử lý theo từng episode để tránh tính đạo hàm sai ở điểm nối giữa các episode
    for ep_idx in range(dataset.num_episodes):
        ep_dict = dataset.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        
        # Shape: (T, 6)
        q = np.array(ep_dict["observation.state"])
        
        # Tự động phát hiện và chuyển đổi Degrees -> Radians
        # Thông thường radian nằm trong khoảng [-2pi, 2pi]. Nếu max value > 10, khả năng rất cao data đang là Degrees.
        if np.max(np.abs(q)) > 10.0:
            q = np.deg2rad(q)
            
        # Áp dụng Butterworth Low-pass filter (Lọc các nhiễu tay rung > 3Hz)
        if len(q) > 15:
            q = apply_lowpass_filter(q, cutoff_freq=cutoff_freq, fs=dataset.fps)
            
        # Tính đạo hàm bằng np.gradient để mượt hơn so với np.diff
        dq = np.gradient(q, dt, axis=0)      # Vận tốc (rad/s)
        ddq = np.gradient(dq, dt, axis=0)    # Gia tốc (rad/s^2)
        jerk = np.gradient(ddq, dt, axis=0)  # Jerk (rad/s^3)
        
        all_q.append(q)
        all_dq.append(dq)
        all_ddq.append(ddq)
        all_jerk.append(jerk)
        
    return {
        "q": np.concatenate(all_q, axis=0),
        "dq": np.concatenate(all_dq, axis=0),
        "ddq": np.concatenate(all_ddq, axis=0),
        "jerk": np.concatenate(all_jerk, axis=0)
    }

def plot_distribution_comparison(sim_data, real_data, metric, unit, save_dir):
    """Vẽ biểu đồ phân phối (KDE) cho từng khớp."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Comparison of {metric.capitalize()} Distribution (Sim vs Real)", fontsize=16)
    axes = axes.flatten()
    
    for i in range(6):
        sns.kdeplot(sim_data[:, i], ax=axes[i], label="Sim", fill=True, color="#378ADD", alpha=0.5)
        sns.kdeplot(real_data[:, i], ax=axes[i], label="Real", fill=True, color="#D85A30", alpha=0.5)
        
        axes[i].set_title(JOINT_NAMES[i])
        axes[i].set_xlabel(f"{metric} ({unit})")
        if i == 0:
            axes[i].legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric}_distribution.png"), dpi=300)
    plt.close()

def calculate_quantitative_metrics(sim_kin, real_kin):
    """
    Tính toán các metric định lượng để đưa vào bảng trong paper.
    """
    metrics = {"Joint": JOINT_NAMES}
    
    # 1. Wasserstein Distance của Position (Đo độ lệch phân phối không gian hoạt động)
    wd_q = [wasserstein_distance(sim_kin["q"][:, i], real_kin["q"][:, i]) for i in range(6)]
    metrics["WD Position (rad)"] = wd_q
    
    # 2. Mean Absolute Velocity (Cái nào di chuyển nhanh hơn trên trung bình)
    mean_vel_sim = np.abs(sim_kin["dq"]).mean(axis=0)
    mean_vel_real = np.abs(real_kin["dq"]).mean(axis=0)
    metrics["Mean Speed Sim (rad/s)"] = mean_vel_sim
    metrics["Mean Speed Real (rad/s)"] = mean_vel_real
    
    # 3. Mean Squared Jerk (Đo độ rung/giật)
    # Dùng log để dễ nhìn do Jerk thường có giá trị rất lớn
    msj_sim = np.mean(sim_kin["jerk"]**2, axis=0)
    msj_real = np.mean(real_kin["jerk"]**2, axis=0)
    metrics["MSJ Sim (rad^2/s^6)"] = msj_sim
    metrics["MSJ Real (rad^2/s^6)"] = msj_real
    metrics["Jitter Increase (%)"] = ((msj_real - msj_sim) / msj_sim) * 100
    
    # 4. Normalized MSJ (Chuẩn hóa Jerk theo vận tốc)
    # Jerk tỷ lệ thuận với v^3. Chia cho v^3 để so sánh độ mượt nội tại của quỹ đạo
    norm_msj_sim = msj_sim / (mean_vel_sim**3 + 1e-6)
    norm_msj_real = msj_real / (mean_vel_real**3 + 1e-6)
    metrics["Norm. MSJ Sim"] = norm_msj_sim
    metrics["Norm. MSJ Real"] = norm_msj_real
    metrics["Norm. Jitter Increase (%)"] = ((norm_msj_real - norm_msj_sim) / norm_msj_sim) * 100
    
    import pandas as pd
    df = pd.DataFrame(metrics)
    return df

def plot_jerk_bar_chart(df, save_dir):
    """Vẽ biểu đồ cột so sánh độ giật (Jerk) giữa Sim và Real."""
    x = np.arange(len(JOINT_NAMES))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, df["Norm. MSJ Sim"], width, label='Sim (Normalized)', color='#378ADD')
    ax.bar(x + width/2, df["Norm. MSJ Real"], width, label='Real (Normalized)', color='#D85A30')
    
    ax.set_ylabel('Velocity-Normalized Mean Squared Jerk (Log Scale)')
    ax.set_title('Smoothness Analysis: Simulation vs Real Robot')
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=15)
    ax.set_yscale('log') # Dùng log scale vì Real jerk thường vọt lên rất cao
    ax.legend()
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "jerk_comparison.png"), dpi=300)
    plt.close()

def main():
    args = parse_args()
    SIM_DATASET = args.sim_dataset
    REAL_DATASET = args.real_dataset
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Trích xuất dữ liệu
    sim_kin = extract_kinematics(SIM_DATASET, cutoff_freq=args.cutoff_freq)
    real_kin = extract_kinematics(REAL_DATASET, cutoff_freq=args.cutoff_freq)
    
    # 2. Vẽ biểu đồ phân phối
    print("Plotting distributions...")
    plot_distribution_comparison(sim_kin["q"], real_kin["q"], "position", "rad", OUTPUT_DIR)
    plot_distribution_comparison(sim_kin["dq"], real_kin["dq"], "velocity", "rad/s", OUTPUT_DIR)
    plot_distribution_comparison(sim_kin["ddq"], real_kin["ddq"], "acceleration", "rad/s^2", OUTPUT_DIR)
    
    # 3. Tính toán các metrics cho bảng trong Paper
    print("Calculating metrics...")
    df_metrics = calculate_quantitative_metrics(sim_kin, real_kin)
    
    # In ra terminal
    print("\n" + "="*80)
    print("KINEMATIC METRICS REPORT (Copy for Paper Table)")
    print("="*80)
    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("="*80)
    
    # Lưu ra CSV
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "kinematic_metrics.csv"), index=False)
    
    # 4. Vẽ biểu đồ Jitter (Jerk)
    plot_jerk_bar_chart(df_metrics, OUTPUT_DIR)
    
    print(f"\nDone! All results saved to ./{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
