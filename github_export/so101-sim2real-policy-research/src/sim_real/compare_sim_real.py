import numpy as np
import torch
import argparse
from scipy.stats import wasserstein_distance
from scipy.signal import correlate
import pandas as pd
import matplotlib.pyplot as plt
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SO-101 sim and real LeRobot datasets.")
    parser.add_argument(
        "--sim-dataset",
        default="ngocthuong2212/so101_pick_red_no_ep100_left",
    )
    parser.add_argument("--real-dataset", default="vasco281204/pick_red_fixed")
    parser.add_argument("--output-dir", default="sim_real_gap_results")
    return parser.parse_args()

def extract_global_data(repo_id, fps=30):
    """
    Tải dataset và trích xuất toàn bộ state, action thành mảng numpy phẳng 
    để đánh giá phân phối toàn cục.
    """
    print(f"Đang tải dataset: {repo_id}...")
    dataset = LeRobotDataset(repo_id)

    # Trích xuất toàn bộ state và action trực tiếp từ Hugging Face dataset
    # Đây là cách hiệu quả hơn so với việc duyệt từng frame một
    states_np = np.array(dataset.hf_dataset['observation.state'])  # Shape: (N, 6)
    actions_np = np.array(dataset.hf_dataset['action'])  # Shape: (N, 6)
    
    # Tính Delta q và Delta a
    delta_q = np.diff(states_np, axis=0)
    delta_a = np.diff(actions_np, axis=0)
    
    # Vận tốc (Velocity)
    dt = 1.0 / dataset.fps
    velocities = delta_q / dt
    
    # Độ giật (Jerk) xấp xỉ bằng sai phân bậc 3
    # j_q(t) = q(t+3) - 3q(t+2) + 3q(t+1) - q(t)
    if states_np.shape[0] >= 4: # Cần ít nhất 4 điểm để tính sai phân bậc 3
        jerks = np.zeros((states_np.shape[0] - 3, states_np.shape[1]))
        for i in range(states_np.shape[0] - 3):
            jerks[i] = states_np[i+3] - 3*states_np[i+2] + 3*states_np[i+1] - states_np[i]
    else:
        jerks = np.array([]) # Trả về mảng rỗng nếu không đủ dữ liệu
        
    return states_np, actions_np, delta_q, delta_a, velocities, jerks

def evaluate_sim_real_gap(sim_repo, real_repo):
    # 1. Trích xuất dữ liệu
    q_sim, a_sim, dq_sim, da_sim, v_sim, jerk_sim = extract_global_data(sim_repo)
    q_real, a_real, dq_real, da_real, v_real, jerk_real = extract_global_data(real_repo)
    
    num_joints = q_sim.shape[1] # Thường là 6 cho SO-101
    
    print("\n" + "="*50)
    print(" BÁO CÁO ĐÁNH GIÁ SIM-REAL GAP (SO-101)")
    print("="*50)
    
    joint_metrics_data = []
    
    # ---------------------------------------------------------
    # 4.2 JOINT-SPACE GAP
    # ---------------------------------------------------------
    print("\n--- 4.2 JOINT-SPACE GAP ---")
    for i in range(num_joints):
        # Mean gap: G_mu(q_i) = |mu(q_real) - mu(q_sim)|
        mean_gap = np.abs(np.mean(q_real[:, i]) - np.mean(q_sim[:, i]))
        
        # Position Distribution Gap (Wasserstein)
        w_pos = wasserstein_distance(q_sim[:, i], q_real[:, i])
        
        # Velocity Distribution Gap (Wasserstein)
        w_vel = wasserstein_distance(v_sim[:, i], v_real[:, i])
        
        joint_metrics_data.append({
            "Joint": JOINT_NAMES[i],
            "Mean Gap (q)": mean_gap,
            "Wasserstein Position (q)": w_pos,
            "Wasserstein Velocity (v)": w_vel
        })
        
        print(f"Khớp {JOINT_NAMES[i]}:")
        print(f"  - Lệch tâm trung bình (Mean Gap) : {mean_gap:.4f}")
        print(f"  - Khác biệt vị trí (Wasserstein) : {w_pos:.4f}")
        print(f"  - Khác biệt vận tốc (Wasserstein) : {w_vel:.4f}")
        
    # Jerk Gap (Smoothness): G_jerk = E[||j_real||] - E[||j_sim||]
    norm_jerk_real = np.linalg.norm(jerk_real, axis=1)
    norm_jerk_sim = np.linalg.norm(jerk_sim, axis=1)
    g_jerk = np.mean(norm_jerk_real) - np.mean(norm_jerk_sim) if norm_jerk_real.size > 0 and norm_jerk_sim.size > 0 else np.nan
    print(f"\nĐộ chênh lệch mượt mà (Global Jerk Gap): {g_jerk:.6f}")
    if g_jerk > 0:
        print(" -> Robot thật (Real) rung giật kém mượt hơn Sim.")
    else:
        print(" -> Robot mô phỏng (Sim) giật hơn Real.")

    # ---------------------------------------------------------
    # 4.3 TEMPORAL RESPONSE GAP
    # ---------------------------------------------------------
    print("\n--- 4.3 TEMPORAL RESPONSE GAP ---")
    
    # Action-response gain: E[ ||dq|| / (||da|| + eps) ]
    eps = 1e-6
    gain_real = np.linalg.norm(dq_real, axis=1) / (np.linalg.norm(da_real, axis=1) + eps) if dq_real.size > 0 and da_real.size > 0 else np.array([])
    gain_sim = np.linalg.norm(dq_sim, axis=1) / (np.linalg.norm(da_sim, axis=1) + eps) if dq_sim.size > 0 and da_sim.size > 0 else np.array([])
    
    g_gain = np.mean(gain_real) - np.mean(gain_sim) if gain_real.size > 0 and gain_sim.size > 0 else np.nan
    print(f"Chênh lệch Gain (G_gain): {g_gain:.4f}")
    if g_gain < 0:
        print(" -> Robot thật phản ứng yếu/lì hơn so với lệnh (Simulation nhạy hơn).")
        
    # Response delay xấp xỉ bằng Cross-Correlation trên toàn tập
    # Ta tính độ trễ trung bình của độ lớn dq và da
    norm_da_real = np.linalg.norm(da_real, axis=1)
    norm_dq_real = np.linalg.norm(dq_real, axis=1) if dq_real.size > 0 else np.array([])
    
    norm_da_sim = np.linalg.norm(da_sim, axis=1)
    norm_dq_sim = np.linalg.norm(dq_sim, axis=1) if dq_sim.size > 0 else np.array([])
    
    # Dùng correlate để tìm tau (độ lệch mẫu) có tương quan max
    tau_real = np.nan
    if norm_dq_real.size > 0 and norm_da_real.size > 0:
        corr_real = correlate(norm_dq_real, norm_da_real, mode='full')
        lags_real = np.arange(-len(norm_da_real) + 1, len(norm_dq_real))
        tau_real = lags_real[np.argmax(corr_real)]
    tau_sim = np.nan
    if norm_dq_sim.size > 0 and norm_da_sim.size > 0:
        corr_sim = correlate(norm_dq_sim, norm_da_sim, mode='full')
        lags_sim = np.arange(-len(norm_da_sim) + 1, len(norm_dq_sim))
        tau_sim = lags_real[np.argmax(corr_sim)]
    
    g_delay_frames = tau_real - tau_sim
    print(f"Độ trễ trung bình (Global Delay Gap): {g_delay_frames} frames.")
    
    # Collect all metrics for table and plots
    global_metrics_data = {
        "Global Jerk Gap": g_jerk,
        "Global Gain Gap": g_gain,
        "Global Delay Gap (frames)": g_delay_frames,
    }
    
    return joint_metrics_data, global_metrics_data

def generate_summary_table(joint_metrics, global_metrics, output_dir="sim_real_gap_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- BẢNG TÓM TẮT CHỈ SỐ ---")
    df_joints = pd.DataFrame(joint_metrics)
    print("\nChỉ số theo khớp:")
    print(df_joints.to_string(index=False, float_format="{:.4f}".format))
    
    df_global = pd.DataFrame([global_metrics])
    print("\nChỉ số toàn cục:")
    print(df_global.to_string(index=False, float_format="{:.4f}".format))
    
    # Save to CSV
    df_joints.to_csv(os.path.join(output_dir, "joint_metrics.csv"), index=False)
    df_global.to_csv(os.path.join(output_dir, "global_metrics.csv"), index=False)
    print(f"\nĐã lưu bảng tóm tắt vào {output_dir}/joint_metrics.csv và {output_dir}/global_metrics.csv")

def generate_plots(joint_metrics, global_metrics, output_dir="sim_real_gap_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- ĐANG TẠO BIỂU ĐỒ ---")
    
    # Biểu đồ 1: Joint-Space Metrics (Mean Gap, Wasserstein Position, Wasserstein Velocity)
    df_joints = pd.DataFrame(joint_metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(len(JOINT_NAMES))
    
    ax.bar(index - bar_width, df_joints["Mean Gap (q)"], bar_width, label="Mean Gap (q)", color="#4CAF50")
    ax.bar(index, df_joints["Wasserstein Position (q)"], bar_width, label="Wasserstein Position (q)", color="#2196F3")
    ax.bar(index + bar_width, df_joints["Wasserstein Velocity (v)"], bar_width, label="Wasserstein Velocity (v)", color="#FF9800")
    
    ax.set_xlabel("Khớp")
    ax.set_ylabel("Giá trị Gap")
    ax.set_title("So sánh Sim-Real Gap trong không gian khớp")
    ax.set_xticks(index)
    ax.set_xticklabels(df_joints["Joint"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "joint_space_metrics_bar_chart.png"))
    plt.close()
    print(f"Đã lưu biểu đồ Joint-Space Metrics vào {output_dir}/joint_space_metrics_bar_chart.png")
    
    # Biểu đồ 2: Global Jerk Gap
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Global Jerk Gap"], [global_metrics["Global Jerk Gap"]], color="#E53935")
    ax.set_ylabel("Giá trị Gap")
    ax.set_title("Global Jerk Gap (Độ mượt)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_jerk_gap_bar_chart.png"))
    plt.close()
    print(f"Đã lưu biểu đồ Global Jerk Gap vào {output_dir}/global_jerk_gap_bar_chart.png")
    
    # Biểu đồ 3: Global Temporal Response Gap (Gain & Delay)
    fig, ax = plt.subplots(figsize=(10, 5))
    temporal_labels = ["Global Gain Gap", "Global Delay Gap (frames)"]
    temporal_values = [global_metrics["Global Gain Gap"], global_metrics["Global Delay Gap (frames)"]]
    
    ax.bar(temporal_labels, temporal_values, color=["#673AB7", "#00BCD4"])
    ax.set_ylabel("Giá trị Gap")
    ax.set_title("Global Temporal Response Gap: Gain & Delay")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_temporal_gap_bar_chart.png"))
    plt.close()
    print(f"Đã lưu biểu đồ Global Temporal Gap vào {output_dir}/global_temporal_gap_bar_chart.png")
    
# Thực thi đánh giá
if __name__ == "__main__":
    args = parse_args()
    SIM_DATASET = args.sim_dataset
    REAL_DATASET = args.real_dataset
    OUTPUT_DIR = args.output_dir
    
    joint_metrics, global_metrics = evaluate_sim_real_gap(SIM_DATASET, REAL_DATASET)
    
    generate_summary_table(joint_metrics, global_metrics, OUTPUT_DIR)
    generate_plots(joint_metrics, global_metrics, OUTPUT_DIR)
    
    print(f"\nHoàn tất phân tích! Kết quả (bảng và biểu đồ) đã được lưu vào thư mục: ./{OUTPUT_DIR}")
