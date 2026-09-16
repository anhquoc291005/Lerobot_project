# Pipeline đầy đủ: cài đặt → thu dataset → train → rollout → eval → phân tích

Toàn bộ lệnh dưới đây chạy trong một môi trường đã cài `lerobot`. Đây là bản
tóm tắt các lệnh copy-paste được cho SO-101; chi tiết đầy đủ hơn nằm trong tài
liệu chính thức của LeRobot (`docs/source/so101.mdx`, `docs/source/il_robots.mdx`
trong repo `lerobot`).

## 1. Cài đặt

Hướng dẫn đầy đủ: [`docs/installation.md`](../installation.md). Tóm tắt:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e ".[core_scripts,training,feetech,diffusion]"
git lfs install && git lfs pull
hf auth login                    # cần để push dataset/policy lên Hugging Face Hub

cd /path/to/so101-sim2real-policy-research
uv pip install -r requirements.txt   # shap, matplotlib, mujoco, ... cho repo này
```

## 2. Nhánh REAL — thu dataset trên robot thật

**2.1 Tìm cổng USB** (chạy 1 lần / tay, rút dây khi được nhắc):

```bash
lerobot-find-port
```

**2.2 Set motor ID & baudrate** (1 lần / tay):

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER_PORT>
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>
```

**2.3 Calibrate** — căn giữa khớp, Enter, quét hết tầm mỗi khớp. `id` là khoá hiệu
chuẩn, dùng lại ở mọi lệnh sau:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>   --teleop.id=my_leader
```

**2.4 Teleoperate** (kiểm tra nhanh, không ghi):

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

**2.5 Ghi dataset** (phím: **→** episode tiếp, **←** ghi lại, **ESC** kết thúc & upload):

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')

lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/my_task \
  --dataset.single_task="<mô tả nhiệm vụ>" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

Lệnh này tự động **push dataset lên Hugging Face Hub** khi hoàn tất (đây là bước
"up lên huggingface" cho dữ liệu thật).

**2.6 Visualize trước khi train** — bắt buộc, kiểm tra frame thiếu, camera mờ, vật
thể đặt không nhất quán: https://huggingface.co/spaces/lerobot/visualize_dataset
(paste `${HF_USER}/my_task`).

**2.7 Replay** (kiểm tra sanity):

```bash
lerobot-replay --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --dataset.repo_id=${HF_USER}/my_task --dataset.episode=0
```

## 3. Nhánh SIM — thu dataset trong MuJoCo (leader thật điều khiển sim)

Chi tiết đầy đủ: [`docs/so101/so101_mujoco_sim.md`](../so101/so101_mujoco_sim.md).
Tóm tắt:

```bash
cd /home/vietanh/lerobot/github_export/so101-sim2real-policy-research
uv run python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_real_like.json
```

Config này thu liên tục, không reset MuJoCo giữa các episode và tự push lên
`vasco281204/so101_mujoco_real_like` sau khi finalize thành công. Xem hướng dẫn
thu mới, kiểm tra trước upload và resume trong
[`docs/so101/so101_mujoco_sim.md`](../so101/so101_mujoco_sim.md).

## 4. Train — ACT hoặc Diffusion Policy

Mặc định dùng **ACT** (nhanh nhất, tốn ít VRAM nhất — xem
`docs/policies/act_policy_lerobot.md`). **Diffusion Policy** (`docs/policies/diffusion_policy_lerobot.md`)
phù hợp khi action đa phương thức (multi-modal), cần GPU tầm trung trở lên.

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_my_task \
  --job_name=act_my_task \
  --batch_size=8 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_my_task
```

Đổi `--policy.type=diffusion` để train Diffusion Policy với cùng cấu trúc lệnh.
Ví dụ code training/inference thủ công (không qua CLI) cho từng policy:
`src/act/act_training_example.py`, `src/act/act_using_example.py`,
`src/diffusion/diffusion_training_example.py`, `src/diffusion/diffusion_using_example.py`.

Quy tắc chọn policy và thời lượng train (epoch/step, batch size, khi nào dừng):
xem `docs/policies/` và tài liệu LeRobot chính (`AGENT_GUIDE.md` §6–§7 trong repo `lerobot`).

## 5. Rollout — chạy policy trên robot/mô phỏng và ghi lại

`lerobot-rollout` là CLI triển khai policy đã train, hỗ trợ nhiều chiến lược:

```bash
# Base — chạy nhanh không ghi
lerobot-rollout \
  --strategy.type=base \
  --policy.path=${HF_USER}/act_my_task \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --task="<mô tả nhiệm vụ>" --duration=30

# Sentry — ghi liên tục + tự upload theo chu kỳ (dùng để sinh dữ liệu cho reports/rollout_metrics/)
lerobot-rollout \
  --strategy.type=sentry \
  --strategy.upload_every_n_episodes=5 \
  --policy.path=${HF_USER}/act_my_task \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --dataset.repo_id=${HF_USER}/rollout_sentry_data \
  --dataset.single_task="<mô tả nhiệm vụ>" --duration=3600
```

Dữ liệu rollout kiểu sentry chính là nguồn cho các file trong `reports/rollout_metrics/`.

## 6. Eval — so sánh success rate với baseline teleop

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_my_task \
  --dataset.single_task="<mô tả nhiệm vụ giống lúc train>" \
  --dataset.num_episodes=10 \
  --policy.path=${HF_USER}/act_my_task
```

## 7. Phân tích sau khi có dataset sim + real (hoặc rollout)

| Câu hỏi | Script |
|---|---|
| Gap giữa sim và real (quỹ đạo, tracking error, độ trễ, độ mượt) | `src/sim_real/paper_quality_sim_real_gap.py`, `src/sim_real/compare_sim_real.py` |
| Ổn định phần cứng & độ trễ lệnh | `src/sim_real/hardware_command_eval.py` |
| Vận tốc/gia tốc/jerk | `src/sim_real/sim_real_kinematic_analysis.py` |
| SHAP — đóng góp của từng camera/khớp vào action của ACT | `src/act/shap_analysis.py` |
| Latent-space VAE của ACT (sim vs real) | `src/act/latent_space_analysis.py` (hoặc `src/act/analyze.py`) |
| Công thức các chỉ số ở trên | `docs/metrics/sim_real_metrics_formulas.md` |

Ví dụ chạy (xem thêm `--help` mỗi script để biết đủ tham số, hoặc README gốc):

```bash
cd /home/vietanh/lerobot
uv run python github_export/so101-sim2real-policy-research/src/sim_real/paper_quality_sim_real_gap.py \
  --sim-dataset ngocthuong2212/so101_pick_red_no_ep100_left \
  --real-dataset vasco281204/pick_red_fixed \
  --output-dir outputs/sim_real_gap_figures

uv run python github_export/so101-sim2real-policy-research/src/act/shap_analysis.py \
  --policy-path ${HF_USER}/act_my_task \
  --dataset-id vasco281204/pick_red_fixed \
  --output-dir outputs/shap_results
```

## Sơ đồ tổng quan

```
        REAL: calibrate → teleoperate → lerobot-record (push HF)
                                                            \
SIM:  scene.xml → teleop_leader_to_mujoco / collect_sim_dataset.py (push HF)
                                                              \
                                                    lerobot-train (--policy.type=act|diffusion)
                                                              \
                                          lerobot-rollout (base/sentry) → lerobot-record (eval)
                                                              \
                              src/sim_real/*  +  src/act/shap_analysis.py, latent_space_analysis.py
                                                              \
                                        reports/  +  docs/metrics/sim_real_metrics_formulas.md
```
