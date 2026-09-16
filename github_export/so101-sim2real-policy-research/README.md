# SO-101 Sim-to-Real Policy Research

Repository nghiên cứu cá nhân (private) về **ảnh hưởng của chất lượng dữ liệu
sim-to-real lên policy imitation learning**, dùng cánh tay robot **SO-101**. Gồm:

- **ACT Policy** và **Diffusion Policy** trong LeRobot: code implementation đầy đủ,
  ví dụ train/inference, và script phân tích (SHAP, latent-space VAE).
- **Điều khiển SO-101** trên robot thật (leader/follower, teleop, calibrate) và
  trong **mô phỏng MuJoCo** (teleop tay thật → sim, thu dataset từ sim).
- **Đánh giá sim-real gap**: quỹ đạo, tracking error, độ trễ lệnh, động học
  (vận tốc/gia tốc/jerk), cùng công thức các chỉ số.
- Pipeline đầy đủ: cài đặt → thu dataset → push Hugging Face → train → rollout → eval.

Repo chỉ chứa **code, tài liệu, và số liệu tổng hợp đã chọn lọc**. Không có dataset
raw, checkpoint model, cache, môi trường ảo, sample CSV lớn, hay **hình ảnh kết quả**
(mọi hình có thể tái tạo bằng script tương ứng — xem
[`docs/pipeline/end_to_end_pipeline.md`](docs/pipeline/end_to_end_pipeline.md)).

## Cấu trúc

```text
.
├── src/
│   ├── act/             # SHAP, latent-space analysis, ví dụ train/inference cho ACT
│   ├── diffusion/        # Ví dụ train/inference cho Diffusion Policy
│   ├── sim_mujoco/       # Teleop tay thật -> MuJoCo, thu dataset từ sim, validate/push HF
│   ├── sim_real/         # Đánh giá sim-real gap, hardware/command tracking, động học
│   ├── dataset_tools/    # Công cụ sửa/clone/label dataset LeRobot
│   └── utils/            # Tiện ích nhỏ, ví dụ xem CSV
├── reference/            # Code vendor từ LeRobot/SO-ARM100 (xem NOTICE.md), để đọc kèm src/
│   ├── policies/act/, policies/diffusion/   # Implementation đầy đủ của 2 policy
│   ├── robots_so101/     # Class điều khiển SO-101 follower/leader thật
│   └── sim_assets/       # Model MuJoCo (MJCF) + URDF của SO-100/SO-101
├── docs/
│   ├── policies/         # Ghi chú ACT/Diffusion Policy
│   ├── so101/             # Setup SO-101 thật + mô phỏng MuJoCo
│   ├── pipeline/          # Pipeline đầy đủ, lệnh copy-paste
│   └── metrics/           # Công thức đánh giá sim-real
└── reports/
    ├── sim_real_gap/     # Số liệu/bảng sim-real gap
    ├── hardware_command/ # Tracking error, command delay, hardware stability
    ├── kinematics/        # Vận tốc, gia tốc, jerk, phân phối trạng thái
    └── rollout_metrics/   # Kết quả rollout (lerobot-rollout --strategy.type=sentry)
```

## Cài đặt

Hướng dẫn đầy đủ (dựa theo [tài liệu cài đặt chính thức của LeRobot](https://huggingface.co/docs/lerobot/installation),
rút gọn cho SO-101 + ACT/Diffusion + MuJoCo): [`docs/installation.md`](docs/installation.md).

Tóm tắt nhanh:

```bash
# 1. Cài lerobot (env Python >= 3.12, xem docs/installation.md để có ffmpeg/CUDA đúng)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e ".[core_scripts,training,feetech,diffusion]"

# 2. Cài dependency riêng của repo này
cd /path/to/so101-sim2real-policy-research
uv pip install -r requirements.txt

# 3. Đăng nhập Hugging Face (push/pull dataset & policy)
hf auth login
```

Nếu repo này đang nằm trong `lerobot/github_export/...` (như checkout hiện tại):

```bash
cd /home/vietanh/lerobot
uv pip install -r github_export/so101-sim2real-policy-research/requirements.txt
```

`requirements.txt` cài các thư viện phân tích (`shap`, `matplotlib`, `pandas`, ...)
và `mujoco` cho phần mô phỏng. Bản thân `lerobot` cần được cài/checkout riêng
(repo này **không** vendor toàn bộ `lerobot` — chỉ vendor các file core cần đọc
kèm trong `reference/`, xem [`NOTICE.md`](NOTICE.md)).

## Đọc theo thứ tự

1. [`docs/installation.md`](docs/installation.md) — cài đặt môi trường LeRobot.
2. [`docs/so101/so101_setup_lerobot.mdx`](docs/so101/so101_setup_lerobot.mdx) — setup SO-101 thật.
3. [`docs/so101/so101_mujoco_sim.md`](docs/so101/so101_mujoco_sim.md) — mô phỏng MuJoCo, teleop thật → sim.
4. [`docs/so101/so101_stack_3_blocks_sim.md`](docs/so101/so101_stack_3_blocks_sim.md) — xếp chồng 3 khối hộp (demo trực quan & thu dataset).
5. [`docs/pipeline/end_to_end_pipeline.md`](docs/pipeline/end_to_end_pipeline.md) — pipeline đầy đủ: cài đặt → thu dataset → train → rollout → eval.
6. [`docs/policies/act_policy_lerobot.md`](docs/policies/act_policy_lerobot.md), [`docs/policies/diffusion_policy_lerobot.md`](docs/policies/diffusion_policy_lerobot.md) — ghi chú policy.
7. [`docs/metrics/sim_real_metrics_formulas.md`](docs/metrics/sim_real_metrics_formulas.md) — công thức đánh giá sim-real.
8. [`reports/sim_real_gap/`](reports/sim_real_gap/) và [`reports/hardware_command/`](reports/hardware_command/) — số liệu kết quả.

## Chạy phân tích sim-real

```bash
cd /home/vietanh/lerobot
uv run python github_export/so101-sim2real-policy-research/src/sim_real/paper_quality_sim_real_gap.py \
  --sim-dataset ngocthuong2212/so101_pick_red_no_ep100_left \
  --real-dataset vasco281204/pick_red_fixed \
  --output-dir outputs/sim_real_gap_figures
```

Đánh giá hardware stability và command execution:

```bash
uv run python github_export/so101-sim2real-policy-research/src/sim_real/hardware_command_eval.py \
  --sim-dataset ngocthuong2212/so101_pick_red_no_ep100_left \
  --real-dataset vasco281204/pick_red_fixed \
  --output-dir outputs/hardware_command_eval
```

Đánh giá động học:

```bash
uv run python github_export/so101-sim2real-policy-research/src/sim_real/sim_real_kinematic_analysis.py \
  --sim-dataset ngocthuong2212/so101_pick_red_no_ep100_left \
  --real-dataset vasco281204/pick_red_fixed \
  --output-dir outputs/kinematic_analysis
```

## Chạy SHAP / latent-space cho ACT

```bash
cd /home/vietanh/lerobot
uv run python github_export/so101-sim2real-policy-research/src/act/shap_analysis.py \
  --policy-path outputs/trained_model/100001/pretrained_model \
  --dataset-id vasco281204/pick_red_fixed \
  --output-dir outputs/shap_results \
  --target-step 0 \
  --target-action-dim 0

uv run python github_export/so101-sim2real-policy-research/src/act/analyze.py \
  --policy-path outputs/trained_model/100001/pretrained_model \
  --sim-dataset-id ngocthuong2212/so101_pick_red_no_ep100_left \
  --real-dataset-id vasco281204/pick_red_fixed \
  --output-dir outputs/latent_analysis_results
```

## Thu dataset trong MuJoCo (leader thật -> sim)

```bash
cd /home/vietanh/lerobot/github_export/so101-sim2real-policy-research
uv run python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_real_like.json
```

Chi tiết: [`docs/so101/so101_mujoco_sim.md`](docs/so101/so101_mujoco_sim.md).

## Demo & Thu dataset xếp chồng 3 khối hộp (Stack 3 Blocks)

Nhiệm vụ xếp chồng 3 khối hộp kích thước 30x30x15 mm: **Đen (dưới)** $\to$ **Xanh lá (giữa)** $\to$ **Xanh nước biển (trên)** với cơ chế tự động random vị trí 3 khối hộp sau mỗi episode.

### 1. Chạy Demo không cần cắm tay Leader:
```bash
# Xem 2 góc camera (top_cam + front_cam), nhấn Space để random vị trí:
python src/sim_mujoco/demo_stack_3_blocks.py

# Mở 3D Interactive Viewer của MuJoCo (xoay 3D, kéo thả chuột tương tác vật lý):
python src/sim_mujoco/demo_stack_3_blocks.py --viewer

# Chạy thử quy trình thu thập dataset (dry-run demo):
python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_stack_3_blocks_demo.json
```

### 2. Thu thập dataset với tay Leader thật:
```bash
python src/sim_mujoco/collect_sim_dataset.py \
  --config src/sim_mujoco/config_sim_collect_stack_3_blocks.json
```

Chi tiết: [`docs/so101/so101_stack_3_blocks_sim.md`](docs/so101/so101_stack_3_blocks_sim.md).

## Ghi chú dữ liệu

Các script mặc định dùng:

- Sim dataset: [`ngocthuong2212/so101_pick_red_no_ep100_left`](https://huggingface.co/datasets/ngocthuong2212/so101_pick_red_no_ep100_left)
- Real dataset: `vasco281204/pick_red_fixed`

Nếu dùng dataset khác, truyền qua `--sim-dataset`/`--sim-dataset-id` và
`--real-dataset`/`--real-dataset-id`.

## Bản quyền & nguồn gốc code

Repo dùng giấy phép **Apache-2.0** (xem [`LICENSE`](LICENSE), giống LeRobot). Một
phần code trong `reference/` và `src/sim_mujoco/` được vendor/thích nghi từ
`huggingface/lerobot`, `TheRobotStudio/SO-ARM100`, và fork
`thuongtran21112004-star/realtosim` — chi tiết nguồn gốc từng phần xem
[`NOTICE.md`](NOTICE.md).
