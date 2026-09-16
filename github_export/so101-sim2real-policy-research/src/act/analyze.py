import argparse

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy

# Import hàm chính từ script phân tích latent space (latent_space_analysis.py, cùng thư mục)
from lerobot.policies.act.latent_space_analysis import run_latent_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ACT VAE latent-space analysis comparing a sim dataset and a real dataset."
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Path or Hugging Face repo id of the ACT pretrained_model directory (must have use_vae=True).",
    )
    parser.add_argument("--sim-dataset-id", required=True, help="LeRobot dataset id for the sim side.")
    parser.add_argument("--real-dataset-id", required=True, help="LeRobot dataset id for the real side.")
    parser.add_argument("--output-dir", default="latent_analysis_results", help="Directory for plots.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Cap on batches per dataset for a quick run. Pass 0 to use the whole dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    max_batches = args.max_batches or None

    print(f"Loading policy from {args.policy_path}...")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(device)

    if not policy.config.use_vae:
        raise ValueError("Model must be trained with `use_vae=True` to expose a latent space.")

    print(f"Loading sim dataset from {args.sim_dataset_id}...")
    sim_dataset = LeRobotDataset(args.sim_dataset_id)
    # ACT VAE encoder yêu cầu input là một chuỗi hành động (chunk) chứ không phải 1 frame,
    # nên cấu hình delta_timestamps để lấy đủ chunk_size actions.
    delta_timestamps_sim = {"action": [i / sim_dataset.fps for i in range(policy.config.chunk_size)]}
    sim_dataset = LeRobotDataset(args.sim_dataset_id, delta_timestamps=delta_timestamps_sim)
    sim_dataloader = DataLoader(
        sim_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"Loading real dataset from {args.real_dataset_id}...")
    real_dataset = LeRobotDataset(args.real_dataset_id)
    delta_timestamps_real = {"action": [i / real_dataset.fps for i in range(policy.config.chunk_size)]}
    real_dataset = LeRobotDataset(args.real_dataset_id, delta_timestamps=delta_timestamps_real)
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print("\nStarting latent space analysis...")
    run_latent_analysis(
        model=policy,
        sim_dataloader=sim_dataloader,
        real_dataloader=real_dataloader,
        dataset_stats=real_dataset.meta.stats,  # Dùng stats của data thật để chuẩn hóa
        device=device,
        output_dir=args.output_dir,
        max_batches=max_batches,
    )

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
