import os
import argparse
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lerobot.utils.constants import OBS_IMAGES
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import warnings
warnings.filterwarnings("ignore")

class ACTShapWrapper(torch.nn.Module):
    def __init__(self, policy, camera_keys, state_key="observation.state", target_step=0, target_action_dim=0):
        super().__init__()
        self.policy = policy
        self.camera_keys = camera_keys
        self.state_key = state_key
        self.target_step = target_step
        self.target_action_dim = target_action_dim

    def forward(self, *inputs):
        # GradientExplainer truyền vào một list inputs (các tensors), ta cần đóng gói lại thành dict
        obs_dict = {}
        for i, key in enumerate(self.camera_keys):
            obs_dict[key] = inputs[i]
        obs_dict[self.state_key] = inputs[-1]
        
        # ACT model yêu cầu gom các ảnh lại thành dạng list thông qua biến môi trường OBS_IMAGES
        obs_dict[OBS_IMAGES] = [obs_dict[key] for key in self.camera_keys]
        
        # Gọi trực tiếp self.policy.model để bỏ qua bước tính loss yêu cầu biến 'action' 
        # Đồng thời giữ lại đồ thị tính toán (gradient graph) thay vì dùng predict_action_chunk
        action_chunk, _ = self.policy.model(obs_dict)

        # Lấy giá trị action tại step và dimension mong muốn để SHAP giải thích
        target_action = action_chunk[:, self.target_step, self.target_action_dim].unsqueeze(-1)
        return target_action

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SHAP GradientExplainer for an ACT policy trained with LeRobot."
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Path or Hugging Face repo id of the ACT pretrained_model directory.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="LeRobot dataset id used as background/test data.",
    )
    parser.add_argument("--output-dir", default="shap_results", help="Directory for SHAP plots.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-step", type=int, default=0)
    parser.add_argument("--target-action-dim", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading policy from {args.policy_path}...")
    policy = ACTPolicy.from_pretrained(args.policy_path).to(device)
    policy.eval()

    # Tự động lấy các camera keys từ cấu hình của mô hình đã train
    CAMERA_KEYS = [k for k in policy.config.image_features]
    print(f"Detected camera keys from policy config: {CAMERA_KEYS}")

    # Bọc model. Ở đây đang setup giải thích cho hành động ở step 0, dimension 0 (ví dụ: tiến/lùi trục X)
    wrapper_model = ACTShapWrapper(
        policy=policy, 
        camera_keys=CAMERA_KEYS, 
        target_step=args.target_step,
        target_action_dim=args.target_action_dim,
    ).to(device)

    print(f"Loading dataset {args.dataset_id}...")
    dataset = LeRobotDataset(args.dataset_id)
    # Batch size cực nhỏ để tránh hết bộ nhớ VRAM khi tính gradient cho ảnh
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Lấy background data (mẫu tham chiếu)
    print("Fetching background data...")
    bg_batch = next(iter(dataloader))
    bg_inputs = [bg_batch[k].to(device) for k in CAMERA_KEYS]
    bg_inputs.append(bg_batch["observation.state"].to(device))
    
    # Lấy test data (mẫu cần giải thích)
    print("Fetching test data to explain...")
    test_batch = next(iter(dataloader))
    test_inputs = [test_batch[k].to(device) for k in CAMERA_KEYS]
    test_inputs.append(test_batch["observation.state"].to(device))

    print("Initializing SHAP GradientExplainer (điều này có thể tốn vài phút)...")
    explainer = shap.GradientExplainer(wrapper_model, bg_inputs)

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(test_inputs)
    
    print(f"Generating and saving plots to ./{args.output_dir}...")
    # 1. Trực quan hoá State (Trạng thái robot)
    state_shap_values = shap_values[-1]
    state_test_data = test_inputs[-1].cpu().numpy()
    state_feature_names = [f"joint_{i}" for i in range(state_test_data.shape[1])]
    
    plt.figure()
    shap.summary_plot(state_shap_values, state_test_data, feature_names=state_feature_names, show=False)
    plt.savefig(os.path.join(args.output_dir, "state_summary_plot.png"), bbox_inches="tight")
    plt.close()
    
    # 2. Trực quan hoá Hình ảnh (Image)
    for i, cam_key in enumerate(CAMERA_KEYS):
        img_test_data = test_inputs[i].cpu().numpy().transpose(0, 2, 3, 1) # Chuyển (B, C, H, W) -> (B, H, W, C)
        img_shap_values = shap_values[i].transpose(0, 2, 3, 1)
        
        # Plot và tự động pop-up UI
        shap.image_plot(img_shap_values, img_test_data, show=False)
        plt.savefig(os.path.join(args.output_dir, f"{cam_key.replace('.', '_')}_plot.png"), bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()
