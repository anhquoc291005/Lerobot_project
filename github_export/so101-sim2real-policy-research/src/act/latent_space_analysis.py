"""
Latent space analysis: So sánh phân phối mu_hat từ sim data vs real data
trong ACT (Action Chunking Transformer).

Pipeline:
  1. collect_latents()   — forward pass qua dataset, thu thập mu_hat (B, 32)
  2. reduce_dimensions() — PCA xuống 50 chiều, rồi t-SNE xuống 2D
  3. plot_*()            — vẽ các biểu đồ so sánh
  4. compute_metrics()   — FID-style, MMD, per-dimension KL divergence

Yêu cầu: model ACT đã train, DataLoader cho sim và real.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import einops
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. THU THẬP mu_hat TỪ MODEL
# ─────────────────────────────────────────────

def collect_latents(model, dataloader, device, max_batches=None, normalizer=None):
    """
    Forward pass qua toàn bộ dataset, thu thập mu_hat và log_sigma_x2_hat.

    model.forward() trả về:
        actions_hat, (mu_hat, log_sigma_x2_hat)
    Trong đó mu_hat shape = (B, latent_dim) = (B, 32)

    Returns:
        mus:    np.ndarray (N, 32)  — mean của latent distribution
        sigmas: np.ndarray (N, 32)  — std = exp(log_sigma_x2 / 2)
    """
    model.eval()
    all_mus, all_sigmas = [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            # Đưa batch lên device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Chuẩn hóa data giống như lúc train để model tính mu_hat chính xác
            if normalizer is not None:
                batch = normalizer(batch)

            from lerobot.utils.constants import ACTION, OBS_STATE
            
            if model.config.robot_state_feature and OBS_STATE not in batch:
                b_size = batch[ACTION].shape[0]
                shape = model.config.robot_state_feature.shape
                batch[OBS_STATE] = torch.zeros((b_size, *shape), dtype=torch.float32, device=device)
                
            if "action_is_pad" not in batch and ACTION in batch:
                batch["action_is_pad"] = torch.zeros(batch[ACTION].shape[:2], dtype=torch.bool, device=device)

            # Tách riêng VAE Encoder để chạy (Giống hệt modeling_act.py nhưng bỏ qua xử lý ảnh và Dropout)
            act_model = model.model
            b_size = batch[ACTION].shape[0]
            
            cls_embed = einops.repeat(act_model.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=b_size)
            action_embed = act_model.vae_encoder_action_input_proj(batch[ACTION])
            
            if act_model.config.robot_state_feature:
                robot_state_embed = act_model.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
                vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)
                pad_len = 2
            else:
                vae_encoder_input = torch.cat([cls_embed, action_embed], axis=1)
                pad_len = 1
                
            pos_embed = act_model.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full((b_size, pad_len), False, device=device)
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            
            cls_token_out = act_model.vae_encoder(
                vae_encoder_input.permute(1, 0, 2), 
                pos_embed=pos_embed.permute(1, 0, 2), 
                key_padding_mask=key_padding_mask
            )[0]
            
            latent_pdf_params = act_model.vae_encoder_latent_output_proj(cls_token_out)
            mu_hat = latent_pdf_params[:, : act_model.config.latent_dim]
            log_sigma_x2_hat = latent_pdf_params[:, act_model.config.latent_dim :]

            if mu_hat is None:
                # use_vae=False hoặc inference mode → không có latent
                raise RuntimeError("use_vae phải = True và batch phải có 'action' key.")

            # sigma = exp(log_sigma_x2 / 2)
            sigma_hat = (log_sigma_x2_hat / 2).exp()

            all_mus.append(mu_hat.cpu().numpy())
            all_sigmas.append(sigma_hat.cpu().numpy())

    mus    = np.concatenate(all_mus,    axis=0)   # (N, 32)
    sigmas = np.concatenate(all_sigmas, axis=0)   # (N, 32)
    return mus, sigmas


# ─────────────────────────────────────────────
# 2. GIẢM CHIỀU: PCA + t-SNE
# ─────────────────────────────────────────────

def reduce_dimensions(mus_sim, mus_real, n_tsne_components=2, perplexity=40, random_state=42):
    """
    Giảm (N, 32) xuống (N, 2) để visualize.

    Pipeline: StandardScaler → PCA(50) → t-SNE(2)
    PCA trước t-SNE giúp tăng tốc và giảm noise.

    Returns dict chứa kết quả PCA và t-SNE của cả sim và real.
    """
    N_sim  = len(mus_sim)
    N_real = len(mus_real)

    # Chuẩn hóa trên toàn bộ data kết hợp
    all_mus = np.concatenate([mus_sim, mus_real], axis=0)
    scaler  = StandardScaler()
    all_mus_scaled = scaler.fit_transform(all_mus)

    # PCA: giữ 95% variance, tối đa 32 components (latent_dim=32 nên PCA <= 32)
    n_pca = min(32, N_sim + N_real - 1)
    pca = PCA(n_components=n_pca, random_state=random_state)
    all_pca = pca.fit_transform(all_mus_scaled)

    print(f"PCA: {n_pca} components giữ "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance")

    # t-SNE trên toàn bộ (để sim và real dùng cùng embedding space)
    tsne = TSNE(
        n_components=n_tsne_components,
        perplexity=perplexity,
        max_iter=1000,
        random_state=random_state,
        verbose=1,
    )
    all_tsne = tsne.fit_transform(all_pca)

    return {
        "pca_sim":   all_pca[:N_sim],
        "pca_real":  all_pca[N_sim:],
        "tsne_sim":  all_tsne[:N_sim],
        "tsne_real": all_tsne[N_sim:],
        "pca_model": pca,
        "scaler":    scaler,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


# ─────────────────────────────────────────────
# 3. VISUALIZATIONS
# ─────────────────────────────────────────────

COLOR_SIM  = "#378ADD"   # blue
COLOR_REAL = "#D85A30"   # coral


def plot_tsne_scatter(reduced, save_path="tsne_sim_vs_real.png"):
    """
    Scatter plot t-SNE 2D: sim vs real.
    Vẽ thêm density contours để thấy overlap.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    tsne_sim  = reduced["tsne_sim"]
    tsne_real = reduced["tsne_real"]

    ax.scatter(tsne_sim[:, 0],  tsne_sim[:, 1],
               c=COLOR_SIM,  alpha=0.4, s=15, label=f"Sim  (n={len(tsne_sim)})",  zorder=2)
    ax.scatter(tsne_real[:, 0], tsne_real[:, 1],
               c=COLOR_REAL, alpha=0.4, s=15, label=f"Real (n={len(tsne_real)})", zorder=2)

    # KDE contours
    for data, color in [(tsne_sim, COLOR_SIM), (tsne_real, COLOR_REAL)]:
        if len(data) > 10:
            try:
                kde = stats.gaussian_kde(data.T, bw_method=0.3)
                x_min, x_max = data[:, 0].min()-2, data[:, 0].max()+2
                y_min, y_max = data[:, 1].min()-2, data[:, 1].max()+2
                xx, yy = np.mgrid[x_min:x_max:80j, y_min:y_max:80j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors=[color], alpha=0.6, linewidths=1)
            except Exception:
                pass

    ax.set_title("t-SNE của latent space µ: Sim vs Real", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2, fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_dimension_distribution(mus_sim, mus_real,
                                    n_dims_show=8,
                                    save_path="latent_per_dim.png"):
    """
    So sánh phân phối µ theo từng chiều latent (dim 0..31).
    Hiển thị n_dims_show chiều có KL divergence lớn nhất.
    """
    latent_dim = mus_sim.shape[1]

    # Tính KL divergence per dimension (Gaussian approx)
    kl_per_dim = []
    for d in range(latent_dim):
        mu1, s1 = mus_sim[:, d].mean(),  mus_sim[:, d].std()  + 1e-8
        mu2, s2 = mus_real[:, d].mean(), mus_real[:, d].std() + 1e-8
        # KL(P||Q) với P=sim, Q=real (Gaussian)
        kl = np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5
        kl_per_dim.append(kl)

    kl_per_dim = np.array(kl_per_dim)
    top_dims = np.argsort(kl_per_dim)[::-1][:n_dims_show]

    fig, axes = plt.subplots(2, n_dims_show // 2, figsize=(14, 6))
    axes = axes.flatten()

    for i, d in enumerate(top_dims):
        ax = axes[i]
        ax.hist(mus_sim[:, d],  bins=40, color=COLOR_SIM,  alpha=0.55,
                density=True, label="Sim")
        ax.hist(mus_real[:, d], bins=40, color=COLOR_REAL, alpha=0.55,
                density=True, label="Real")

        # Gaussian fit
        for data, color in [(mus_sim[:, d], COLOR_SIM), (mus_real[:, d], COLOR_REAL)]:
            mu_fit, s_fit = data.mean(), data.std()
            xs = np.linspace(data.min(), data.max(), 100)
            ax.plot(xs, stats.norm.pdf(xs, mu_fit, s_fit),
                    color=color, linewidth=1.8, linestyle="--")

        ax.set_title(f"dim {d}  (KL={kl_per_dim[d]:.3f})", fontsize=9)
        ax.set_xlabel("µ value", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"Top-{n_dims_show} latent dims với KL divergence cao nhất (Sim vs Real)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return kl_per_dim


def plot_kl_per_dimension(kl_per_dim, save_path="kl_per_dimension.png"):
    """Bar chart KL divergence cho tất cả 32 chiều latent."""
    fig, ax = plt.subplots(figsize=(12, 4))
    dims = np.arange(len(kl_per_dim))
    colors = plt.cm.Reds(kl_per_dim / kl_per_dim.max())
    ax.bar(dims, kl_per_dim, color=colors, edgecolor="none")
    ax.axhline(kl_per_dim.mean(), color="gray", linestyle="--",
               linewidth=1, label=f"Mean KL = {kl_per_dim.mean():.3f}")
    ax.set_xlabel("Latent dimension (0–31)")
    ax.set_ylabel("KL divergence  KL(Sim || Real)")
    ax.set_title("KL divergence per latent dimension: Sim vs Real")
    ax.legend()
    ax.set_xticks(dims)
    ax.set_xticklabels(dims, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_mu_sigma_scatter(mus_sim, sigmas_sim, mus_real, sigmas_real,
                          save_path="mu_sigma_scatter.png"):
    """
    Scatter mean(µ) vs mean(σ) per latent dimension.
    Cho thấy chiều nào sim vs real lệch nhau về cả mean lẫn uncertainty.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, label, mus, sigmas, color in [
        (axes[0], "Sim",  mus_sim,  sigmas_sim,  COLOR_SIM),
        (axes[1], "Real", mus_real, sigmas_real, COLOR_REAL),
    ]:
        dim_mu    = mus.mean(axis=0)     # (32,)
        dim_sigma = sigmas.mean(axis=0)  # (32,)
        sc = ax.scatter(dim_mu, dim_sigma, c=np.arange(32),
                        cmap="viridis", s=60, zorder=3)
        for d in range(32):
            ax.annotate(str(d), (dim_mu[d], dim_sigma[d]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")
        ax.set_xlabel("Mean(µ) across samples")
        ax.set_ylabel("Mean(σ) across samples")
        ax.set_title(f"{label}: µ vs σ per latent dim")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                   label="σ=1 (prior)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        plt.colorbar(sc, ax=ax, label="dim index")

    plt.suptitle("µ vs σ per latent dimension: so sánh Sim và Real", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pca_variance(reduced, save_path="pca_variance.png"):
    """Scree plot của PCA explained variance."""
    evr = reduced["explained_variance_ratio"]
    cumulative = np.cumsum(evr)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(evr)), evr * 100, color=COLOR_SIM, alpha=0.7, label="Per component")
    ax.plot(range(len(evr)), cumulative * 100,
            color=COLOR_REAL, linewidth=2, marker="o", markersize=4, label="Cumulative")
    ax.axhline(95, color="gray", linestyle="--", linewidth=1, label="95% threshold")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance — latent space µ (Sim + Real combined)")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# 4. METRICS ĐỊNH LƯỢNG
# ─────────────────────────────────────────────

def compute_mmd(X, Y, kernel="rbf", sigma=1.0):
    """
    Maximum Mean Discrepancy giữa hai tập điểm X (N, D) và Y (M, D).
    MMD = 0 khi hai phân phối giống hệt nhau.
    Dùng RBF kernel: k(x,y) = exp(-||x-y||² / (2σ²))
    """
    def rbf(A, B, s):
        diff = A[:, None, :] - B[None, :, :]       # (N, M, D)
        sq   = (diff ** 2).sum(-1)                  # (N, M)
        return np.exp(-sq / (2 * s ** 2))

    XX = rbf(X, X, sigma).mean()
    YY = rbf(Y, Y, sigma).mean()
    XY = rbf(X, Y, sigma).mean()
    return XX + YY - 2 * XY


def compute_metrics(mus_sim, mus_real, sigmas_sim, sigmas_real):
    """
    Tổng hợp các metrics định lượng để báo cáo.
    """
    metrics = {}

    # --- KL divergence per dim (Gaussian approx) ---
    kl_dims = []
    for d in range(mus_sim.shape[1]):
        mu1, s1 = mus_sim[:, d].mean(),  mus_sim[:, d].std()  + 1e-8
        mu2, s2 = mus_real[:, d].mean(), mus_real[:, d].std() + 1e-8
        kl = np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5
        kl_dims.append(kl)
    kl_dims = np.array(kl_dims)
    metrics["kl_mean"]   = kl_dims.mean()
    metrics["kl_max"]    = kl_dims.max()
    metrics["kl_max_dim"]= int(kl_dims.argmax())

    # --- MMD (trên PCA 10D để tiết kiệm compute) ---
    all_mus    = np.concatenate([mus_sim, mus_real], axis=0)
    pca_small  = PCA(n_components=min(10, all_mus.shape[1]))
    all_pca    = pca_small.fit_transform(StandardScaler().fit_transform(all_mus))
    N = len(mus_sim)
    mmd = compute_mmd(all_pca[:N], all_pca[N:], sigma=1.0)
    metrics["mmd"] = mmd

    # --- Mean và std của |µ| per domain ---
    metrics["sim_mu_abs_mean"]   = np.abs(mus_sim).mean()
    metrics["real_mu_abs_mean"]  = np.abs(mus_real).mean()
    metrics["sim_sigma_mean"]    = sigmas_sim.mean()
    metrics["real_sigma_mean"]   = sigmas_real.mean()

    # --- Posterior collapse check: dims với sigma ≈ 1 (không học) ---
    sim_collapsed  = (sigmas_sim.mean(axis=0)  > 0.95).sum()
    real_collapsed = (sigmas_real.mean(axis=0) > 0.95).sum()
    metrics["sim_collapsed_dims"]  = int(sim_collapsed)
    metrics["real_collapsed_dims"] = int(real_collapsed)

    return metrics, kl_dims


def print_metrics_report(metrics):
    print("\n" + "="*55)
    print("  LATENT SPACE COMPARISON REPORT: Sim vs Real")
    print("="*55)
    print(f"  KL divergence (mean over 32 dims): {metrics['kl_mean']:.4f}")
    print(f"  KL divergence (max):               {metrics['kl_max']:.4f}  (dim {metrics['kl_max_dim']})")
    print(f"  MMD (PCA-10D, RBF kernel):         {metrics['mmd']:.6f}")
    print(f"  |µ| mean — sim:  {metrics['sim_mu_abs_mean']:.4f}  | real: {metrics['real_mu_abs_mean']:.4f}")
    print(f"  σ mean   — sim:  {metrics['sim_sigma_mean']:.4f}  | real: {metrics['real_sigma_mean']:.4f}")
    print(f"  Posterior collapse dims (σ≈1):")
    print(f"    sim:  {metrics['sim_collapsed_dims']}/32 dims")
    print(f"    real: {metrics['real_collapsed_dims']}/32 dims")
    print("="*55)

    # Interpretation
    print("\n  Diễn giải nhanh:")
    if metrics["kl_mean"] < 0.1:
        print("  ✓ KL thấp → sim và real có phân phối latent gần nhau")
    elif metrics["kl_mean"] < 0.5:
        print("  ~ KL trung bình → có domain gap, nên fine-tune")
    else:
        print("  ✗ KL cao → domain gap lớn, sim-to-real transfer khó")

    if metrics["mmd"] < 0.01:
        print("  ✓ MMD thấp → latent space overlap tốt")
    else:
        print(f"  ! MMD = {metrics['mmd']:.4f} → phân phối sim/real khác nhau đáng kể")

    if metrics["sim_collapsed_dims"] > 8:
        print(f"  ! Sim: {metrics['sim_collapsed_dims']} dims bị posterior collapse (không học)")


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_latent_analysis(
    model,
    sim_dataloader,
    real_dataloader,
    dataset_stats=None,
    device="cuda",
    output_dir=".",
    max_batches=None,
):
    """
    Entry point chính. Gọi hàm này với model ACT đã load và 2 dataloader.

    Ví dụ sử dụng:
        from act import ACTPolicy
        model = ACTPolicy.from_pretrained("path/to/checkpoint")
        run_latent_analysis(model, sim_loader, real_loader, device="cuda")
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    p = lambda name: os.path.join(output_dir, name)

    normalizer = None
    if dataset_stats is not None:
        from lerobot.processor import NormalizerProcessorStep
        normalizer = NormalizerProcessorStep(
            features={**model.config.input_features, **model.config.output_features},
            norm_map=model.config.normalization_mapping,
            stats=dataset_stats,
            device=device,
        )

    print("1. Collecting latents from sim dataset...")
    mus_sim, sigmas_sim = collect_latents(model, sim_dataloader, device, max_batches, normalizer)
    print(f"   Sim: {mus_sim.shape}  (N_samples, latent_dim)")

    print("2. Collecting latents from real dataset...")
    mus_real, sigmas_real = collect_latents(model, real_dataloader, device, max_batches, normalizer)
    print(f"   Real: {mus_real.shape}")

    print("3. Computing metrics...")
    metrics, kl_dims = compute_metrics(mus_sim, mus_real, sigmas_sim, sigmas_real)
    print_metrics_report(metrics)

    print("4. Reducing dimensions (PCA + t-SNE)...")
    reduced = reduce_dimensions(mus_sim, mus_real)

    print("5. Plotting...")
    plot_pca_variance(reduced,                      p("pca_variance.png"))
    plot_tsne_scatter(reduced,                      p("tsne_sim_vs_real.png"))
    plot_per_dimension_distribution(mus_sim, mus_real, save_path=p("latent_per_dim.png"))
    plot_kl_per_dimension(kl_dims,                  p("kl_per_dimension.png"))
    plot_mu_sigma_scatter(mus_sim, sigmas_sim,
                          mus_real, sigmas_real,    p("mu_sigma_scatter.png"))

    print(f"\nDone. Tất cả output trong: {output_dir}/")
    return metrics, reduced, kl_dims


# ─────────────────────────────────────────────
# DEMO: Chạy với dữ liệu giả để test pipeline
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== DEMO với dữ liệu giả (synthetic) ===\n")

    np.random.seed(42)
    # Giả lập sim: latent gần prior N(0,1)
    mus_sim    = np.random.randn(500, 32) * 0.8
    sigmas_sim = np.ones((500, 32)) * 0.7 + np.random.randn(500, 32) * 0.05

    # Giả lập real: bị shift ở một số dim, phân tán hơn
    mus_real    = np.random.randn(300, 32) * 1.2
    mus_real[:, 3]  += 1.5   # dim 3 shift lớn
    mus_real[:, 17] -= 1.0   # dim 17 shift vừa
    sigmas_real = np.ones((300, 32)) * 0.9 + np.random.randn(300, 32) * 0.08

    metrics, kl_dims = compute_metrics(mus_sim, mus_real, sigmas_sim, sigmas_real)[0], \
                       compute_metrics(mus_sim, mus_real, sigmas_sim, sigmas_real)[1]
    print_metrics_report(metrics)

    reduced = reduce_dimensions(mus_sim, mus_real)

    import os
    os.makedirs("/mnt/user-data/outputs/latent_plots", exist_ok=True)
    p = lambda n: f"/mnt/user-data/outputs/latent_plots/{n}"

    plot_pca_variance(reduced,                      p("pca_variance.png"))
    plot_tsne_scatter(reduced,                      p("tsne_sim_vs_real.png"))
    plot_per_dimension_distribution(mus_sim, mus_real, save_path=p("latent_per_dim.png"))
    plot_kl_per_dimension(kl_dims,                  p("kl_per_dimension.png"))
    plot_mu_sigma_scatter(mus_sim, sigmas_sim,
                          mus_real, sigmas_real,    p("mu_sigma_scatter.png"))
    print("\nDemo hoàn thành.")
