# Cài đặt môi trường LeRobot

Repo này chạy trên nền `lerobot` đã cài (xem [`reference/README.md`](../reference/README.md)
để biết phần nào của `lerobot` được vendor sẵn để đọc, và phần nào vẫn cần cài
qua `pip`/`uv`). Trang này tóm tắt lại hướng dẫn cài đặt chính thức của LeRobot
([huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation)),
rút gọn cho đúng nhu cầu của repo này: ACT + Diffusion Policy, SO-101
(`feetech`), và mô phỏng MuJoCo.

## Bước 1 — Môi trường ảo Python (>= 3.12)

Chọn 1 trong 2 cách:

**`conda`** (khuyến nghị nếu chưa quen `uv`):

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

conda create -y -n lerobot python=3.12
conda activate lerobot   # chạy lại lệnh này mỗi khi mở shell mới
```

**`uv`** (repo này ưu tiên dùng `uv`, xem `AGENTS.md`/`CLAUDE.md` của `lerobot`):

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\activate
```

> **WSL (Windows Subsystem for Linux):** cài thêm `evdev`:
> `conda install evdev -c conda-forge` (conda) hoặc
> `sudo apt install libevdev-dev && uv pip install evdev` (uv).

## Bước 2 — `ffmpeg` (giải mã video)

LeRobot dùng [TorchCodec](https://github.com/meta-pytorch/torchcodec) để giải
mã video theo mặc định, cần `ffmpeg`.

```bash
# Trong conda env — hoạt động với mọi phiên bản PyTorch
conda install ffmpeg -c conda-forge
# Nếu gặp lỗi thiếu libsvtav1 hoặc lệch phiên bản với torchcodec:
conda install ffmpeg=7.1.1 -c conda-forge
```

```bash
# Với uv/venv thuần và PyTorch >= 2.10 (torchcodec >= 0.10), dùng ffmpeg hệ thống
sudo apt install ffmpeg        # Ubuntu/Debian
brew install ffmpeg            # macOS Apple Silicon
```

> macOS Intel, Linux ARM, hoặc Windows với PyTorch < 2.8: TorchCodec không hỗ
> trợ — LeRobot tự chuyển sang `pyav`, có thể bỏ qua bước cài `ffmpeg`.

## Bước 3 — Cài `lerobot`

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Editable install (khuyến nghị khi chỉnh sửa/đọc kèm code như repo này)
uv pip install -e ".[core_scripts]"   # record, replay, calibrate
uv pip install -e ".[training]"       # train policy
uv pip install -e ".[all]"            # tất cả (mọi policy/env/hardware/dev tools)
```

Các extra hay dùng với repo này:

| Extra | Dùng cho |
|---|---|
| `core_scripts` | `lerobot-record`, `lerobot-replay`, `lerobot-calibrate` |
| `training` | `lerobot-train` |
| `feetech` | Motor SO-100/SO-101 |
| `diffusion` | Dependency riêng của Diffusion Policy (`diffusers`) |

```bash
uv pip install -e ".[core_scripts,training,feetech,diffusion]"
```

Hoặc cài từ PyPI (không cần clone `lerobot`, không có editable/reference source):

```bash
uv pip install 'lerobot[core_scripts,training,feetech,diffusion]'
```

### GPU (Linux, NVIDIA)

Cài `lerobot` qua `uv sync`/`uv pip install -e .` đã tự chọn wheel PyTorch CUDA
12.8 (driver >= 570.86) phù hợp Ampere/Ada/Hopper/Blackwell — không cần làm gì
thêm với GPU đời thường. Nếu cần bản CUDA khác:

```bash
uv pip install --force-reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/cu126   # driver cũ hơn; cu130 cho Blackwell + driver >= 580
```

### Nếu build lỗi (thiếu thư viện hệ thống)

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

## Bước 4 — Cài dependency riêng của repo này

```bash
cd /path/to/so101-sim2real-policy-research
uv pip install -r requirements.txt
```

`requirements.txt` cài các thư viện phân tích (`shap`, `matplotlib`, `pandas`,
`scikit-learn`, ...) và `mujoco` cho phần mô phỏng (`src/sim_mujoco/`).

## Bước 5 — Đăng nhập Hugging Face (để push/pull dataset & policy)

```bash
hf auth login
```

## Kiểm tra cài đặt

```bash
python -c "import lerobot, torch, mujoco, shap; print('OK')"
```

Sau khi cài xong: xem [`docs/so101/so101_setup_lerobot.mdx`](so101/so101_setup_lerobot.mdx)
để setup phần cứng SO-101, hoặc [`docs/pipeline/end_to_end_pipeline.md`](pipeline/end_to_end_pipeline.md)
để chạy pipeline đầy đủ.
