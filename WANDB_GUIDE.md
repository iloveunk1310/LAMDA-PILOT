# Hướng dẫn Thiết lập và Sử dụng Weights & Biases (W&B) trên LAMDA-PILOT

Tài liệu này hướng dẫn chi tiết cách cấu hình và sử dụng **Weights & Biases (W&B)** để theo dõi thí nghiệm học tăng cường (Class-Incremental Learning) trực tuyến trên dự án Public của nhóm, đồng thời đảm bảo quy ước đồng bộ dữ liệu khi chạy chung nhiều repository nghiên cứu khác nhau.

---

## 1. Thông tin Dự án Chung của Nhóm

- **W&B Entity (Tài khoản / Team)**: `nnm-hcmus`
- **W&B Project**: `Group7-ML-Lab`
- **Chế độ hiển thị (Privacy)**: **Public** (Mọi thành viên đều có thể xem dashboard và đồ thị so sánh trực tiếp)
- **Đường dẫn Dashboard**: [https://wandb.ai/nnm-hcmus/Group7-ML-Lab](https://wandb.ai/nnm-hcmus/Group7-ML-Lab)

---

## 2. Các bước Người dùng Tự Thực hiện (Manual Setup)

### Bước 1: Tạo tài khoản & Lấy API Key
1. Truy cập [https://wandb.ai](https://wandb.ai) để đăng ký hoặc đăng nhập tài khoản W&B.
2. Mở đường dẫn [https://wandb.ai/authorize](https://wandb.ai/authorize) để sao chép mã **API Key** cá nhân (chuỗi 40 ký tự).

### Bước 2: Cài đặt và Đăng nhập trong Môi trường Conda
Mở terminal (PowerShell hoặc Command Prompt) và kích hoạt môi trường ảo:

```powershell
# Kích hoạt môi trường conda đã cài đặt (pilot)
conda activate pilot

# Cài đặt gói wandb (nếu chưa cài)
pip install wandb

# Xác thực tài khoản với API Key
wandb login
```
> *(Dán API Key đã sao chép ở Bước 1 vào terminal khi được yêu cầu và nhấn Enter)*.

### Bước 3: Cấu hình Dự án Public trên Giao diện Web
1. Truy cập vào project [https://wandb.ai/nnm-hcmus/Group7-ML-Lab](https://wandb.ai/nnm-hcmus/Group7-ML-Lab).
2. Vào mục **Project Settings** (biểu tượng bánh răng ở thanh menu bên trái).
3. Tại phần **Privacy**, chuyển trạng thái sang **Public (Open to everyone)** để kết quả được chia sẻ công khai với các thành viên.

---

## 3. Quy ước Chạy Chung Nhiều Repository trên 1 Public Project

Để so sánh trực tiếp đồ thị và bảng chỉ số giữa **LAMDA-PILOT** với các repository Continual Learning khác (ví dụ: *PyCIL*, *FACIL*, v.v.) trên cùng project `Group7-ML-Lab`, hệ thống tuân theo các quy ước sau:

### 3.1. Gom nhóm (Grouping) & Định danh (Run Name)
- **Group**: Mặc định là `"LAMDA-PILOT"` (giúp lọc riêng các runs của repo này hoặc gom nhóm theo từng phương pháp).
- **Tên Run chuẩn hóa**: `PILOT_{model_name}_{dataset}_b{init_cls}inc{increment}_s{seed}`
  - Ví dụ: `PILOT_simplecil_cifar224_b0inc10_s1993`
  - Riêng `Finetune Adapter` (`model_name="finetune"` với `backbone_type` chứa `"adapter"`) được gắn nhãn tự động là `finetune_adapter` (ví dụ: `PILOT_finetune_adapter_cifar224_b0inc5_s1993`, tag `method:finetune_adapter`) để phân biệt với `finetune` toàn bộ backbone.
- **Metadata tự động đính kèm**:
  - `repo_name`: `"LAMDA-PILOT"`
  - `git_commit`: Mã hash commit hiện tại từ git
  - `tags`: `["repo:LAMDA-PILOT", "method:<model_name>", "dataset:<dataset>", "backbone:<backbone_type>"]`

### 3.2. Bảng Chỉ số Chuẩn hóa (Unified Metric Schema)
Sau mỗi incremental task, các chỉ số sau được đẩy lên W&B đồng bộ theo `step=task`:
- `task`: Thứ tự task (0, 1, 2, ...)
- `eval/top1_acc`: Độ chính xác Top-1 CNN (%)
- `eval/top5_acc`: Độ chính xác Top-5 CNN (%)
- `eval/avg_acc`: Độ chính xác trung bình tích lũy (%)
- `params/total`: Tổng số tham số mô hình
- `params/trainable`: Số tham số có thể huấn luyện
- *(Nếu phương pháp hỗ trợ NME)*:
  - `eval/nme_top1_acc`
  - `eval/nme_top5_acc`
  - `eval/nme_avg_acc`
- *(Sau khi hoàn thành tất cả các task)*:
  - `eval/forgetting`: Mức độ suy giảm độ chính xác qua các task (Catastrophic Forgetting)

---

## 4. Hướng dẫn Chạy Thí nghiệm Thực tế

### 4.1. Chạy với cấu hình mặc định (Đẩy trực tiếp lên W&B)
Chạy thuật toán SimpleCIL trên CIFAR-100 (rescaled 224x224):

```powershell
python main.py --config ./exps/simplecil.json --use_wandb
```

### 4.2. Ghi đè thông số qua tham số dòng lệnh CLI
Bạn có thể tùy biến batch size, learning rate, project, entity, group hoặc thêm tags mà không cần sửa file JSON:

```powershell
python main.py --config ./exps/simplecil.json \
    --batch_size 32 \
    --init_lr 0.01 \
    --use_wandb \
    --wandb_project Group7-ML-Lab \
    --wandb_entity nnm-hcmus \
    --wandb_group LAMDA-PILOT \
    --wandb_tags vit-b16 baseline
```

### 4.3. Chế độ Chạy Ngoại Tuyến (Offline Mode)
Khi đường truyền mạng không ổn định hoặc muốn huấn luyện offline và đồng bộ lên W&B sau:

```powershell
# Chạy huấn luyện lưu dữ liệu cục bộ vào thư mục wandb/
python main.py --config ./exps/simplecil.json --use_wandb --wandb_mode offline

# Sau khi huấn luyện xong, đồng bộ dữ liệu lên cloud:
wandb sync ./wandb/latest-run
```

### 4.4. Tắt W&B Logging
Để chạy kiểm thử nhanh cục bộ mà không đẩy dữ liệu lên W&B:

```powershell
python main.py --config ./exps/simplecil.json --no_wandb
```
*(Hoặc không truyền `--use_wandb` nếu trong file JSON `use_wandb` là `false`)*.

---

## 5. Quy trình Benchmark Suite 6 Dataset & 12 Phương pháp trên Google Colab

Để chạy thực nghiệm đồng bộ theo chuẩn `B0 Inc-n` trên 6 bộ dữ liệu (đã loại bỏ `ObjectNet`) với trọn vẹn 12 phương pháp mục tiêu, sử dụng quy trình 3 bước đọc trực tiếp từ `dataset.txt` và `exps/benchmarks.json`.

### 5.1. Quy trình 3 bước trên Google Colab

```bash
# Bước 1: Tải file nén từ dataset.txt (mặc định tải cả 5 bộ ngoài CIFAR-100 vào ./data/archives)
# Có thể lọc bằng --dataset hoặc --benchmark; dùng --archive_dir để lưu trên Google Drive
!python download_data.py --benchmark all_6 --archive_dir ./data/archives

# Bước 2: Giải nén và chuẩn hóa cây thư mục train/ và test/ vào ./data/
# Dùng --cleanup để xóa file nén sau khi xác nhận thư mục đích hợp lệ
!python extract_data.py --benchmark all_6 --archive_dir ./data/archives --data_dir ./data --cleanup

# Bước 3: Chạy huấn luyện theo phương pháp (--method) và preset hoặc cụm (--benchmark)
!python main.py --method finetune_adapter --benchmark fast --use_wandb
```

> **Lưu ý:**
> - `cifar224` (CIFAR-100) tự động tải qua `torchvision` trong lúc chạy `main.py`, nên không nằm trong `dataset.txt` (`download_data.py --dataset cifar224` sẽ thông báo bỏ qua).
> - Nếu muốn đổi nguồn tải (ví dụ sang Google Drive cá nhân khi bị giới hạn lượt tải), chỉ cần sửa URL tương ứng trong `dataset.txt` (`<url> <filename>`).
> - Dùng `--dry_run` trên `main.py` để kiểm tra cấu hình đã phân giải trước khi huấn luyện:
>   `!python main.py --method finetune_adapter --benchmark all_6 --dry_run`
> - Có thể ghi đè trực tiếp dữ liệu hoặc backbone qua CLI: `--dataset`, `--init_cls`, `--increment`, `--backbone_type`, `--batch_size`, `--init_lr`.

### 5.2. Danh sách 6 Preset & 4 Cụm Thực thi (`exps/benchmarks.json`)

| Cụm (`Cluster`) | Preset ID | Dataset (`utils/data.py`) | Số lớp | `init_cls` / `increment` | Số tasks | File nguồn (`dataset.txt`) |
| :--- | :--- | :--- | :-: | :-: | :-: | :--- |
| `fast` | `vtab_b0inc10` | `vtab` | 50 | `10` / `10` | 5 | `vtab.zip` |
| `fast` | `cifar_b0inc5` | `cifar224` | 100 | `5` / `5` | 20 | *(torchvision tự tải)* |
| `inc10_200c` | `cub_b0inc10` | `cub` | 200 | `10` / `10` | 20 | `cub.zip` |
| `inc10_200c` | `ina_b0inc10` | `imageneta` | 200 | `10` / `10` | 20 | `imagenet-a.zip` |
| `long_or_large` | `omnibench_b0inc30` | `omnibenchmark` | 300 | `30` / `30` | 10 | `omnibenchmark.zip` |
| `long_or_large` | `inr_b0inc5` | `imagenetr` | 200 | `5` / `5` | 40 | `imagenet-r.zip` |
| `all_6` | *(Toàn bộ 6 preset trên)* | — | — | — | — | — |

### 5.3. Danh sách 12 Phương pháp Hỗ trợ (`--method`)

- **Baselines & Prompt / PEFT:** `finetune`, `finetune_adapter`, `l2p`, `dualprompt`, `coda_prompt`, `lae`
- **RevisitingCIL (`SimpleCIL` & `APER`):** `simplecil`, `aper_finetune`, `aper_ssf`, `aper_vpt_shallow`, `aper_vpt_deep`, `aper_adapter`

*(Khi chạy `cifar224`, `main.py` tự động chọn template `<method>.json`; với 5 bộ ảnh 224x224 còn lại, `main.py` tự động chọn `<method>_inr.json`.)*

---

## 6. Lưu ý Tối ưu cho Phần Cứng Quadro T1000 (4GB VRAM) trên Windows 11

Nếu gặp lỗi **CUDA Out of Memory (OOM)** hoặc đóng băng luồng dữ liệu trên Windows:
1. **Batch Size**: Giảm `batch_size` trong file cấu hình JSON xuống `64` hoặc `32` (mặc định của ViT là 256 thường chiếm nhiều VRAM).
2. **Num Workers**: Đối với Windows, đặt `num_workers: 0` hoặc `2` trong data loader để tránh lỗi đa tiến trình bộ nhớ dùng chung.

