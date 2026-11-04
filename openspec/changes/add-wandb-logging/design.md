# Design

## Context

Xem `proposal.md` để biết động lực thay đổi và `specs/experiment-tracking/spec.md` cho các yêu cầu hành vi.
Trong kiến trúc hiện tại của LAMDA-PILOT:
- `main.py` đọc file cấu hình JSON từ `--config` và ghi đè vào dictionary `args`, sau đó gọi `train(args)` trong `trainer.py`.
- `trainer.py` lặp qua danh sách `args["seed"]` và gọi `_train(args)` cho từng seed. Bên trong `_train(args)`, vòng lặp `for task in range(data_manager.nb_tasks):` thực hiện huấn luyện và đánh giá cho bất kỳ phương pháp nào trong 24 phương pháp của repo.
- Môi trường chạy thực tế của người dùng là Windows 11 với GPU NVIDIA Quadro T1000 (4GB VRAM), và dự án W&B sẽ được dùng chung cho nhiều repository Continual Learning khác nhau.

## Goals / Non-Goals

**Goals:**
- Tích hợp W&B tại một điểm tập trung duy nhất (`trainer.py` và `main.py`) để toàn bộ 24 phương pháp đều tự động có khả năng đẩy kết quả lên W&B mà không cần sửa từng file trong `models/`.
- Cho phép bật/tắt và cấu hình W&B linh hoạt qua cả file JSON (`exps/*.json`) lẫn đối số dòng lệnh CLI (`main.py`).
- Chuẩn hóa hệ thống đặt tên Run, Group, Tags và Metric Schema để phân tách rõ ràng khi chạy chung nhiều repo trên cùng một Public W&B Project.
- Phân định rõ ràng các bước **Người dùng tự thực hiện (Manual Setup)** và **Các bước triển khai trong Code**.

**Non-Goals:**
- Không can thiệp sâu vào vòng lặp epoch/batch bên trong từng file của 24 mô hình (`models/*.py`), nhằm giữ nguyên tính toàn vẹn của code gốc từ nhóm tác giả LAMDA.
- Không tự động lưu trữ các file trọng số `.pkl` dung lượng lớn lên W&B Artifacts để tránh tràn băng thông và giới hạn dung lượng miễn phí của tài khoản W&B.

## Decisions

### 1. Điểm tích hợp tập trung tại `trainer.py` thay vì `models/*.py`
- **Lựa chọn**: Đặt `wandb.init()`, `wandb.log()`, và `wandb.finish()` trong hàm `_train(args)` của `trainer.py` (hoặc tách hàm helper trong `utils/toolkit.py` được gọi bởi `trainer.py`).
- **Lý do**: Tất cả 24 thuật toán đều trả kết quả chuẩn hóa (`cnn_accy`, `nme_accy`) về `trainer.py` sau mỗi task. Đặt hook tại đây giúp hỗ trợ đồng loạt 100% thuật toán chỉ với một lần chỉnh sửa.
- **Phương án thay thế đã cân nhắc**: Thêm `wandb.log` vào `models/base.py` hoặc từng file model — bị loại bỏ vì các lớp con override hàm `_train` rất khác nhau, gây trùng lặp code.

### 2. Cơ chế kết hợp cấu hình JSON và CLI Overrides trong `main.py`
- **Lựa chọn**: Trong `main.py`, bổ sung các tham số tùy chọn (`--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_group`, `--wandb_mode`). Khi người dùng truyền từ CLI, giá trị này sẽ ưu tiên ghi đè lên file JSON; nếu không truyền, hệ thống đọc từ JSON (mặc định `use_wandb = False`).
- **Lý do**: Repo có tới 50 file JSON trong `exps/`. Hỗ trợ CLI giúp người dùng chạy bất kỳ file JSON nào kèm `--use_wandb --wandb_project my-public-project` mà không phải sửa tay cả 50 file.

### 3. Quy ước Metadata để chạy chung nhiều Repo trên 1 Public Project
- **Lựa chọn**:
  - `project`: Mặc định `"continual-learning-benchmark"` (có thể tùy chỉnh).
  - `group`: Mặc định `"LAMDA-PILOT"` (giúp gom nhóm toàn bộ các lần chạy của repo này, tách biệt với `group="PyCIL"` hay repo khác).
  - `name`: `f"PILOT_{model_name}_{dataset}_b{init_cls}inc{increment}_s{seed}"`.
  - `tags`: Tự động sinh `["repo:LAMDA-PILOT", f"method:{model_name}", f"dataset:{dataset}", f"backbone:{backbone_type}"]` cộng với các tag tùy chọn của người dùng.
  - `config`: Lưu toàn bộ dictionary `args` kèm `"repo_name": "LAMDA-PILOT"` và mã hash `"git_commit"` (lấy qua `subprocess` nếu có `.git`).

### 4. Lazy Import cho thư viện `wandb`
- **Lựa chọn**: Chỉ thực hiện `import wandb` khi `args.get("use_wandb", False)` là `True`.
- **Lý do**: Đảm bảo code vẫn chạy bình thường trên các môi trường chưa cài `wandb` nếu người dùng không bật cờ `use_wandb`.

### 5. Phân định trách nhiệm: Những việc Người Dùng Tự Làm vs. Việc Code Tự Động Làm

| Phân loại | Chi tiết công việc |
| :--- | :--- |
| **Người dùng tự thực hiện (Manual)** | 1. Đăng ký/đăng nhập tài khoản tại `https://wandb.ai`.<br>2. Truy cập `https://wandb.ai/authorize` để copy **API Key** cá nhân.<br>3. Tạo Project mới trên web (hoặc để code tự tạo ở lần chạy đầu) và vào **Project Settings -> Privacy -> Chuyển sang Public**.<br>4. Chạy `pip install wandb` và `wandb login` trong môi trường ảo (`pilot` hoặc `minhenv`).<br>5. Điều chỉnh `batch_size` (ví dụ `32` hoặc `64`) và `num_workers` (`0` hoặc `2`) phù hợp với GPU Quadro T1000 4GB VRAM trên Windows. |
| **Code tự động thực hiện (Automated)** | 1. Đọc cấu hình W&B từ CLI/JSON.<br>2. Khởi tạo phiên làm việc với `wandb.init` cho từng seed.<br>3. Đẩy tự động các chỉ số `eval/top1_acc`, `eval/avg_acc`, `eval/forgetting`, số lượng tham số sau mỗi task.<br>4. Đóng phiên `wandb.finish()` an toàn kể cả khi có nhiều seed trong `seed_list`. |

## Risks / Trade-offs

- **[Risk] Mất kết nối mạng giữa chừng làm gián đoạn quá trình huấn luyện dài** → **Mitigation**: Hỗ trợ tham số `wandb_mode` (`"online"`, `"offline"`, `"disabled"`). Nếu mạng không ổn định, người dùng có thể chạy với `--wandb_mode offline` và đồng bộ lên cloud sau bằng lệnh `wandb sync`.
- **[Risk] Chạy nhiều seed trong `args["seed"]` bị gộp chung vào 1 run duy nhất** → **Mitigation**: Đặt `wandb.init(..., reinit=True)` ở đầu hàm `_train(args)` và bọc `wandb.finish()` trong khối `try...finally` để đảm bảo mỗi seed tạo một Run độc lập.

