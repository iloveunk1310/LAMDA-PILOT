# Proposal

## Why

Hiện tại, LAMDA-PILOT chỉ ghi kết quả huấn luyện học tăng cường (Class-Incremental Learning) ra màn hình console và các file văn bản cục bộ trong thư mục `logs/`. Việc tích hợp **Weights & Biases (W&B)** cho phép theo dõi thí nghiệm trực tuyến theo thời gian thực trên một **Public W&B Project** (`https://wandb.ai`), đồng thời chuẩn hóa cấu trúc dữ liệu (metrics, group, tags) để có thể chạy chung và so sánh trực tiếp kết quả giữa LAMDA-PILOT với các repository nghiên cứu khác trên cùng một bảng điều khiển (dashboard).

## What Changes

- Bổ sung tính năng theo dõi thí nghiệm tùy chọn với **Weights & Biases (`wandb`)** vào luồng huấn luyện chính (`main.py` và `trainer.py`), điều khiển qua file cấu hình JSON hoặc tham số dòng lệnh CLI (`--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_group`, `--wandb_tags`).
- Chuẩn hóa **Unified Metric Schema** theo từng incremental task (`task`, `eval/top1_acc`, `eval/top5_acc`, `eval/avg_acc`, `eval/nme_top1_acc`, `eval/nme_avg_acc`, `eval/forgetting`, `params/total`, `params/trainable`) giúp đồng bộ trục hoành và tên chỉ số khi vẽ biểu đồ so sánh đa repo.
- Tự động đính kèm metadata nhận diện nguồn gốc (`repo_name: "LAMDA-PILOT"`, `git_commit`, `group="LAMDA-PILOT"`, và các `tags` theo mô hình/tập dữ liệu/backbone) vào mỗi W&B run.
- Giữ tương thích ngược hoàn toàn: Khi `use_wandb` là `false` hoặc không được khai báo, code hoạt động bình thường mà không yêu cầu cài đặt thư viện `wandb`.
- Cung cấp tài liệu hướng dẫn chi tiết các bước thiết lập thủ công dành cho người dùng (tạo tài khoản/dự án W&B Public, lấy API Key, đăng nhập `wandb login` và cấu hình).

## Capabilities

### New Capabilities
- `experiment-tracking`: Tích hợp Weights & Biases (W&B) để ghi nhận cấu hình, tham số mô hình, chỉ số đánh giá qua từng task tăng cường (incremental tasks) và phân loại đa repository trên một Public Project.

### Modified Capabilities
<!-- Không có capability sẵn có nào bị thay đổi yêu cầu -->

## Impact

- **Mã nguồn bị ảnh hưởng**:
  - `main.py`: Thêm các đối số dòng lệnh tùy chọn cho W&B để ghi đè nhanh mà không cần sửa từng file JSON.
  - `trainer.py`: Khởi tạo `wandb.init()` đầu mỗi seed run, gọi `wandb.log()` sau mỗi task đánh giá, ghi nhận chỉ số `forgetting` cuối cùng vào `wandb.summary`, và đóng kết nối với `wandb.finish()`.
  - `exps/*.json` (ví dụ `exps/simplecil.json`): Bổ sung các trường cấu hình mẫu cho W&B.
- **Thư viện phụ thuộc (Dependencies)**:
  - Thêm gói `wandb` (chỉ import khi `use_wandb=True`).
- **Thao tác từ phía người dùng (User Manual Setup)**:
  - Yêu cầu đăng ký tài khoản tại `https://wandb.ai`, lấy API Key, tạo hoặc cấu hình project ở chế độ **Public**, và xác thực trên máy chạy.

