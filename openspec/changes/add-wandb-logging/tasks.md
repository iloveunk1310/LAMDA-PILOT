# Tasks

## 1. Cấu hình CLI và JSON cho Weights & Biases (`main.py` & `exps/`)

- [ ] 1.1 Mở rộng `setup_parser()` và `main()` trong `main.py` để hỗ trợ các đối số dòng lệnh tùy chọn (`--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_group`, `--wandb_tags`, `--wandb_mode`) sao cho giá trị CLI chỉ ghi đè lên JSON khi được truyền vào; kiểm tra bằng lệnh `python main.py --help` hiển thị đầy đủ các cờ mới.
- [ ] 1.2 Cập nhật file cấu hình mẫu `exps/simplecil.json` bổ sung các trường mặc định `"use_wandb": false`, `"wandb_project": "continual-learning-benchmark"`, `"wandb_entity": null`, `"wandb_group": "LAMDA-PILOT"`, `"wandb_tags": []` và kiểm tra file JSON parse hợp lệ bằng `python -c "import json; json.load(open('exps/simplecil.json'))"`.

## 2. Tích hợp W&B Lifecycle và Unified Metric Logging (`trainer.py` & `utils/toolkit.py`)

- [ ] 2.1 Thêm hàm tiện ích lấy `git_commit` và khởi tạo W&B run (`init_wandb(args)`) với lazy import `wandb`, tự động gán `group="LAMDA-PILOT"`, `repo_name="LAMDA-PILOT"`, tên run chuẩn hóa và danh sách `tags`; kiểm tra bằng đoạn script test chạy với `wandb_mode="offline"` hoặc `"disabled"`.
- [ ] 2.2 Tích hợp `wandb.log()` vào vòng lặp từng task trong `_train(args)` của `trainer.py` để ghi nhận `task`, `eval/top1_acc`, `eval/top5_acc`, `eval/avg_acc`, `params/total`, `params/trainable` (và `eval/nme_top1_acc`, `eval/nme_top5_acc`, `eval/nme_avg_acc` khi có NME); kiểm tra các key được đẩy đúng cấu trúc sau mỗi task.
- [ ] 2.3 Ghi nhận chỉ số `eval/forgetting` (CNN và NME nếu `print_forget=True`) và đảm bảo gọi `wandb.finish()` trong khối `try...finally` cuối mỗi seed run trong `_train(args)`; kiểm tra chạy thử với `use_wandb=False` và `use_wandb=True` (`--wandb_mode offline`) đều kết thúc không lỗi.

## 3. Tài liệu Hướng dẫn Thiết lập Thủ công & Đa Repo (`WANDB_GUIDE.md`)

- [ ] 3.1 Tạo file hướng dẫn `WANDB_GUIDE.md` trình bày chi tiết từng bước người dùng tự thực hiện (Tạo tài khoản `wandb.ai`, lấy API Key tại `wandb.ai/authorize`, đăng nhập `wandb login`, chuyển Project sang chế độ **Public**, và quy ước chạy chung nhiều repo trên cùng 1 project); kiểm tra file được tạo và đầy đủ các lệnh mẫu.

## Workflow follow-up (Phần việc Người Dùng Tự Thực Hiện Thủ Công)

- **Bước A (Tạo tài khoản & Lấy API Key)**: Truy cập https://wandb.ai để đăng ký/đăng nhập tài khoản, sau đó mở https://wandb.ai/authorize để sao chép khóa API (40 ký tự).
- **Bước B (Cài đặt & Đăng nhập W&B trong môi trường ảo)**: Kích hoạt môi trường ảo (`conda activate pilot` hoặc `conda activate minhenv`), chạy `pip install wandb` và `wandb login`, sau đó dán API Key vào terminal.
- **Bước C (Tạo Public Project trên W&B)**: Tạo một project mới trên giao diện web W&B (ví dụ: `continual-learning-benchmark`), vào **Settings -> Privacy -> Chọn Public (Open to everyone)** để công khai kết quả.
- **Bước D (Chạy thử nghiệm thực tế)**: Chạy lệnh `python main.py --config=./exps/simplecil.json --use_wandb --wandb_project continual-learning-benchmark` và kiểm tra đồ thị trực tiếp trên đường dẫn Public Project.

