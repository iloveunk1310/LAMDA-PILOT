# Proposal

## Why

Hiện tại thư mục `exps/` mới chỉ cung cấp sẵn cấu hình cho `cifar224` (`B0 Inc10`) và `imagenetr` (`B0 Inc20`), chưa hỗ trợ đồng bộ 6 bộ dataset benchmark mục tiêu theo thiết lập `B0 Inc-n` (`CIFAR B0 Inc5`, `CUB B0 Inc10`, `IN-R B0 Inc5`, `IN-A B0 Inc10`, `OmniBench B0 Inc30`, `VTAB B0 Inc10` — loại bỏ `ObjectNet` do dung lượng quá nặng) và chưa cho phép chạy `Finetune Adapter` trực tiếp với `model_name: "finetune"`. Ngoài ra, khi chạy trên môi trường tạm thời theo từng phiên như Google Colab, việc tải và giải nén 5 bộ dữ liệu ngoài CIFAR (đã được liệt kê sẵn trong `dataset.txt`) cần được tự động hóa gọn nhẹ. Chuẩn hóa kiến trúc preset, gom cụm cấu hình, và bổ sung cặp script đọc trực tiếp `dataset.txt` giúp toàn bộ quy trình Colab rút gọn thành 3 bước (`!python download_data.py` -> `!python extract_data.py` -> `!python main.py`) cho trọn vẹn 12 phương pháp (`Finetune`, `Finetune Adapter`, `L2P`, `DualPrompt`, `CODA-Prompt`, `LAE`, `SimpleCIL`, cùng các biến thể `APER`) với bộ siêu tham số mặc định.

## What Changes

- Tạo file cấu hình tập trung `exps/benchmarks.json` định nghĩa 6 preset dataset chuẩn (`cifar_b0inc5`, `cub_b0inc10`, `inr_b0inc5`, `ina_b0inc10`, `omnibench_b0inc30`, `vtab_b0inc10`), 4 cụm gom nhóm chạy tuần tự (`fast`, `inc10_200c`, `long_or_large`, `all_6`), và bảng ánh xạ 12 phương pháp mục tiêu.
- Thêm 2 script dòng lệnh đơn giản ở thư mục gốc đọc trực tiếp từ [`dataset.txt`](file:///d:/tai_lieu_hoc%20(1)/base_ML/LAMDA-PILOT/dataset.txt) (áp dụng cho 5 bộ ngoài `cifar224`, vì `cifar224` đã tự tải bởi `torchvision`):
  - `download_data.py`: Đọc từng dòng `<url> <filename>` trong `dataset.txt` (`cub.zip`, `imagenet-r.zip`, `imagenet-a.zip`, `omnibenchmark.zip`, `vtab.zip`), tự động tải link Google Drive (qua `gdown`) hoặc link SharePoint/OneDrive (`&download=1`) vào `--archive_dir` (mặc định `./data/archives`); hỗ trợ lọc theo tên (`--dataset`) và bỏ qua file đã tồn tại.
  - `extract_data.py`: Giải nén các file archive (`.zip`/`.tar`/`.tar.gz`) tương ứng từ `dataset.txt` vào `--data_dir` (mặc định `./data`), tự chuẩn hóa cấu trúc `train/` và `test/` khớp chính xác đường dẫn đọc của `utils/data.py` (kể cả `./data/vtab-cil/vtab/`), hỗ trợ `--cleanup` xóa file nén sau khi giải nén xong.
- Mở rộng `main.py` thêm các đối số CLI (`--method`, `--benchmark`, `--dataset`, `--init_cls`, `--increment`, `--backbone_type`) cho phép chạy đơn lẻ 1 preset hoặc chạy nối tiếp cả cụm dataset, tự động chọn file template siêu tham số mặc định phù hợp (`<model>.json` cho `cifar224` và `<model>_inr.json` cho 5 bộ ảnh tự nhiên 224x224).
- Hỗ trợ trực tiếp phương pháp **Finetune Adapter**: Cập nhật `utils/inc_net.py` (`get_backbone`) cho phép `model_name == "finetune"` sử dụng backbone `"pretrained_vit_b16_224_adapter_in21k"` / `"pretrained_vit_b16_224_in21k_adapter"` (cùng các biến thể adapter khác) với `ffn_num` mặc định là `64`; tối ưu `models/finetune.py` chỉ cập nhật các tham số `requires_grad=True`; bổ sung `exps/finetune_adapter.json` và `exps/finetune_adapter_inr.json`.
- Cập nhật `utils/toolkit.py` (`init_wandb`) để tự động phân biệt tên run và tag giữa `finetune_adapter` với `finetune` thông thường khi đẩy kết quả lên W&B.
- Sửa lỗi cấu hình gốc trong `exps/aper_vpt_shallow_inr.json` (đang thiếu hậu tố `_vpt` ở `backbone_type` và cấu hình nhầm `vpt_type` thành `"deep"`).

## Capabilities

### New Capabilities
- `benchmark-configuration`: Hệ thống preset và gom cụm cấu hình thống nhất cho 6 bộ dataset benchmark `B0 Inc-n`, script tự động tải từ `dataset.txt` và giải nén chuẩn hóa thư mục theo từng phiên chạy (trừ `cifar224`), tự động ánh xạ template siêu tham số mặc định cho 12 phương pháp mục tiêu và hỗ trợ backbone adapter cho `Finetune`.

### Modified Capabilities
- `experiment-tracking`: Phân biệt định danh run name và method tag trên W&B cho biến thể `finetune_adapter` khi chạy `model_name="finetune"` kết hợp với backbone adapter.

## Impact

- **Mã nguồn & cấu hình bị ảnh hưởng**:
  - `download_data.py` *(mới)*: CLI đọc `dataset.txt` để tải 5 bộ dataset (`cub`, `imagenet-r`, `imagenet-a`, `omnibenchmark`, `vtab`).
  - `extract_data.py` *(mới)*: CLI giải nén và chuẩn hóa cấu trúc `train/` + `test/` vào `./data/`.
  - `main.py`: Thêm `--method`, `--benchmark`, `--dataset`, `--init_cls`, `--increment`, `--backbone_type` và logic resolve template/cụm chạy.
  - `utils/inc_net.py`: Mở rộng điều kiện cho `model_name == "finetune"` trong nhánh `_adapter`, thêm alias `pretrained_vit_b16_224_adapter_in21k` và `ffn_num` mặc định `64`.
  - `models/finetune.py`: Lọc tham số `requires_grad` khi khởi tạo optimizer để tránh áp dụng weight decay lên trọng số backbone đã đóng băng khi dùng adapter.
  - `utils/toolkit.py`: Cập nhật logic đặt tên run và tag trong `init_wandb` cho biến thể `finetune_adapter`.
  - `exps/benchmarks.json` *(mới)*: Định nghĩa 6 preset dataset, 4 cụm thực thi và bảng ánh xạ 12 phương pháp.
  - `exps/finetune_adapter.json`, `exps/finetune_adapter_inr.json` *(mới)*: Cấu hình mặc định cho `Finetune Adapter`.
  - `exps/aper_vpt_shallow_inr.json`: Sửa `backbone_type` và `vpt_type` cho đúng `APER-VPT-Shallow`.
- **File đầu vào hiện có được tái sử dụng**: [`dataset.txt`](file:///d:/tai_lieu_hoc%20(1)/base_ML/LAMDA-PILOT/dataset.txt).
- **Thư viện phụ thuộc tùy chọn**: `gdown` (đã cài sẵn trên Google Colab) dùng cho `download_data.py`.
