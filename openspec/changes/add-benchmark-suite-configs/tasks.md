# Tasks

## 1. Hỗ trợ Finetune Adapter & Sửa Cấu Hình Gốc

- [x] 1.1 Cập nhật `utils/inc_net.py` (`get_backbone`) để nhánh `_adapter` chấp nhận thêm `model_name == "finetune"`, hỗ trợ cả `pretrained_vit_b16_224_adapter_in21k` và `pretrained_vit_b16_224_in21k_adapter`, đồng thời dùng `args.get("ffn_num", 64)`; kiểm tra bằng script Python khởi tạo `IncrementalNet({"model_name": "finetune", "backbone_type": "pretrained_vit_b16_224_adapter_in21k"}, True)` không ném `NotImplementedError` hay `KeyError`.
- [x] 1.2 Cập nhật `models/finetune.py` (`_train`) để truyền `filter(lambda p: p.requires_grad, self._network.parameters())` vào `optim.SGD`, đảm bảo khi chạy `finetune` với adapter chỉ các tham số adapter và classifier head được tối ưu; kiểm tra bằng kiểm thử khởi tạo `Learner` và kiểm tra số lượng tham số có `requires_grad=True` trong optimizer `param_groups`.
- [x] 1.3 Tạo `exps/finetune_adapter.json` và `exps/finetune_adapter_inr.json` kế thừa siêu tham số mặc định từ `finetune` kèm `"backbone_type": "pretrained_vit_b16_224_adapter_in21k"` và `"ffn_num": 64`, đồng thời sửa `exps/aper_vpt_shallow_inr.json` thành `"backbone_type": "pretrained_vit_b16_224_vpt"` và `"vpt_type": "shallow"`; kiểm tra parse JSON hợp lệ cho cả 3 file bằng `python -c "import json; ..."`.
- [x] 1.4 Cập nhật `utils/toolkit.py` (`init_wandb`) để đặt run name và tag theo `finetune_adapter` khi `model_name == "finetune"` và `backbone_type` chứa `"adapter"`; kiểm tra bằng unit check `init_wandb` với `wandb_mode="disabled"` cho ra `name` bắt đầu bằng `PILOT_finetune_adapter_`.

## 2. File Preset Tập Trung & Tích Hợp CLI (`exps/benchmarks.json` & `main.py`)

- [x] 2.1 Tạo `exps/benchmarks.json` định nghĩa 6 preset (`cifar_b0inc5`, `cub_b0inc10`, `inr_b0inc5`, `ina_b0inc10`, `omnibench_b0inc30`, `vtab_b0inc10`), 4 cụm (`fast`, `inc10_200c`, `long_or_large`, `all_6` — loại bỏ `objectnet`), và bảng ánh xạ 12 phương pháp mục tiêu tới cặp template `(cifar, inr)`; kiểm tra cấu trúc JSON hợp lệ và đủ 6 preset + 12 phương pháp.
- [x] 2.2 Mở rộng `main.py` thêm các đối số `--method`, `--benchmark`, `--dataset`, `--init_cls`, `--increment`, `--backbone_type`, xử lý tự động chọn template (`*.json` cho `cifar224` và `*_inr.json` cho 5 bộ ảnh 224x224) và chạy tuần tự từng preset khi truyền tên cụm; kiểm tra `python main.py --help` hiển thị đầy đủ các cờ mới và dry-run logic phân giải preset cho `12 phương pháp x 6 dataset` không lỗi.

## 3. Script Tải & Giải Nén Đơn Giản Theo `dataset.txt` (`download_data.py` & `extract_data.py`)

- [x] 3.1 Tạo `download_data.py` đọc trực tiếp `<url> <filename>` từ `dataset.txt` (5 bộ ngoài `cifar224`: `cub.zip`, `imagenet-r.zip`, `imagenet-a.zip`, `omnibenchmark.zip`, `vtab.zip`), hỗ trợ `--manifest` (mặc định `dataset.txt`), `--dataset`, `--benchmark`, `--archive_dir` (mặc định `./data/archives`), `--force`; dùng `gdown` cho link Google Drive và tải trực tiếp (`&download=1` + kiểm tra không phải HTML) cho link SharePoint/OneDrive, đồng thời bỏ qua file đã tồn tại và thông báo rõ nếu truyền `cifar224`; kiểm tra `python download_data.py --help` và unit test parse `dataset.txt` + lọc dataset + mock download thành công.
- [x] 3.2 Tạo `extract_data.py` đọc danh sách archive từ `dataset.txt`, hỗ trợ `--manifest`, `--dataset`, `--benchmark`, `--archive_dir`, `--data_dir` (mặc định `./data`), `--cleanup`, `--force`; nhận diện định dạng nén (`.zip`, `.tar`, `.tar.gz`), giải nén và chuẩn hóa thư mục `train/` + `test/` khớp chính xác `utils/data.py` (bao gồm `./data/vtab-cil/vtab/{train,test}` cho `vtab` và `./data/{cub,imagenet-r,imagenet-a,omnibenchmark}/{train,test}`), bỏ qua nếu đã giải nén và chỉ xóa archive khi `--cleanup` sau khi xác minh hợp lệ; kiểm tra bằng unit test tạo archive giả lập (có thư mục lồng) rồi xác minh cấu trúc đầu ra.

## 4. Tài Liệu Quy Trình Colab & Kiểm Thử Tích Hợp

- [x] 4.1 Cập nhật `WANDB_GUIDE.md` bổ sung quy trình 3 bước trên Google Colab (`!python download_data.py ... -> !python extract_data.py ... -> !python main.py ...` dựa trên `dataset.txt`), bảng 6 preset, 4 cụm chạy, và danh sách 12 phương pháp; kiểm tra toàn bộ cờ trong tài liệu khớp với `--help` của `download_data.py`, `extract_data.py` và `main.py`.
- [x] 4.2 Kiểm thử tích hợp bằng cách chạy `python download_data.py --dataset cifar224`, kiểm tra `extract_data.py --help`, và dry-run phân giải cấu hình `python main.py --method finetune_adapter --benchmark all_6` để xác nhận toàn bộ pipeline hoạt động nhất quán.

## Workflow follow-up (Lưu ý khi chạy phiên mới trên Google Colab)

- Nếu muốn đổi nguồn tải (ví dụ sang Google Drive cá nhân để tránh giới hạn lượt tải), chỉ cần sửa trực tiếp URL tương ứng trong file `dataset.txt` theo định dạng `<url> <filename.zip>`.
- Để tiết kiệm thời gian tải lại giữa các phiên Colab, có thể mount Google Drive và truyền `--archive_dir /content/drive/MyDrive/pilot_archives` cho `download_data.py` (chỉ tải 1 lần) và `extract_data.py` (giải nén vào `./data/` mỗi phiên mới).
