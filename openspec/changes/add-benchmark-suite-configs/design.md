# Design

## Context

Xem `proposal.md` để biết động lực thay đổi và `specs/benchmark-configuration/spec.md`, `specs/experiment-tracking/spec.md` cho các yêu cầu hành vi.

Trong codebase hiện tại của LAMDA-PILOT:
- Quy ước `"B/Base-m, Inc-n"` với $m=0$ trong bài báo được ánh xạ vào `DataManager` bằng cách đặt `init_cls = n` và `increment = n` (khi `args["init_cls"] == args["increment"]`, `trainer.py` tự chuyển `init_cls` thành `0` trong tên log và W&B run name là `b0inc{n}`). Không được đặt `init_cls = 0` trong JSON vì `DataManager` sẽ tạo task đầu tiên có `0` lớp.
- Trong 7 bộ dataset nêu ở giai đoạn explore, **ObjectNet** là bộ nặng nhất về dung lượng tải và lưu trữ, nên được loại bỏ khỏi bộ benchmark chạy. 5 bộ dữ liệu ngoài CIFAR đã được người dùng tổng hợp gọn trong [`dataset.txt`](file:///d:/tai_lieu_hoc%20(1)/base_ML/LAMDA-PILOT/dataset.txt) theo đúng định dạng `<url> <filename>`:

| Cụm (`Cluster`) | Preset ID | Dataset (`utils/data.py`) | Lớp | `init_cls` / `inc` | Tasks | File nguồn trong `dataset.txt` | Đường dẫn đích (`utils/data.py`) |
| :--- | :--- | :--- | :-: | :-: | :-: | :--- | :--- |
| `fast` | `vtab_b0inc10` | `vtab` | 50 | `10` / `10` | 5 | `vtab.zip` (Google Drive) | `./data/vtab-cil/vtab/{train,test}/` |
| `fast` | `cifar_b0inc5` | `cifar224` | 100 | `5` / `5` | 20 | *(Không trong `dataset.txt` — torchvision tự tải)* | `./data/cifar-100-python/` |
| `inc10_200c` | `cub_b0inc10` | `cub` | 200 | `10` / `10` | 20 | `cub.zip` (Google Drive) | `./data/cub/{train,test}/` |
| `inc10_200c` | `ina_b0inc10` | `imageneta` | 200 | `10` / `10` | 20 | `imagenet-a.zip` (Google Drive) | `./data/imagenet-a/{train,test}/` |
| `long_or_large` | `omnibench_b0inc30` | `omnibenchmark` | 300 | `30` / `30` | 10 | `omnibenchmark.zip` (SharePoint/OneDrive) | `./data/omnibenchmark/{train,test}/` |
| `long_or_large` | `inr_b0inc5` | `imagenetr` | 200 | `5` / `5` | 40 | `imagenet-r.zip` (Google Drive) | `./data/imagenet-r/{train,test}/` |
| `all_6` | *(Cả 6 preset trên)* | — | — | — | — | — | — |

### Bảng ánh xạ 12 Phương pháp Mục tiêu & Siêu tham số Mặc định

1. **Nhóm 1 — Baselines & Prompt / PEFT:**
   - `finetune`: `exps/finetune.json` / `exps/finetune_inr.json`
   - `finetune_adapter`: `exps/finetune_adapter.json` / `exps/finetune_adapter_inr.json` (dùng `model_name="finetune"` kết hợp `backbone_type="pretrained_vit_b16_224_adapter_in21k"`, `ffn_num=64`, giữ nguyên các siêu tham số mặc định của `finetune`)
   - `l2p`: `exps/l2p.json` / `exps/l2p_inr.json`
   - `dualprompt`: `exps/dualprompt.json` / `exps/dualprompt_inr.json`
   - `coda_prompt`: `exps/coda_prompt.json` / `exps/coda_prompt_inr.json`
   - `lae`: `exps/lae.json` / `exps/lae_inr.json`
2. **Nhóm 2 — Nhóm bài báo RevisitingCIL (`SimpleCIL` & `APER`):**
   - `simplecil`: `exps/simplecil.json` / `exps/simplecil_inr.json`
   - `aper_finetune`: `exps/aper_finetune.json` / `exps/aper_finetune_inr.json`
   - `aper_ssf`: `exps/aper_ssf.json` / `exps/aper_ssf_inr.json`
   - `aper_vpt_shallow`: `exps/aper_vpt_shallow.json` / `exps/aper_vpt_shallow_inr.json`
   - `aper_vpt_deep`: `exps/aper_vpt_deep.json` / `exps/aper_vpt_deep_inr.json`
   - `aper_adapter`: `exps/aper_aperpter.json` / `exps/aper_aperpter_inr.json` *(giữ nguyên tên file gốc có chữ `aperpter` của repo để không phá vỡ tương thích)*

## Goals / Non-Goals

**Goals:**
- Chạy được bất kỳ cặp `(12 phương pháp × 6 bộ dataset)` nào bằng 1 lệnh duy nhất qua `--benchmark` mà vẫn dùng trọn vẹn siêu tham số mặc định của tác giả.
- Tái sử dụng trực tiếp [`dataset.txt`](file:///d:/tai_lieu_hoc%20(1)/base_ML/LAMDA-PILOT/dataset.txt) làm nguồn duy nhất (Single Source of Truth) cho link tải 5 bộ dataset ngoài `cifar224`, để người dùng chỉ cần sửa link ngay trong `dataset.txt` nếu muốn đổi sang Google Drive cá nhân.
- Cung cấp quy trình 3 bước tối giản trên Google Colab:
  1. `!python download_data.py` (tải file `.zip` từ `dataset.txt`, mặc định cả 5 bộ hoặc lọc qua `--dataset` / `--benchmark`)
  2. `!python extract_data.py --cleanup` (giải nén & chuẩn hóa `train/` + `test/` vào `./data/`)
  3. `!python main.py --method ... --benchmark ... --use_wandb`
- Cho phép `model_name="finetune"` hoạt động trơn tru với `backbone_type="pretrained_vit_b16_224_adapter_in21k"` (và `"pretrained_vit_b16_224_in21k_adapter"`), phân tách rõ kết quả `finetune` và `finetune_adapter` trên W&B.

**Non-Goals:**
- Không thay đổi các siêu tham số mặc định (`init_lr`, `tuned_epoch`, `epochs`, `weight_decay`, `batch_size`, ...) của các file config gốc ngoài việc sửa bug ở `aper_vpt_shallow_inr.json`.
- Không đưa `cifar224` vào `download_data.py`/`extract_data.py` vì `utils/data.py` (`iCIFAR224`) đã tự tải CIFAR-100 qua `torchvision`.
- Không tự động tải dữ liệu bên trong `main.py` nhằm giữ quy trình 3 bước tường minh và dễ debug trên Colab.

## Decisions

### 1. Sửa `utils/inc_net.py` (`get_backbone`) để hỗ trợ Finetune Adapter
- **Hiện trạng & Vấn đề**: Hiện tại ở [`utils/inc_net.py#L66-L96`](file:///d:/tai_lieu_hoc%20(1)/base_ML/LAMDA-PILOT/utils/inc_net.py#L66-L96), nhánh `elif '_adapter' in name:` chỉ cho phép `args["model_name"] in ["aper_adapter", "ranpac", "fecam"]` — nếu đặt `model_name="finetune"` sẽ ném `NotImplementedError`. Ngoài ra dòng 67 đọc trực tiếp `args["ffn_num"]` (dễ `KeyError`) và chỉ nhận `"pretrained_vit_b16_224_in21k_adapter"`.
- **Quyết định**:
  - Cho phép thêm `"finetune"` trong nhánh `_adapter`.
  - Dùng `ffn_num = args.get("ffn_num", 64)`.
  - Hỗ trợ cả hai alias `"pretrained_vit_b16_224_in21k_adapter"` và `"pretrained_vit_b16_224_adapter_in21k"`.
  - Tạo `exps/finetune_adapter.json` và `exps/finetune_adapter_inr.json` với `"model_name": "finetune"`, `"backbone_type": "pretrained_vit_b16_224_adapter_in21k"`, `"ffn_num": 64`.

### 2. Lọc tham số `requires_grad` trong `models/finetune.py`
- **Quyết định**: Đổi `self._network.parameters()` thành `filter(lambda p: p.requires_grad, self._network.parameters())` trong `_train` của `models/finetune.py` để chỉ tối ưu tham số Adapter + classification head khi chạy `finetune_adapter`, giữ nguyên hành vi khi chạy `finetune` thường.

### 3. Kiến trúc Config 2 tầng: `exps/benchmarks.json` + `--benchmark` trong `main.py`
- **Quyết định**: `exps/benchmarks.json` chỉ lưu 6 preset (`cifar_b0inc5`, `cub_b0inc10`, `inr_b0inc5`, `ina_b0inc10`, `omnibench_b0inc30`, `vtab_b0inc10`), 4 cụm (`fast`, `inc10_200c`, `long_or_large`, `all_6`) và bảng ánh xạ 12 phương pháp tới cặp template `(cifar, inr)` — không lặp lại link tải vốn đã nằm trong `dataset.txt`.
- **Cơ chế phân giải trong `main.py`**:
  1. Khi có `--benchmark` (và/hoặc `--method`), tra `exps/benchmarks.json` để lấy danh sách preset cần chạy.
  2. Chọn template `<model>.json` (`cifar224`) hoặc `<model>_inr.json` (5 bộ ảnh tự nhiên 224x224).
  3. Áp `dataset`, `init_cls`, `increment` từ preset vào `param`, rồi ghi đè tiếp bằng các cờ CLI (`--dataset`, `--init_cls`, `--increment`, `--backbone_type`, `--batch_size`, `--init_lr`, `--use_wandb`, ...).

### 4. Sửa lỗi cấu hình gốc `exps/aper_vpt_shallow_inr.json`
- **Quyết định**: Sửa `"backbone_type"` thành `"pretrained_vit_b16_224_vpt"` và `"vpt_type"` thành `"shallow"` để tránh lỗi `NotImplementedError('VPT requires VPT backbone')` trong `models/aper_vpt.py`.

### 5. Phân biệt `finetune_adapter` trong W&B (`utils/toolkit.py`)
- **Quyết định**: Trong `init_wandb(args)`, nếu `model_name == "finetune"` và `"adapter" in backbone_type.lower()`, gán biến hiển thị `effective_method = "finetune_adapter"` cho `run_name` và tag `method:finetune_adapter`.

### 6. Tải và Giải nén đơn giản theo `dataset.txt` (`download_data.py` -> `extract_data.py`)
- **Đọc trực tiếp `dataset.txt`**:
  - Mỗi dòng dạng `<url> <filename>` được parse thành entry `(url, filename)`.
  - Ánh xạ tên archive sang khóa dataset và đường dẫn đích trong `utils/data.py`:
    - `cub.zip` $\leftrightarrow$ `cub` $\rightarrow$ `./data/cub/{train,test}`
    - `imagenet-r.zip` $\leftrightarrow$ `imagenetr` / `imagenet-r` $\rightarrow$ `./data/imagenet-r/{train,test}`
    - `imagenet-a.zip` $\leftrightarrow$ `imageneta` / `imagenet-a` $\rightarrow$ `./data/imagenet-a/{train,test}`
    - `omnibenchmark.zip` $\leftrightarrow$ `omnibenchmark` $\rightarrow$ `./data/omnibenchmark/{train,test}`
    - `vtab.zip` $\leftrightarrow$ `vtab` $\rightarrow$ `./data/vtab-cil/vtab/{train,test}`
- **Thiết kế `download_data.py`**:
  - Hỗ trợ `--manifest` (mặc định `dataset.txt`), `--dataset` (tùy chọn lọc 1 hoặc nhiều bộ, nếu bỏ trống sẽ tải toàn bộ dòng trong `dataset.txt`), `--benchmark` (lọc theo preset/cụm trong `exps/benchmarks.json`, tự động bỏ qua `cifar224` kèm thông báo), `--archive_dir` (mặc định `./data/archives`), `--force`.
  - **Tải Google Drive** (`drive.google.com`): dùng `gdown.download(url, output_path, fuzzy=True, quiet=False)` (hỗ trợ vượt màn hình xác nhận file lớn).
  - **Tải SharePoint / OneDrive** (`sharepoint.com`, như dòng `omnibenchmark.zip` trong `dataset.txt`): tự động gắn `&download=1` (hoặc `?download=1`) vào URL và tải streaming bằng `urllib`/`requests` (có kiểm tra `Content-Type` không phải `text/html` để tránh lưu nhầm trang web báo lỗi thành file `.zip`).
  - Idempotent: Bỏ qua nếu `<archive_dir>/<filename>` đã tồn tại và kích thước > 0 (trừ khi truyền `--force`).
- **Thiết kế `extract_data.py`**:
  - Đọc danh sách archive từ `--manifest` (`dataset.txt`), hỗ trợ lọc `--dataset`, `--benchmark`, `--archive_dir`, `--data_dir` (mặc định `./data`), `--cleanup`, `--force`.
  - Hỗ trợ nhận diện định dạng nén qua magic bytes (`zipfile.is_zipfile`, `tarfile.is_tarfile`).
  - **Chuẩn hóa cây thư mục (`Directory Normalization`)**: Giải nén vào thư mục tạm bên trong `--data_dir`, tự động tìm thư mục gốc chứa đồng thời hai thư mục con `train/` và `test/`, rồi di chuyển gọn về đúng `target_dir` mà `utils/data.py` yêu cầu (đảm bảo dù `vtab.zip` giải nén ra `vtab/train` hay `vtab-cil/vtab/train`, kết quả cuối luôn nằm ở `./data/vtab-cil/vtab/train/` và `./data/vtab-cil/vtab/test/`).
  - Bỏ qua nếu `target_dir/train` và `target_dir/test` đã tồn tại (trừ khi `--force`), và chỉ xóa file `.zip` khi truyền `--cleanup` sau khi đã xác nhận đích hợp lệ.

## Risks / Trade-offs

- **[Risk] Link Google Drive public bị giới hạn lượt tải (Quota) hoặc link SharePoint đổi hành vi redirect** → **Mitigation**: Vì `download_data.py` đọc trực tiếp từ `dataset.txt`, người dùng chỉ cần thay link mới (ví dụ link Drive cá nhân hoặc mirror) vào `dataset.txt` mà không phải sửa code; đồng thời `download_data.py` kiểm tra phản hồi HTML thay vì file nén thực để báo lỗi rõ ràng thay vì tạo file `.zip` hỏng.
- **[Risk] Dung lượng ổ đĩa Colab miễn phí (~70–100GB) có thể đầy nếu giải nén cùng lúc cả 5 bộ lớn** → **Mitigation**: Truyền `--dataset` hoặc `--benchmark` để tải/giải nén theo từng cụm và dùng cờ `--cleanup` để xóa file `.zip` ngay sau khi giải nén xong.
- **[Risk] Tên `backbone_type` cho Adapter có 2 cách viết (`_adapter_in21k` vs `_in21k_adapter`)** → **Mitigation**: Hỗ trợ cả hai chuỗi tên trong `utils/inc_net.py`.
