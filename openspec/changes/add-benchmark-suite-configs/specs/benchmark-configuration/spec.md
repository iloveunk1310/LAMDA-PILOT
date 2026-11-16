# Spec Delta

## Purpose

Provides a unified benchmark preset, clustering, and simple `dataset.txt`-driven download/extraction workflow for the standard continual learning datasets and 12 target methods, including native support for Finetune Adapter.

## ADDED Requirements

### Requirement: Six-Dataset Benchmark Presets and Clustering
The system SHALL provide centralized benchmark presets for the 6 target datasets (`cifar_b0inc5`, `cub_b0inc10`, `inr_b0inc5`, `ina_b0inc10`, `omnibench_b0inc30`, `vtab_b0inc10`) and logical execution clusters (`fast`, `inc10_200c`, `long_or_large`, `all_6`), excluding `ObjectNet` from default clusters.

#### Scenario: Single preset selection for B0 Inc-n
- **WHEN** the user selects a single benchmark preset such as `cifar_b0inc5`
- **THEN** the system resolves `dataset` to `"cifar224"` and sets equal `init_cls=5` and `increment=5` so classes are equally divided across tasks

#### Scenario: Sequential execution of a dataset cluster
- **WHEN** the user specifies a cluster name such as `inc10_200c` or `all_6`
- **THEN** the system runs each benchmark preset in that cluster sequentially without including `objectnet`

### Requirement: Automatic Hyperparameter Template Selection
The system SHALL automatically select default method hyperparameters matching the selected benchmark dataset category without requiring per-dataset duplicate JSON files.

#### Scenario: Selecting CIFAR default template
- **WHEN** the benchmark preset targets `cifar224`
- **THEN** the system loads the CIFAR default configuration template (`<model>.json`) for the chosen method before applying benchmark dataset splits

#### Scenario: Selecting natural 224x224 default template
- **WHEN** the benchmark preset targets `cub`, `imagenetr`, `imageneta`, `omnibenchmark`, or `vtab`
- **THEN** the system loads the natural-image default configuration template (`<model>_inr.json`) for the chosen method when available

### Requirement: Finetune Adapter Backbone Support
The system SHALL support running the `finetune` learner with adapter-augmented ViT backbones (including `pretrained_vit_b16_224_adapter_in21k` and `pretrained_vit_b16_224_in21k_adapter`) using default adapter bottleneck dimensions when omitted.

#### Scenario: Running Finetune Adapter with IN21K adapter backbone
- **WHEN** the user configures `model_name` as `"finetune"` and `backbone_type` as `"pretrained_vit_b16_224_adapter_in21k"` or `"pretrained_vit_b16_224_in21k_adapter"`
- **THEN** the system initializes the adapter ViT backbone with non-adapter weights frozen, defaults `ffn_num` to `64` if unset, and optimizes only trainable parameters

#### Scenario: Consistent APER-VPT-Shallow configuration on natural images
- **WHEN** the user selects the `aper_vpt_shallow_inr` configuration
- **THEN** the system uses a VPT-enabled backbone and sets `vpt_type` to `"shallow"`

### Requirement: Direct CLI Dataset and Backbone Overrides
The system SHALL allow users to override dataset name, initial class count, increment step size, and backbone type directly via command-line flags.

#### Scenario: Overriding dataset and backbone via CLI
- **WHEN** the user supplies `--dataset`, `--init_cls`, `--increment`, or `--backbone_type` on the command line
- **THEN** the provided CLI values take precedence over both the base JSON template and benchmark preset settings

### Requirement: Manifest-Driven Dataset Download from dataset.txt
The system SHALL provide a download script (`download_data.py`) that parses `<url> <filename>` entries from `dataset.txt` (excluding `cifar224`) and downloads Google Drive and SharePoint/OneDrive archives into an archive directory.

#### Scenario: Downloading all or selected datasets from dataset.txt
- **WHEN** the user runs `download_data.py` with no filter (all entries in `dataset.txt`) or with `--dataset cub omnibenchmark`
- **THEN** the script downloads each matching archive into `--archive_dir` using `gdown` for Google Drive links and direct HTTP download for SharePoint/OneDrive links, skipping existing files unless `--force` is set

#### Scenario: Skipping CIFAR-100 in download_data.py
- **WHEN** the user passes `cifar224` or `cifar100` to `download_data.py`
- **THEN** the script informs the user that CIFAR-100 is automatically downloaded by torchvision during training and does not require a `dataset.txt` entry

### Requirement: Automated Archive Extraction and Directory Normalization
The system SHALL provide an extraction script (`extract_data.py`) that extracts archives listed in `dataset.txt` and normalizes directory layouts into the exact `train/` and `test/` paths expected by `utils/data.py`.

#### Scenario: Extracting and normalizing nested archive layouts
- **WHEN** the user runs `extract_data.py` for archives from `dataset.txt` (`cub.zip`, `imagenet-r.zip`, `imagenet-a.zip`, `omnibenchmark.zip`, `vtab.zip`)
- **THEN** the script extracts each archive and places `train/` and `test/` at the expected destination (e.g., `./data/vtab-cil/vtab/train/` for `vtab` and `./data/cub/train/` for `cub`), skipping already-extracted datasets unless `--force` is set

#### Scenario: Cleaning up archives after extraction
- **WHEN** the user passes `--cleanup` to `extract_data.py`
- **THEN** the script removes each archive file only after verifying that both `train/` and `test/` target directories exist
