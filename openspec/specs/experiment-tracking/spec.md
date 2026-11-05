# Experiment Tracking Specification

## Purpose

Provides configurable experiment tracking with Weights & Biases (W&B) to record hyperparameters, repository metadata, and incremental learning evaluation metrics across tasks on a shared public project.

## Requirements

### Requirement: Opt-in Experiment Tracking Configuration
The system SHALL allow users to enable or disable experiment tracking via JSON configuration files or command-line arguments without breaking existing local logging when disabled.

#### Scenario: Tracking disabled by default
- **WHEN** the user runs training without `use_wandb` set or with `use_wandb` set to `false`
- **THEN** the system executes training using only standard console and local file logging without importing or requiring the `wandb` library

#### Scenario: Tracking enabled via config or CLI
- **WHEN** the user runs training with `use_wandb` set to `true` via JSON configuration or `--use_wandb` CLI argument
- **THEN** the system initializes an experiment tracking run for each random seed using the configured project, entity, group, and tags

### Requirement: Multi-Repository Metadata Tagging
The system SHALL attach repository identification metadata (`repo_name`, `group`, `tags`, and `git_commit`) to every tracking run so experiments from LAMDA-PILOT can coexist with other repositories in a single public W&B project.

#### Scenario: Default group and tags assignment
- **WHEN** a tracking run is initialized without an explicit `wandb_group` override
- **THEN** the system sets the run group to `"LAMDA-PILOT"`, records `repo_name: "LAMDA-PILOT"` in config, and includes the model name, dataset name, and backbone type in the run tags

#### Scenario: Custom project and entity configuration
- **WHEN** `wandb_project` and `wandb_entity` are provided in configuration or CLI arguments
- **THEN** the system routes the run to the specified Weights & Biases entity and public project

### Requirement: Standardized Incremental Task Metric Logging
The system SHALL log evaluation metrics after each incremental task using a unified metric schema indexed by the task step.

#### Scenario: Logging CNN metrics after a task
- **WHEN** an incremental task completes evaluation with CNN predictions
- **THEN** the system logs `task`, `eval/top1_acc`, `eval/top5_acc`, `eval/avg_acc`, `params/total`, and `params/trainable` at the corresponding task step

#### Scenario: Logging NME metrics when available
- **WHEN** an incremental task completes evaluation and returns non-null NME (Nearest Mean of Exemplars) predictions
- **THEN** the system additionally logs `eval/nme_top1_acc`, `eval/nme_top5_acc`, and `eval/nme_avg_acc` at the corresponding task step

### Requirement: Final Forgetting and Run Lifecycle Management
The system SHALL record final forgetting metrics at the end of all incremental tasks and cleanly close the tracking session before starting the next seed run.

#### Scenario: Recording forgetting and finishing run
- **WHEN** all incremental tasks for a given seed complete
- **THEN** the system logs the final forgetting metric (if enabled) and invokes run completion so subsequent seeds in `seed_list` start as independent runs

