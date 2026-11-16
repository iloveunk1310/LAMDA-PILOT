# Spec Delta

## ADDED Requirements

### Requirement: Distinct Identification for Finetune Adapter Runs
The system SHALL label W&B run names and method tags as `finetune_adapter` when `model_name` is `"finetune"` and `backbone_type` contains `"adapter"`, distinguishing them from full-backbone `finetune` runs on the shared dashboard.

#### Scenario: Running Finetune Adapter with W&B enabled
- **WHEN** a W&B run is initialized with `model_name="finetune"` and `backbone_type="pretrained_vit_b16_224_adapter_in21k"` (or any `"_adapter"` backbone)
- **THEN** the run name uses `finetune_adapter` instead of `finetune` and the run tags include `method:finetune_adapter`
