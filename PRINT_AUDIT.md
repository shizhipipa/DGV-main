# Print Audit

This file lists the public-facing terminal messages that can appear in the open-source release.

## `run_dgv.py`

- `Using device: ...`
- `Loading processed samples from: ...`
- `DataLoaders created (workers=...): train=..., val=..., test=...`
- `Layer-wise learning rates: CodeLM=..., Q-Former/GNN=...`
- `Starting DGV training`
- `Epoch ... summary: ...`
- `New best validation F1: ...`
- `Early stopping triggered after ...`
- `Early stopping triggered because F1 stayed at 0 ...`
- `Dual-view fusion statistics:`
- `Training finished. Best validation F1: ...`
- `DGV test results`

## `training/training_val_test.py`

- `Created ... loss ...`
- `=== Train Epoch: ... Loss ...`
- `Training prediction distribution: ...`
- `=== Validation Epoch: ...`
- `Confusion Matrix: ...`
- `=== Test Results - ...`
- `Test Confusion Matrix:`
- `Checkpoint saved to ...`
- `Checkpoint loaded ...`

## `models/DualGraphVulD.py`

- `Loading CodeLM from: ...`
- `Using dual-view fusion: ...`
- `Using Q-Former fusion path.`
- `Bias adjustment triggered (...): ...`
- `Saving model to: ...`
- `Model saved successfully.`
- `Loaded model from: ...`

## `utils/data/datamanager.py`

- Dataset split summary

## Review advice

If you do not want a message to appear during public training, search for the exact English phrase above inside `DGV-main` and remove or downgrade it before publishing.
