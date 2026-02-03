# Change log (vs upstream TOAD)

This document summarizes the main changes introduced in this repository compared to upstream TOAD https://github.com/mahmoodlab/TOAD.

## Functional changes

### 1) Task: primary-site classification only
- Removed/disabled the upstream multi-objective setup that includes primary vs metastatic aspects.
- Updated the architecture and loss to match single-task primary-site classification.

### 2) Learning-curve support via label fractions (x% data)
- Added CLI/script support to train/evaluate at a user-defined label fraction (e.g., 10–100%).
- Added sequential experiment runner to loop over label fractions and store outputs per fraction.

### 3) Patient-level stratification enabled by default
- Set `patient_strat=True` by default for split creation/training to stratify by patient.

### 4) Released pretrained classifier weights (100% data, UNI features)
- Exported and documented pretrained weights trained on 100% of the training set with UNI features.

### 5) Released extracted features used in the study
- Documented and packaged extracted `.pt` features for ResNet-50 and UNI encoders.

### 6) Container-based reproducibility
- Provided a Singularity/Apptainer image (or recipe + checksum) for stable, repeatable execution.

### 7) Logging and traceability
- Added per-run log naming, run metadata files, and consistent results folder layout.

