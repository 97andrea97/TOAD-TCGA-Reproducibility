# TOAD (modified) reproducibility package — TCGA primary-site classification

This repository accompanies our PLOS Digital Health submission:

**Fully reproducible TOAD learning curves on TCGA for primary site classification (ResNet-50 vs UNI)**

## Abstract
> Deep learning models for cancer histopathology are increasingly used to classify the primary site of origin, yet many existing approaches remain difficult to reproduce because the models themselves and their underlying training data are not fully accessible. As a result, it is often unclear how much data these methods require and how their accuracy degrades in limited- training data settings—a common scenario for many research groups. In this study, we provide a fully reproducible evaluation of the Tumor Origin Assessment via Deep Learning (TOAD) framework using only whole-slide images from the public The Cancer Genome Atlas (TCGA) repository. By systematically varying the amount of training data, we construct detailed learning curves that quantify how performance scales across data regimes. Our results show that a TCGA-only TOAD reproduction recovers much of the original model’s global performance and that per-class data requirements vary substantially across tissue types. We also compare two image encoders—a ResNet-50 pretrained on natural images and UNI, a pathology foundation model trained on large-scale histopathology data—and find that UNI consistently improves data efficiency, achieving equal or higher performance with substantially fewer training slides, especially in low- and medium-training data regimes. All code is released in containerized form to ensure strict reproducibility, and we provide a ready-to-use TCGA-trained TOAD model for immediate application. Overall, this work delivers the first open, data-driven benchmark of the TOAD framework on TCGA whole-slide images, clarifying its dependence on dataset size, identifying encoder-driven differences in data efficiency, and offering practical guidance for developing reproducible models for cancer site classification.

## What you get in this repo
- A **containerized** pipeline to train and evaluate a **TOAD-derived** model on **TCGA** for **primary site classification**.
- A reproducibility runner that builds **learning curves** by training/evaluating at multiple label fractions (10% → 100%).
- Extra logging

---

# Key differences vs the original TOAD repository (upstream)

This codebase is a **slightly modified version** of the original TOAD repo (see `UPSTREAM.md`). The main differences are:

1. **Task definition changed (primary-site only).**  
   The upstream TOAD code targets a broader setup (e.g., includes primary vs metastatic aspects).  
   **Here we only classify the *primary site***, therefore:
   - the **model head / architecture** and
   - the **loss/objective**
   are adjusted accordingly for single-task primary-site classification.

2. **Built-in training on _x%_ of the dataset (learning curves).**  
   We added scripting and argument plumbing to easily train/evaluate at a chosen label fraction (e.g., 10%, 20%, …, 100%), producing learning curves.

3. **Patient stratification enabled by default.**  
   We set `patient_strat=True` by default to ensure patient-level stratification during split creation and training.

4. **We release a ready-to-use TCGA-trained classifier (100% data) using UNI features.**  
   We provide the **trained classifier weights** obtained with **100% of TCGA training data** using **CLAM-extracted UNI features**, so users can immediately apply the model.

5. **We release extracted features used in the paper.**  
   We provide the extracted slide-level feature tensors (`.pt` files) used for training/evaluation:
   - ResNet-50 features (ImageNet-pretrained encoder)
   - UNI features (SSL pathology encoder)
   See `MODEL_CARD.md` for artifact names and how to download/use them.

6. **Containerized execution for strict reproducibility.**  
   We provide a Singularity/Apptainer image (or recipe + checksum) to reproduce the exact software environment used in the paper (see `CONTAINER.md`).

7. **More logging for traceability.**  
   Each run writes:
   - full stdout/stderr logs
   - `run_meta.txt` with timestamps, label fraction, GPU id, container info, etc.

A more detailed, file-by-file change summary is in `CHANGES.md`.

---

# Repository structure

```
.
├── run_exps.sh                  # main entry point: loops over label fractions
├── src/                         # modified TOAD code used for experiments
│   ├── run_python_scripts.sh
│   ├── create_splits.py
│   ├── main_mtl_concat.py
│   ├── eval_mtl_concat.py
│   ├── datasets/
│   ├── utils/
│   └── dataset_csv/
├── containers/                  # optional: container recipe or pointer to image
├── RESULTS/                     # outputs created by the runner (gitignored)
└── docs/                        # optional extra notes
```

---

# Requirements

# Data & feature preparation (use the original CLAM repo)

This repo consumes **precomputed patch features** by CLAM (https://github.com/mahmoodlab/CLAM).

TOAD training expects:
```
FEATURES/
  └── pt_files/
      ├── <slide_id>.pt
      └── ...
```

## Step 1 — obtain TCGA WSIs
Follow TCGA/GDC access rules and your institutional requirements. We do not redistribute TCGA WSIs.

## Step 2 — extract features using the original CLAM pipeline
Use the **original CLAM** repository (https://github.com/mahmoodlab/CLAM) to extract patch features, producing `FEATURES/pt_files/*.pt`.

High-level steps (see CLAM docs for exact commands and options):
1. Create patches / coordinates (e.g., `create_patches_fp.py`)
2. Extract features (e.g., `extract_features_fp.py`) to produce `pt_files/`

**Encoder choice:** Run CLAM feature extraction with the ResNet-50 or UNI encoder.

Make sure the output folder passed to this repo is the folder that directly contains `pt_files/`.

---

# Running the experiments (learning curves)

From the repository root:

```bash
bash run_exps.sh --features /path/to/FEATURES --gpu 0 --image /path/to/container.simg
```

Outputs are created under:
```
RESULTS/RESULTS_EXP_10/
RESULTS/RESULTS_EXP_20/
...
RESULTS/RESULTS_EXP_100/
```

Each run folder contains a full log and `run_meta.txt`.

---

# Reproducing the paper figures/tables

- Learning curves: produced by running all label fractions
- Per-class performance tables: computed from the evaluation logs/artifacts
- Pretrained weights and feature artifacts: see `MODEL_CARD.md`

---

# License

This repository is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)** because it is a modified version of upstream TOAD, which is AGPL-3.0 licensed. See `LICENSE`.

---

# Contact

Open a GitHub issue for questions/bugs, or contact the corresponding author listed in the manuscript.
