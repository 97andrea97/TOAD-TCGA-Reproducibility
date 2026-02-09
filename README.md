# TOAD-TCGA-Reproducibility

A fully reproducible evaluation of **Tumor Origin Assessment via Deep Learning (TOAD)** on **TCGA whole-slide images (WSI)**, with learning curves that quantify how performance scales as training data is reduced.

This repository provides:
- **Containerized preprocessing** (WSI patching + feature extraction)
- **Containerized training/evaluation** with a **learning-curve runner** (x% training data)
- A **local inference GUI** for running the TCGA-trained model on a slide and exporting JSON summaries

## Why this exists

Deep learning models for cancer histopathology are increasingly used to classify the primary site of origin, yet many approaches are difficult to reproduce because trained models and/or training data are not fully accessible. Here, we provide a **TCGA-only** reproduction of TOAD and a systematic evaluation across data regimes (learning curves), including an encoder comparison between **ResNet-50** and **UNI**.

---

## Model and feature availability

### Trained model weights
- ✅ **Publicly released weights (via Zenodo):** **ResNet-based TOAD trained model weights** (AGPL-3.0).
- ❌ **Not released:** **UNI-based TOAD trained model weights**.

**Why UNI-based trained weights are not released:** the **UNI** encoder is distributed under terms that restrict redistribution. To respect those terms, we do **not** upload or bundle UNI model files, nor the trained TOAD weights that depend on UNI features.

> You can still run UNI-based experiments: if you have approved UNI access, you can download UNI through the official channel and reproduce the UNI-based variant using this repository’s scripts.

### Extracted features
- ❌ **We do not distribute extracted feature files** (neither ResNet nor UNI features).  
  Feature files can be very large and are derived from WSI data; instead, this repository provides **reproducible preprocessing** so you can generate features locally from TCGA slides.

---

## Data: downloading TCGA slides (GDC)

Slides must be obtained from the **GDC portal** / **GDC Data Transfer Tool** using the **GDC file IDs** listed in:

- `src_preprocessing/CLAM_encoder/dataset_csv/TCGA.csv`

### Step 1 — Inspect which column contains the GDC file IDs
From the repo root:
```bash
python3 - <<'PY'
import pandas as pd
p = "src_preprocessing/CLAM_encoder/dataset_csv/TCGA.csv"
df = pd.read_csv(p)
print("Columns:", list(df.columns))
print(df.head(2))
PY
```

### Step 2 — Export file IDs to a text file
Replace `FILE_ID_COL` below with the correct column name you saw in Step 1 (commonly something like `file_id`):
```bash
FILE_ID_COL="file_id"  # <-- edit this

python3 - <<PY
import pandas as pd, os
p = "src_preprocessing/CLAM_encoder/dataset_csv/TCGA.csv"
col = os.environ["FILE_ID_COL"]
df = pd.read_csv(p)
assert col in df.columns, f"Column '{col}' not found. Available: {list(df.columns)}"
df[col].dropna().astype(str).to_csv("gdc_file_ids.txt", index=False, header=False)
print("Wrote gdc_file_ids.txt with", df[col].notna().sum(), "IDs")
PY
```

### Step 3 — Download slides with the GDC Data Transfer Tool (`gdc-client`)
Using `xargs` avoids very long command lines:
```bash
# Create an output folder for downloaded slides
mkdir -p tcga_svs

# Download each file ID
cat gdc_file_ids.txt | xargs -n 1 -P 4 gdc-client download --no-related-files --dir tcga_svs
```

After download, point preprocessing to the folder containing the `.svs` files (or a folder tree that contains them).

---

## Artifacts (Zenodo)

Large artifacts are **not** stored in git. You will download them from Zenodo and place them at the expected paths.

**Containers**
- `assets/containers/singularity_preprocessing.simg`
- `assets/containers/singularity_train_eval.simg`

**Public trained model weights (ResNet-based)**
- `assets/inference_model_checkpoint/<your_model_file>.pt`

**Zenodo DOI:** _TODO (add after publishing)_

Recommended: verify integrity using `SHA256SUMS` from Zenodo:
```bash
sha256sum -c SHA256SUMS
```

---

## Quickstart

### 0) Choose a container runtime (Singularity vs Apptainer)

Some systems provide `singularity`, others provide `apptainer`. These tools are compatible for most usage, but the command name differs.

To make the scripts portable, they support a `RUNTIME` variable:
- If `RUNTIME` is not set, the script auto-detects `apptainer` first, then `singularity`.
- You can override it explicitly:
  ```bash
  RUNTIME=apptainer bash run_preprocessing.sh ...
  # or
  RUNTIME=singularity bash run_preprocessing.sh ...
  ```

### 1) Preprocessing (WSI → FEATURES)

This script searches for `.svs` under `--input-dir` up to `--maxdepth` and creates:
- `<out-root>/logs_and_metadata/` (per-slide logs + extracted metadata)
- `<out-root>/FEATURES/` (feature files)

**Default encoder:** `uni_v1` (best-performing in our study).  
If you do not have UNI access, use `--encoder resnet50_trunc`.

```bash
bash run_preprocessing.sh   --input-dir /abs/path/to/svs_folder_or_tree   --out-root  /abs/path/to/preprocessing_output   --gpu 0
```

Example using ResNet features:
```bash
bash run_preprocessing.sh   --input-dir /abs/path/to/svs_folder_or_tree   --out-root  /abs/path/to/preprocessing_output   --encoder resnet50_trunc   --gpu 0
```

Useful options:
- `--jobs` parallel slides on the same GPU
- `--batch-size` feature extraction batch size (default 800)
- `--target-patch-size` patch size (default 224)
- `--maxdepth` search depth (default 2)
- `--image` preprocessing container path (default `assets/containers/singularity_preprocessing.simg`)
- `--code` preprocessing code path (default `src_preprocessing/CLAM_encoder`)

### 2) Training & evaluation (FEATURES → learning curves)

Runs training and evaluation for multiple training-data fractions (configured in `run_train_eval.sh`) and writes results under:
- `train_eval_output/RESULTS_EXP_*/`

```bash
bash run_train_eval.sh   --features /abs/path/to/preprocessing_output/FEATURES   --gpu 0
```

> **Important:** the encoder used for feature extraction must match what you train/evaluate on.
> - UNI features → train/eval a UNI-feature model (not distributed)
> - ResNet features → train/eval a ResNet-feature model (public weights distributed)

### 3) Inference GUI (local)

Launch the GUI (default port 8765):
```bash
bash run_inference_GUI.sh
```

Change port:
```bash
PORT=9000 bash run_inference_GUI.sh
```

The GUI writes outputs under:
- `output/inference_GUI_runs/<timestamp>/...`

To locate where the GUI expects the trained model weights path:
```bash
rg -n "weights|model|\.pt|checkpoint" src_inference_GUI/*.py
```

---

## Repository structure (high level)

- `run_preprocessing.sh` — containerized preprocessing
- `run_train_eval.sh` — containerized learning-curve training/eval
- `run_inference_GUI.sh` — run local GUI
- `src_preprocessing/CLAM_encoder/` — preprocessing subset (derived from CLAM)
- `src_train_eval/` — train/eval code (derived from TOAD + modifications)
- `src_inference_GUI/` — local GUI + inference logic
- `assets/containers/` — container build recipes (Dockerfiles)
- `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `licenses/` — licensing + attribution
- `CITATION.cff`, `DATA_AVAILABILITY.md`, `CHANGES.md` — paper-support files

---

## Licensing & attribution (summary)

- **Repository license:** GNU **AGPL-3.0** (see `LICENSE`).
- Derived components (summary):
  - **TOAD (AGPL-3.0):** basis for training/evaluation; modified for primary-site only, learning curves, patient-level stratification defaults, extra logging, and containerized execution support.
  - **CLAM (GPL-3.0):** subset used for WSI patching and feature extraction (classifier training scripts excluded).
- See `THIRD_PARTY_NOTICES.md` and `licenses/` for full details.

### Model weights license
- Public **ResNet-based** trained TOAD weights are released under **AGPL-3.0**.
- UNI-based trained weights are not distributed (see “Model and feature availability”).

---

## Citation

See `CITATION.cff`. Please also cite the associated paper and the Zenodo record (DOI) once available.
