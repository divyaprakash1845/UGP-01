## Setup & Installation

```bash
pip install -r requirements.txt
# Ensure the 'raw_data' folder is present in the root directory

```

## 1. Prepare Data

```text
Place your dataset folders (sub-01, sub-02, etc.) inside a `raw_data/` folder.

```

## 2. Run Preprocessing

```bash
python dataset.py

```

* **What it does**: Scans `raw_data`, resamples signals to **500 Hz**, and slices data into **1.5s** snippets (750 samples).
* **Channels**: Extracts 9 specific channels (`Fz`, `FCz`, `Pz`, `Oz`, `C3`, `C4`, `P3`, `P4`, `ECG1`).

## 3. Training

```bash
python train.py

```

* **Architecture**: Uses a Transformer Encoder with 2 layers and 4 attention heads.
* **Class Weighting**: Automatically handles data imbalance by applying inverse frequency weights to the loss function.

---
