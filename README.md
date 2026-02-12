# MAESTRO: Cognitive Workload Classification

**Multi-modal Adaptive Embedding Sequence TRansformer Optimization**

This repository implements a Transformer-based architecture to classify cognitive workload using 9-channel EEG and ECG data. The pipeline processes 1.5-second windows (750 data points) extracted from physiological signals.

## ## Setup & Installation

To initialize the environment in Google Colab, run the following block to clone the repository and enter the project directory:

```bash
git clone https://github.com/divyaprakash1845/UGP-01
%cd UGP-01

```

Once inside the directory, install the necessary dependencies:

```bash
pip install -r requirements.txt

```

## ## 1. Prepare Data

```text
1. Place your subject folders (e.g., sub-01, sub-02) inside a folder named `raw_data`.
2. Ensure the `raw_data` folder is located in the root (/content/) directory.
   (Structure: /content/raw_data/ and /content/UGP-01/)

```

## ## 2. Run Preprocessing

```bash
python dataset.py

```

* **Functionality**: Automatically scans the `raw_data` folder for subjects.
* **Signal Processing**: Resamples data to 500 Hz and extracts 8 EEG channels plus 1 ECG channel.
* **Output**: Generates 1.5-second snippets (750 samples) saved as `.pt` tensors.

## ## 3. Training

```bash
python train.py
