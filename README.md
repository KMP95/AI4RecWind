# AI4RecWind (Artificial Intelligence for the Reconstruction of Wind)  
**by Climatoc-Lab**

AI4RecWind provides tools for reconstructing missing windspeed observations in AEMET’s historical dataset using a U-Net with partial convolutions (CRAI model).  
This implementation builds on the CRAI software developed by DKRZ and adapts it to reconstruct daily windspeed maps in Spain.

---

## Overview

- 🔧 **Reconstruction-first design**: This repository is intended primarily for **reconstructing missing data** using a **pre-trained model**.
- 🧠 **Model architecture**: Based on a partial convolution U-Net (PCNN) with optional LSTM, GRU, and Attention mechanisms.
- 🗺️ **Use case**: Filling gaps in daily windspeed maps from AEMET’s station network.

---

## 1. Environment Setup

You can set up the Python environment using Conda.

### 🧪 CPU Environment
```bash
conda env create -f environment.yml
conda activate crai
```

### 🚀 GPU Environment (CUDA)
```bash
conda env create -f environment-cuda.yml
conda activate crai
```

---

## 2. Installing CRAI

With the environment activated, install the `climatereconstructionAI` package:
```bash
pip install model/climatereconstructionAI
```

---

## 3. Reconstruction (Validation)

Once the model is installed and the environment is ready, you can reconstruct missing values using a pre-trained model (typically saved as `best.pth`):

```bash
bash run_eval_CRAI.sh
```

### 🛠️ Setup Instructions

Before running the script, ensure `run_eval_CRAI.sh` is updated with:

- Paths to:
  - **Input data** (`input_data/`)
  - **Masks** (defining valid and missing pixels)
  - **Model checkpoints**
  - **Output directory** (where infilled data will be saved)
- Configuration details such as:
  - File names of inputs and masks
  - Output file name
  - Device selection: `cuda` (for GPU) or `cpu`

### 📁 Input Files

- `input_data/` should include:
  - `masks/`: observation masks (1 = valid, 0 = missing)
  - `steady_mask`: inverted land/sea mask (1 = sea, 0 = land), required for evaluation

### ⚙️ Configuration File

The script uses `evaluation_spain.inp`, which defines:
- The variable to reconstruct
- Model hyperparameters
- Number of partitions (to manage GPU memory usage)

### 📤 Outputs

After evaluation, results are saved in the `evaluation/` folder:
- `name_output.nc`: **Final reconstruction**, merging model output with original observations
- `name_infilled.nc`: Raw model prediction (infilled data only)
- `name_gt.nc`, `name_image.nc`, `name_mask.nc`: Supporting files (see `evaluation/Output of CRAI evaluation.txt`)

---

## 4. Training

If you wish to train your own model from scratch:

### 📁 Input Structure

- Place training and validation data in `input_data/train/` and `input_data/val/`
- Validation files must have the same names as training files
- Include:
  - **Gridded data files** (complete datasets)
  - **Observation masks** for each timestamp in `input_data/masks/`  
    (reflecting which grid points are valid/missing at each time)
  - **Steady land/sea mask**: 1 = land, 0 = sea
  - **Inverted land/sea mask** (used in evaluation)

### ▶️ Launching Training

Navigate to the `execution/` folder and run:
```bash
bash run_train_CRAI.sh
```

Edit this script to specify:
- Paths to input data and masks
- Device to use (`cpu` or `cuda`)
- Output directories for logs and checkpoints

### 📄 Training Configuration

Defined in `ws_crai_training.inp`, including:
- Batch size
- Learning rate
- Model architecture (layers, attention, etc.)
- Variable names and masks used

### 📦 Training Outputs

- Logs are saved in the `logs/` folder
- Model checkpoints in `snapshots/ckpt/`, including:
  - `best.pth`: best-performing model based on validation loss

---

## 5. CRAI Model Origin

This software is based on the **CRAI model** developed by the **Data Analysis Group** led by Christopher Kadow at the **Deutsches Klimarechenzentrum (DKRZ)**.

### 📚 Reference

Kadow et al. (2020), Nature Geoscience  
**DOI**: [10.1038/s41561-020-0582-5](https://doi.org/10.1038/s41561-020-0582-5)

### 👥 Contributors

Maintained by the **Climate Informatics and Technology Group at DKRZ**  
- *Past*: Naoto Inoue, Christopher Kadow, Stephan Seitz  
- *Present*: Johannes Meuer, Maximilian Witte, Étienne Plésiat

### 🔑 License

CRAI is distributed under the **BSD 3-Clause License**.

---

