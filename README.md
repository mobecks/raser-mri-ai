# RASER AI Code Repository

Official code for the publication:

> **Deep learning corrects artifacts in RASER MRI profiles**  
> Moritz Becker, Filip Arvidsson, Jonas Bertilson, Elene Aslanikashvili, Jan G. Korvink, Mazin Jouda, Sören Lehmkuhl  
> Magnetic Resonance Imaging, Volume 115, January 2025, 110247  
> https://doi.org/10.1016/j.mri.2024.110247

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation & Dependencies](#installation--dependencies)
- [Usage](#usage)
  - [Configuration](#1-configuration)
  - [Data Preparation](#2-data-preparation)
  - [Model Training & Evaluation](#3-model-training--evaluation)
  - [Image Reconstruction & Inference](#4-image-reconstruction--inference)
- [Reference](#reference)
- [Citation](#citation)
- [Contact](#contact)

---

## Project Overview
This repository provides the code for a deep learning pipeline that corrects severe artifacts in RASER MRI images using a two-step approach:

- **1D Correction:** CNN for 1D RASER projections (nonlinear distortion correction)
- **2D Enhancement:** U-net for 2D image enhancement

Models are trained on synthetic data but generalize to experimental data, enabling artifact correction in otherwise unusable RASER MRI images.

## Project Structure
```
raser_mri_ai/
  AI.py                  # Model training & evaluation (1D/2D)
  image_AI.py            # Image reconstruction & inference
  image_data_processing.py # Data preprocessing
  NN_architectures.py    # PyTorch model definitions
  utility_functions.py   # Helper functions
  models/
    config_models.py     # Pydantic experiment/model configs
LICENSE
pyproject.toml
README.md
```

## Dataset
The dataset (synthetic & experimental RASER MRI data) is openly available:
- https://publikationen.bibliothek.kit.edu/1000168053

See the dataset repository for structure and terms.

## Installation
Install dependencies (using Poetry):
```bash
poetry install
```
Or with pip:
```bash
pip install .
```

## Usage

### 1. Configuration
All main scripts use a Pydantic config object (`RaserConfig`), with an optional `cnn` field for CNN hyperparameters (kernel size, filters, activation, etc.). Adjust the `initial_config` variable in each script to control experiment setup, model, and training options.

### 2. Data Preparation
- **Script:** `raser_mri_ai/image_data_processing.py`
- **Purpose:** Preprocess raw/simulated data into NumPy arrays for training/testing.
- **How to use:**
  1. Edit `initial_config` in `image_data_processing.py` for your dataset/system.
  2. Run:
     ```bash
     python -m raser_mri_ai.image_data_processing
     ```
  3. Processed `.npy` files are saved to the output directory.

### 3. Model Training & Evaluation
- **Script:** `raser_mri_ai/AI.py`
- **Purpose:** Train 1D/2D neural networks and evaluate performance.
- **How to use:**
  1. Edit `initial_config` in `AI.py` for model/data/training options.
  2. Run:
     ```bash
     python -m raser_mri_ai.AI
     ```
  3. Results and trained weights are saved in the output directory.

### 4. Image Reconstruction & Inference
- **Script:** `raser_mri_ai/image_AI.py`
- **Purpose:** Apply trained models for image correction and evaluation.
- **How to use:**
  1. Edit `initial_config` in `image_AI.py` for model/input/output settings.
  2. Run:
     ```bash
     python -m raser_mri_ai.image_AI
     ```
  3. Outputs reconstructed images and evaluation results.

---

## Reference
See the paper for detailed methods, datasets, and results:

> **Deep learning corrects artifacts in RASER MRI profiles**  
> Moritz Becker, Filip Arvidsson, Jonas Bertilson, Elene Aslanikashvili, Jan G. Korvink, Mazin Jouda, Sören Lehmkuhl  
> Magnetic Resonance Imaging, Volume 115, January 2025, 110247  
> https://doi.org/10.1016/j.mri.2024.110247

## Citation
If you use this code or data, please cite:

Becker, M., Arvidsson, F., Bertilson, J., Aslanikashvili, E., Korvink, J. G., Jouda, M., & Lehmkuhl, S. (2025). Deep learning corrects artifacts in RASER MRI profiles. Magnetic Resonance Imaging, 115, 110247. https://doi.org/10.1016/j.mri.2024.110247

## Contact
For questions or collaborations, contact the corresponding author as listed in the paper.
