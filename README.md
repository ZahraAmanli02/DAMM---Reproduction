


# Density-Adaptive Model Based on Motif Matrix for Multi-Agent Trajectory Prediction

This repository contains implementations for reproducing results from the paper **"Density-Adaptive Model Based on Motif Matrix for Multi-Agent Trajectory Prediction"** using the **nuScenes** and **Argoverse** datasets.


## Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Dataset Preparation](#dataset-preparation)
- [Usage Instructions](#usage-instructions)
- [Results & Evaluation](#results--evaluation)
- [Visualization](#visualization)
- [Citation](#citation)
- [Contact](#contact)

## Overview

Density-Adaptive Model Based on Motif Matrix for Multi-Agent Trajectory Prediction" presents a new technique for multi-agent trajectory prediction which makes use of a Density-Adaptive Motif Matrix (DAMM). The main idea is describing better how agents act in complicated situations with adaptive spatial and temporal interactions. In contrast to existing techniques that depend primarily on graph-based or transformer-based models, DAMM explicitly adds a Spatial-Temporal Motif Matrix (STMM), which dynamically encodes relational information across agents.

This work is a reproduction of the CVPR 2024 paper:

> [**"Density-Adaptive Model Based on Motif Matrix for Multi-Agent Trajectory Prediction"**](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_Density-Adaptive_Model_Based_on_Motif_Matrix_for_Multi-Agent_Trajectory_Prediction_CVPR_2024_paper.html)

> *Di Wen, Haoran Xu, Zhaocheng He, Zhe Wu, Guang Tan, Peixi Peng* 


Key features:

- **Adaptive Spatial Interaction (ASI)** and **Adaptive Temporal Interaction (ATI)** modules
- **Graph Attention Network (GAT)** for relational modeling
- Experiments on **nuScenes** and **Argoverse** datasets

## Setup Instructions

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```
- Python 3.8+
- Google Colab (if running on Colab)


**Library Versions:**

- `torch`
- `torchvision`
- `torchaudio`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

**For more information: requirements.txt**


### Cloning the Repository

```bash
git clone https://github.com/your-repository-name.git
cd your-repository-name
```

## Dataset Preparation

To run the experiments, you need to obtain the dataset first. There are two ways to acquire the dataset:

### Option 1: Download from the Official Website
The datasets can be downloaded from their official sources:
- **Argoverse:** [https://www.argoverse.org](https://www.argoverse.org) 
- **nuScenes:** [https://www.nuscenes.org](https://www.nuscenes.org)

After downloading, extract the dataset and organize it into `train`, `val`, and `test` folders before use.

### Option 2: Load from Cloud
Instead of manually downloading the dataset, you can directly load it from the cloud. The datasets are stored in Google Drive, and the scripts provided in this repository will automatically mount the drive and load the required files.

Ensure that your Google Drive has the datasets stored in the following structure:
```
/content/drive/My Drive/Colab Notebooks/nuScenes/
    ├── train/
    ├── val/
    ├── test/
```
```
/content/drive/My Drive/Colab Notebooks/Argoverse/
    ├── train/
    ├── val/
    ├── test_obs/
```
If the datasets are not available in your Google Drive, manually upload them or download them from the official website.



## Usage Instructions

### Running Training & Evaluation

Since the implementation is provided in Jupyter Notebooks, you need to open and run them directly:

- **For Argoverse:** Open and run `Zahra_Amanli_Argoverse.ipynb`
- **For nuScenes:** Open and run `Zahra_Amanli_nuScenes.ipynb`

## Results & Evaluation

The performance of different model variants is evaluated using **ADE (Average Displacement Error)** and **FDE (Final Displacement Error)**. The results are summarized in tables and plots comparing reproduced results with the original paper.

### Example Ablation Study Results:

After training and evaluating the model, we report the following key trajectory prediction metrics:

**ADE (Average Displacement Error):** Measures the average Euclidean distance between the predicted and ground truth trajectories over all time steps.

**FDE (Final Displacement Error):** Measures the Euclidean distance between the predicted and ground truth final positions at the last time step.

**Results Example (nuScenes dataset):**

| Model Variant   | ADE1/FDE1   | ADE5/FDE5   | ADE10/FDE10 |
| --------------- | ----------- | ----------- | ----------- |
| M1 (Baseline)   | 3.20 / 6.90 | 2.01 / 4.92 | 1.70 / 3.60 |
| M2 (STMM + ASI) | 3.12 / 6.75 | 1.98 / 4.83 | 1.66 / 3.52 |
| M3 (STMM + ATI) | 3.11 / 6.80 | 1.97 / 4.88 | 1.67 / 3.55 |
| M4 (STMM + ASI + ATI) | 2.95 / 6.60 | 1.90 / 4.70 | 1.63 / 3.48 |

The full model (M4: STMM + ASI + ATI) achieves the lowest ADE and FDE scores, demonstrating the effectiveness of integrating spatial and temporal motifs.

**Ablation Study Results**

To assess the contribution of each module, an ablation study was conducted comparing the following:

- **Baseline model (M1)** – No spatial or temporal motifs.
- **STMM + ASI (M2)** – Spatial interactions only.
- **STMM + ATI (M3)** – Temporal interactions only.
- **Full model (M4)** – Both spatial and temporal interactions.

**Findings:**

- Removing STMM (Spatial-Temporal Motif Matrix) significantly increases ADE and FDE errors.
- Combining ASI (Adaptive Spatial Interaction) and ATI (Adaptive Temporal Interaction) leads to the best performance.

## Visualization

**Reproduced Figure 1: High-Density vs. Low-Density Scenarios**

The visualization showcases the behavior of the Density-Adaptive Motif Matrix (DAMM) in high-density and low-density settings.

- **Target Agent:** Represented as a red star.
- **Neighbor Selection Range:** Indicated by a blue dashed circle.
- **Agents:** Other moving entities displayed as green circles.

**Key Findings from the Reproduced Figure:**

- The model successfully captures density-dependent trends, dynamically adjusting the selection of neighboring agents based on density conditions.
- The selection radius expands in low-density scenarios to compensate for fewer neighboring agents.

## Citation

If you find this repository useful, please consider citing the original paper:

```
@inproceedings{wen2024damm,
  title={Density-Adaptive Model Based on Motif Matrix for Multi-Agent Trajectory Prediction},
  author={Wen, et al.},
  booktitle={CVPR},
  year={2024}
}
```


## Contact

My mail: zahra.amanli@fau.de

My personal mail: zahra.amanli.za@gmail.com

My LinkedIn profile: www.linkedin.com/in/zahra-amanli 



---

**Zahra Amanli**
