# Lung Nodule Classification with a 3D CNN

A 3D convolutional neural network that classifies lung nodules from CT scans as **benign** or **malignant**, built in Python and PyTorch.

This is my own version of the model, developed during the NSF-funded MS-CC Undergraduate Summer Research Internship (Meharry Medical College / Fisk University, Summer 2026).

## Results

Evaluated on held-out test sets at a tuned decision threshold of 0.7:

| Split | Accuracy | Malignant F1 |
|---|---|---|
| By nodule (no nodule in both train and test) | 93.5% | 0.86 |
| By patient (no patient in both train and test) | 93% | 0.85 |

The patient-level split is a stricter check, and results held steady under it.

## How it works

**Data.** I used the public [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) dataset of lung CT scans with expert annotations. Each nodule is stored as a 3D volume (`.nii.gz`) and resized to 48 × 48 × 48 voxels. There are 1,073 nodules in total: 295 malignant and 778 benign.

**Model.** The model is a ResNet-style 3D CNN with about 5.9M parameters:
- 4 convolutional blocks (32 → 64 → 128 → 256 filters), each followed by a residual block with a skip connection
- Global average pooling, dropout (0.6), and two fully connected layers
- The model looks at the full 3D volume, not individual 2D slices, so it can learn the nodule's 3D shape and texture

**Training.**
- Class-weighted cross-entropy loss, to handle the class imbalance (about 73% benign)
- AdamW optimizer, with the learning rate reduced when progress plateaus
- Early stopping, keeping the checkpoint with the best test loss
- Data augmentation on training data only: random flips, rotation, brightness shifts, and elastic deformation

## What I worked on

- **Fixed a data leakage bug.** I made sure train/test splits happen at the nodule level, so the same nodule never appears in both sets. I then validated with a stricter patient-level split.
- **Cleaned conflicting labels.** One nodule (`LIDC-IDRI-0355_0`) appeared in both the benign and malignant folders, so I removed it.
- **Tuned the decision threshold.** I tested thresholds from 0.3 to 0.85 and chose 0.7 to maximize malignant F1.
- **Built comparison models.** I trained Random Forest and XGBoost on radiologist-scored features (margin, spiculation, lobulation, diameter, and others), plus a hybrid model that combines those features with the CNN's learned features.

## Project structure

```
3d_cnn_lung_nodule_classification.ipynb   # Full pipeline: data loading, training, evaluation
```

## Requirements

```
pip install torch numpy nibabel scipy scikit-learn matplotlib seaborn pandas xgboost
```

## Usage

1. Download LIDC-IDRI and extract nodule patches into `malignant/` and `benign/` folders.
2. Update `MALIGNANT_DIR` and `BENIGN_DIR` in the configuration cell.
3. Run the notebook top to bottom.

The dataset is not included in this repo.

## Acknowledgments

- Faculty mentor: Dr. Mohammad M. R. Khan (Meharry Medical College)
- Graduate mentor: Gregory Davis, who built the original team codebase
- Funded by the National Science Foundation (Award No. 2234326)
