# Semi-MoE: Mixture-of-Experts meets Semi-Supervised Histopathology Segmentation

## Introduction
Semi-MoE is the first multi-task Mixture-of-Experts framework designed for semi-supervised histopathology image segmentation. This model leverages expert networks to enhance segmentation performance, particularly in scenarios with limited labeled data.
<img width="1437" height="946" alt="image" src="https://github.com/user-attachments/assets/3975dd21-89b3-4e97-8d89-3083f6bde052" />


---

## Data Preparation

### 1. Download the Datasets
- **GlaS dataset**: [Download Here](https://datasets.activeloop.ai/docs/ml/datasets/glas-dataset/)
- **CRAG dataset**: [Download Here](https://opendatalab.com/OpenDataLab/CRAG/tree/main)

### 2. Organize the Dataset in the Following Structure
```bash
 dataset/
 ├── GlaS
 │    ├── train_sup_85
 │    │    ├── image
 │    │    │    ├── 1.tif
 │    │    │    ├── 2.tif
 │    │    │    └── ...
 │    │    └── mask
 │    │         ├── 1.png
 │    │         ├── 2.png
 │    │         └── ...
 ├── CRAG
 │    ├── train_sup_35
 │    │    ├── image
 │    │    └── mask
 │    ├── train_unsup_138
 │    │    ├── image
 │    │    └── mask
 │    └── val
 │         ├── image
 │         └── mask
```

---

## Training and Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Test the Model
Load your checkpoint and run:
```bash
python test.py
```
