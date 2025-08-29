# Semi-MoE: Mixture-of-Experts meets Semi-Supervised Histopathology Segmentation

## Introduction
Semi-MoE is the first multi-task Mixture-of-Experts framework designed for semi-supervised histopathology image segmentation. This model leverages expert networks to enhance segmentation performance, particularly in scenarios with limited labeled data.
![image](https://github.com/user-attachments/assets/dfa20392-3037-4e8c-84ce-271f6448ac8d)

---

## 📂 Data Preparation

### 1️⃣ Download the Datasets
- **GlaS dataset**: [Download Here](https://datasets.activeloop.ai/docs/ml/datasets/glas-dataset/)
- **CRAG dataset**: [Download Here](https://opendatalab.com/OpenDataLab/CRAG/tree/main)

### 2️⃣ Organize the Dataset in the Following Structure
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

## 🚀 Training and Testing

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```bash
python train.py
```

### 3️⃣ Test the Model
Load your checkpoint and run:
```bash
python test.py
```
