# Lightweight Top-Quark Classification with Deep Learning

This project explores how deep learning can be used to identify hadronically decaying top quarks from low-level detector-like data. Using a compact jet-image representation and a lightweight CNN, the model learns to distinguish Top-Quark jets from QCD background jets, achieving:
- Validation Accuracy: ~86.8%
- ROC AUC: ~0.93
- CPU-friendly training (≤16 GB RAM)

## 1. Physics Background

In high-energy proton–proton collisions (e.g., at the LHC), a hadronically decaying top quark follows:

`t → Wb → q q̄′ b`

The resulting three quarks appear as a **three-prong energy pattern** inside a single, boosted jet.  
Standard QCD jets, by contrast, typically show only **one dominant core**.

These structural differences are important for analyses involving boosted-object tagging, new-physics searches, and trigger-level selections. To capture this substructure, raw particle 4-momentum is projected onto a **40×20 η–ϕ grid**, forming a calorimeter-like jet image. Even this simplified representation preserves energy concentration, jet width, and multi-prong patterns—features that CNNs learn effectively.

## 2. Dataset

This project uses the open jet dataset from CERN (Zenodo):  
https://zenodo.org/records/2603256

Each jet has up to 200 particles, stored as four-momentum values:

`E, px, py, pz`

(Zero-padded if fewer than 200 constituents.)

A preprocessed sample of 90k jets used in this project is available here:
**Google Drive:**  
https://drive.google.com/file/d/1ISTa0HIJZT8hH_Zf0tRVVoHQ5Ci1QJmK/view?usp=sharing


To create the Parquet file yourself:

```python
import pandas as pd

df = pd.read_hdf("train.h5", "table").sample(n=90000)
df.to_parquet("jets90000.parquet.gzip", compression="gzip")
```
## 3. CNN Architecture

A compact CNN was chosen to balance physics performance with computational constraints (CPU-only training).

```
┌──────────────────────┐      ┌────────────────────────┐     ┌──────────────────────┐
│    Data Pipeline     │────▶ │   Jet Image Builder    │────▶│    CNN Classifier   │
│                      │      │                        │     │                      │
│ • 4-Momentum Input   │      │ • 40×20 Grid Mapping   │     │ • Conv5×5 (32)       │
│ • Normalization      │      │ • Energy Projection    │     │ • Conv3×3 (64)       │
│ • Train/Val Split    │      │ • Per-Jet Scaling      │     │ • Conv3×3 (128)      │
└──────────────────────┘      └────────────────────────┘     │ • FC(128) + Dropout  │
             │                         │                     │ • Softmax (Top/QCD)  │
             ▼                         ▼                     └───────────┬──────────┘
      ┌────────────────┐      ┌────────────────────────┐                 │
      │ Visualization  │◀────│     Evaluation Tools   │─────────────────┘
      │                │      │                        │
      │ • Jet Images   │      │ • ROC / AUC            │
      │ • Avg Heatmaps │      │ • Confusion Matrix     │
      │ • Comparisons  │      │ • Score Distribution   │
      └────────────────┘      └────────────────────────┘
```
Key hyperparameters:
  
| **HYPERPARAMETER**      | **VALUE** | **DESCRIPTION**                                  |
|-------------------------|-----------|--------------------------------------------------|
| Initial Learn Rate      | 5e-3      | Reduces risk of overshooting the optimum         |
| Learn Rate Schedule     | Piecewise | Simple and stable decay strategy                 |
| Learn Rate Decay        | 0.3       | Multiplies LR by 0.3 at each drop step           |
| Learn Rate Drop Period  | 4 epochs  | Decays LR every 4 epochs                         |
| L2 Regularization       | 1e-4      | Light penalty to control overfitting             |
| Batch Size              | 64        | Balanced between training speed and RAM use      |
| Max Epochs              | 12        | Model begins to overfit past ~14 epochs          |
| Optimizer               | Adam      | Stable and adaptive for small CNNs               |

## 4. Evaluation Results
Achieved 86.8% validation accuracy and AUC ≈ 0.93 using 90k training samples.
### Confusion Matrix
<p align="center">
<img src="results/v1_confusion_matrix.png" width="700">
</p>

### ROC Curve
<p align="center">
<img src="results/v1_roc_curve.png" width="700">
</p>

### Score Distribution
<p align="center">
<img src="results/v1_score_distribution.png" width="700">
</p>

## 5. Jet Image Visualizations

The project includes visualization tools that helped me understand the physics behind the classification:

### Single Jet Visualization: A single jet image showing the sparse, tower-like energy pattern.
<p align="center">
<img src="results/jet123.png" width="300">
</p>

### Signal vs Background: Average jet images highlighting structural differences between top and QCD jets.
<p align="center">
<img src="results/avg_quark.png" width="600">
</p>

### Side-by-Side Comparisons: Direct comparison of individual signal and background jets.
<p align="center">
<img src="results/sig_vs_back.png" width="600">
</p>

## 6. Folder Structure
```
quark_detection/
│
├── data/
│   ├── jets90000.parquet.gzip        # Refer to gdrive
│   ├── cnn_v1_data.mat               # Train/val data
|   ├── cnn_v1_split.mat              # Split info
│   └── cnn_v1_eval.mat               # Evaluation outputs
|   
│
├── model/
│   └── cnn_model.mat                 # Final trained CNN model
│
├── results/
│   ├── v1_confusion_matrix.png
│   ├── v1_roc_curve.png
│   ├── v1_score_distribution.png
│   ├── jet123.png
│   ├── avg_quark.png
│   └── sig_vs_back.png
│
├── scripts/
|   ├── cnn.m
│   ├── evaluatem.m
│   └── visualize_jets.m
│
└── README.md


```
## 7. Running the Project

- Step 1 — Convert data to jet images  
```run scripts/cnn_v1_data.m```

- Step 2 — Train the CNN  
```run scripts/cnn_v1_split.m```

- Step 3 — Evaluate the model  
```run scripts/cnn_v1_eval.m```

- Step 4 — Visualize jets  
```visualize_jets("single", 50);```
