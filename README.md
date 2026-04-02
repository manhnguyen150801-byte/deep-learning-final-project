## Project Overview

This project is divided into two independent parts, each exploring a different family of deep learning architectures applied to distinct problem domains.

**Part 1 — Image Classification on Flickr8k**
We apply and compare multiple neural network architectures (MLP, CNN, Transfer Learning with ResNet50) to classify images from the Flickr8k dataset into five semantic categories: animals, objects, people, scenes, and others. The study progressively improves performance through architectural optimisations including Batch Normalisation, Dropout, Global Average Pooling, data augmentation, and finally Transfer Learning.

**Part 2 — Time Series Forecasting on London Weather Data**
We build and compare LSTM, GRU, and 1D-CNN models for univariate temperature prediction using the London weather dataset. The study includes a systematic hyperparameter grid search across sequence lengths, hidden units, and layer depths, followed by an attention-enhanced RNN to improve interpretability and assess whether attention improves predictive performance.

---

## Folder Structure

```
DL Github/
│
├── README.md                        ← You are here (project overview)
├── .gitignore                       ← Python/Jupyter ignore rules
│
├── notebook/                        ← Jupyter notebooks (main analysis)
│   ├── Team 8_Part1_Notebook.ipynb  ← Part 1: Image classification
│   ├── Team 8_Part2_Notebook.ipynb  ← Part 2: Time series forecasting
│   └── README.md                    ← Notebook descriptions & instructions
│
├── report/                          ← Written project report
│   ├── Team 8_Final Project Report.pdf
│   └── README.md                    ← Report structure & submission details
│
├── outputs/                         ← Model results and prediction artefacts
│   └── README.md                    ← Description of outputs and metrics
│
└── slides/                          ← Presentation materials
    ├── Team 8_Final Project Slide.pdf
    └── README.md                    ← Slide deck description
```

---

## How to Run

### Prerequisites

Install all required Python packages:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

> **Note:** A GPU-enabled environment (e.g., Google Colab with T4 runtime) is strongly recommended for Part 1 due to the large image dataset. Part 2 runs comfortably on CPU.

### Part 1 — Image Classification

1. Mount your Google Drive (if using Colab) and ensure the Flickr8k dataset is available at the expected path with the following structure:

```
data/
├── train_image_class.csv
├── valid_image_class.csv
└── <image files>
```

2. Open and run `notebook/Team 8_Part1_Notebook.ipynb` top to bottom.

### Part 2 — Time Series Forecasting

1. Ensure `london_weather.csv` is accessible from Google Drive or a local path.
2. Open and run `notebook/Team 8_Part2_Notebook.ipynb` top to bottom.

### Data Sources

| Dataset | Description | Access |
|---|---|---|
| Flickr8k | ~8,000 images across 5 categories | Provided via course materials |
| London Weather | Daily mean temperature from 2015 onwards | `london_weather.csv` via Google Drive |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` / `keras` | ≥ 2.x | Model building and training |
| `numpy` | ≥ 1.21 | Array operations |
| `pandas` | ≥ 1.3 | Data loading and manipulation |
| `matplotlib` | ≥ 3.4 | Visualisation |
| `scikit-learn` | ≥ 0.24 | Data scaling and evaluation metrics |

---

## Key Results

### Part 1 — Image Classification (Flickr8k, 5 classes)

| Model | Best Validation Accuracy | Notes |
|---|---|---|
| MLP 1 (Baseline) | 54.81% | Unstable; high dimensionality causes variance |
| MLP 2 (Deeper + Dropout) | 53.37% | More stable but no spatial inductive bias |
| CNN (Appendix Architecture) | 59.32% | Overfitting: training acc ~98%, val drops to 53% |
| CNN + 2×2 Pooling + GAP | Improved | Small pooling preserves spatial detail |
| CNN + Batch Normalisation | Stable improvement | Normalisation stabilises gradient flow |
| CNN + Deeper + Dropout | Regularised | Dropout prevents overfitting at depth |
| CNN + Data Augmentation | Best custom CNN | Augmentation improves generalisation |
| **ResNet50 (Transfer Learning)** | **Best overall** | Pre-trained ImageNet features dominate |

**Key finding:** Transfer Learning with a frozen ResNet50 backbone significantly outperforms all from-scratch CNN architectures on this moderately-sized dataset.

### Part 2 — Time Series Forecasting (London Temperature)

| Model | Configuration | Key Metric |
|---|---|---|
| LSTM | Best from 24 configurations (4 seq × 3 units × 2 layers) | Lowest validation MSE |
| GRU | Best from 24 configurations | Comparable to LSTM, faster to train |
| 1D-CNN | Best from 8 configurations (2 seq × 2 filters × 2 kernels) | Competitive with RNNs |
| **RNN + Attention** | Applied to best baseline RNN | Marginal improvement + interpretability |

**Key finding:** All three architectures achieve competitive performance on this univariate regression task. The attention mechanism provides useful interpretability by visualising which past timesteps most influence each prediction.

---

## Interpretability

Both parts include model interpretability analysis:

- **Part 1:** Saliency Maps and Grad-CAM visualisations highlight image regions that activate the classifier most strongly.
- **Part 2:** Attention weight plots show which historical timesteps (days) the model focuses on when forecasting the next day's temperature.

---

## Authors

| Name | Role |
|---|---|
| Qiushuang Liu | Modelling, analysis, report writing |
| Duc Manh Nguyen | Modelling, analysis, report writing |
| Minh Hoan Tran | Modelling, analysis, report writing |
