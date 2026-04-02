# Notebooks

This folder contains the two Jupyter notebooks that form the core analysis of the project.

---

## Files

| File | Description |
|---|---|
| `Team 8_Part1_Notebook.ipynb` | Image classification on the Flickr8k dataset |
| `Team 8_Part2_Notebook.ipynb` | Time series temperature forecasting on London weather data |

---

## Part 1 — Flickr Image Classification (`Team 8_Part1_Notebook.ipynb`)

### Objective

Classify images from the Flickr8k dataset into five semantic categories — **animals, objects, people, scenes, and others** — using progressively more sophisticated deep learning architectures.

### Dataset

- **Name:** Flickr8k
- **Input:** RGB images resized to 224 × 224 pixels, normalised to [0, 1]
- **Labels:** 5 classes loaded from `train_image_class.csv` and `valid_image_class.csv`
- **Batch size:** 32 (via `ImageDataGenerator.flow_from_dataframe`)

### Notebook Structure

1. **Data Loading & EDA** — Load CSVs, display class distribution via countplots, visualise sample images per category.
2. **Image Preprocessing** — Normalise pixel values, standardise size to 224×224, configure data generators.
3. **MLP Experiments** — Train and compare two Multi-Layer Perceptron baselines on flattened images.
4. **Appendix CNN** — Build the CNN prescribed in the project appendix; analyse severe overfitting.
5. **Optimisation Experiments** — Three targeted improvements over the appendix CNN:
   - Experiment 1: Replace large pooling with 2×2 MaxPooling + GlobalAveragePooling
   - Experiment 2: Add Batch Normalisation after convolutional layers
   - Experiment 3: Deepen the network and add Dropout for regularisation
6. **Data Augmentation** — Apply random rotations, shifts, flips, and zoom to improve generalisation.
7. **Transfer Learning (ResNet50)** — Fine-tune a frozen ResNet50 backbone pre-trained on ImageNet.
8. **Interpretability** — Visualise Saliency Maps and Grad-CAM heatmaps to explain model decisions.

### Models Trained

| Model | Architecture Summary | Parameters |
|---|---|---|
| MLP 1 | Flatten → Dense(64) → Dense(32) → Softmax(5) | ~9.6M |
| MLP 2 | Flatten → Dense(128) → Dropout(0.3) → Dense(64) → Dense(32) → Softmax(5) | ~19.3M |
| Appendix CNN | Conv2D(16) → MaxPool(4×4) → Conv2D(32,gelu) → MaxPool(6×6) → Dense(128) → Softmax(5) | ~47.7M |
| Exp 1 CNN | Conv2D → MaxPool(2×2) → Conv2D → MaxPool(2×2) → GAP → Softmax(5) | Reduced |
| Exp 2 CNN | Conv2D + BatchNorm → MaxPool → Conv2D + BatchNorm → MaxPool → Dense → Softmax(5) | Stable |
| Exp 3 CNN | 2× [Conv2D + Conv2D + MaxPool + Dropout(0.25)] → Dense(512) + Dropout(0.5) → Softmax(5) | Deep |
| Augmented CNN | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → GAP → Softmax(5) | Best custom |
| ResNet50 + Head | ResNet50 (frozen) → GAP → Dense(512) → Dropout(0.5) → Softmax(5) | Transfer |

### Key Results

| Model | Best Validation Accuracy |
|---|---|
| MLP 1 | 54.81% (Epoch 8, then drops) |
| MLP 2 | 53.37% (very stable, no improvement) |
| Appendix CNN | 59.32% (then severe overfitting) |
| Exp 1–3 CNNs | Progressive improvement over appendix |
| ResNet50 | **Best overall** |

### Dependencies

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv2D, MaxPooling2D,
                                     BatchNormalization, Activation,
                                     GlobalAveragePooling2D, Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
```

---

## Part 2 — London Weather Time Series Forecasting (`Team 8_Part2_Notebook.ipynb`)

### Objective

Predict the **next day's mean temperature** from a sliding window of past daily observations using LSTM, GRU, and 1D-CNN architectures. Extend the best RNN with an **attention mechanism** to improve interpretability.

### Dataset

- **Name:** London Weather Dataset (`london_weather.csv`)
- **Period:** 2015–present (filtered from full dataset)
- **Features used:** `date`, `mean_temp` (univariate)
- **Split:** 70% train / 10% validation / 20% test (chronological order preserved)
- **Normalisation:** MinMaxScaler fitted on training set only; inverse-transformed for evaluation

### Notebook Structure

1. **Data Loading & Cleaning** — Load CSV, filter to 2015+, parse dates, drop NaNs.
2. **EDA** — Plot temperature time series over time.
3. **Sequence Construction** — Sliding window function `create_sequences(data, seq_length)` transforms the series into (X, y) pairs.
4. **Train/Val/Test Split** — Chronological split with scaler fitted on train only.
5. **LSTM Hyperparameter Search** — 24 configurations (4 seq lengths × 3 unit sizes × 2 layer depths), early stopping on val loss.
6. **GRU Hyperparameter Search** — Same 24-configuration grid as LSTM.
7. **CNN Hyperparameter Search** — 8 configurations (2 seq lengths × 2 filter counts × 2 kernel sizes).
8. **Model Comparison** — Side-by-side comparison of best LSTM, GRU, and CNN.
9. **Attention-Enhanced RNN** — Custom attention layer (Dense tanh → Softmax → element-wise multiply → reduce_sum) applied to the best baseline RNN.
10. **Final Comparison** — Baseline RNN vs. Attention RNN vs. Best CNN on test metrics.
11. **Visualisations** — Loss curves, prediction overlays (first 150 steps), attention weight bar plots.

### Hyperparameter Search Space

| Parameter | Values Tested |
|---|---|
| Sequence length (`seq_length`) | 3, 7, 12, 21 days |
| RNN hidden units | 32, 50, 64 |
| Number of RNN layers | 1, 2 |
| CNN filters | 32, 64 |
| CNN kernel size | 2, 3 |
| Batch size | 16 |
| Max epochs | 30 (with early stopping, patience=5) |
| Dropout rate | 0.2 |
| Optimiser | Adam |
| Loss | Mean Squared Error (MSE) |

### Key Results

All models are evaluated on the test set using **MSE** and **MAE** on the original temperature scale (inverse-scaled from MinMaxScaler). Comparison tables are generated in the notebook:

- `lstm_hyper_df` — all 24 LSTM configurations ranked by validation loss
- `gru_hyper_df` — all 24 GRU configurations ranked by validation loss
- `cnn_hyper_df` — all 8 CNN configurations ranked by validation loss
- `final_comparison_df` — best Baseline RNN vs. Attention RNN vs. Best CNN

The **attention mechanism** offers marginal quantitative improvement over the baseline but provides meaningful interpretability: attention weight plots reveal which past days most influence the forecast.

### Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Conv1D,
                                     Flatten, Multiply, Softmax, Lambda, Input)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
```

---

## How to Run

Both notebooks are designed to run in **Google Colab** with Google Drive mounted. To run locally, replace Drive mount calls with direct file path references.

```bash
# Install dependencies (if running locally)
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Run each notebook from top to bottom. All random seeds are set to `42` for reproducibility.
