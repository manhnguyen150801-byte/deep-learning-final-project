# Report

This folder contains the final written report for the project.

---

## File

| File | Description |
|---|---|
| `Team 8_Final Project Report.pdf` | Full written report submitted for the Fundamentals of Deep Learning course |

---

## Report Overview

The report documents the complete methodology, experiments, and findings for both parts of the project. It is structured to accompany the Jupyter notebooks in `notebook/` and the slide deck in `slides/`.

### Structure

**Introduction** — Project motivation and scope across both tasks.

**Part 1 — Flickr Image Classification**
- Dataset loading and EDA (class distribution, sample images)
- Image preprocessing pipeline (normalisation, resizing, batching)
- MLP baselines: architecture diagrams, training curves, comparison table
- Appendix CNN: architecture, overfitting analysis
- Three optimisation experiments (pooling strategy, Batch Normalisation, depth + Dropout)
- Data augmentation: effect on training stability and generalisation
- Transfer Learning with ResNet50: frozen backbone setup and results
- Interpretability: Saliency Maps and Grad-CAM visualisations

**Part 2 — London Weather Time Series Forecasting**
- Dataset description and preprocessing (sliding window, chronological split, MinMaxScaler)
- LSTM hyperparameter grid search (24 configurations)
- GRU hyperparameter grid search (24 configurations)
- 1D-CNN hyperparameter search (8 configurations)
- Model comparison: best LSTM vs. GRU vs. CNN
- Attention mechanism: design, implementation, and visualisation
- Final comparison: Baseline RNN vs. Attention RNN vs. Best CNN

**Conclusion** — Summary of findings, limitations, and potential future work.