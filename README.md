# 🎗️ Breast Cancer Detection using Machine Learning
### Classifying Malignant vs Benign Tumors with Logistic Regression

<div align="center">
  <img src="banner.png" alt="Breast Cancer Detection" width="95%">
</div>

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Classification](https://img.shields.io/badge/Task-Classification-teal?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Live Web App

> 🔗 [Click here to open the Breast Cancer Detection App](https://onchoscan-breast-cancer-classification.streamlit.app/)

---

## What This Project Does

This is a machine learning project that classifies breast tumors as either **Malignant** (cancerous) or **Benign** (non-cancerous) based on diagnostic measurements.

I built this as a complete end-to-end ML project — covering everything from data exploration to model deployment. The goal was to practice the full ML workflow, not just training a model, but also doing feature engineering, building preprocessing pipelines, comparing multiple algorithms, and wrapping the result into a web app.

---

## Why Breast Cancer Detection?

Early detection is one of the most important factors in surviving breast cancer. The earlier a tumor is caught and correctly classified, the better the treatment options. ML can support doctors by providing fast, data-driven classification from measurable features — especially useful where specialist access is limited.

---

## Dataset

Used the **built-in Breast Cancer Wisconsin dataset from scikit-learn** — no downloads needed. It contains real diagnostic measurements from digitized biopsy images.

| Property       | Value                           |
|----------------|---------------------------------|
| Samples        | 569 patient records             |
| Features       | 30 numeric biomarkers           |
| Target classes | 0 = Malignant, 1 = Benign       |
| Missing values | None                            |

---

## Project Structure

```
📁 Breast_Cancer_Detection/
├── app.py                         ← Streamlit web app
├── banner.png                     ← Project banner
├── README.md
├── requirements.txt
├── notebook/
│   └── cancer.ipynb               ← Full ML notebook
└── models/
    └── best_breast_cancer_model.pkl
```

---

## Feature Engineering

Before training, I added two extra derived columns. This was mainly to practice using different encoders inside a pipeline:

- **`tumor_size_category`** (Ordinal): Small / Medium / Large — based on `mean radius` bins
- **`texture_type`** (Nominal): Smooth / Rough — based on a median split of `mean texture`

---

## Preprocessing Pipeline

I used a `ColumnTransformer` to handle different column types cleanly:

| Column Type | Columns                   | Transformer     |
|-------------|---------------------------|-----------------|
| Numerical   | All 30 original features  | StandardScaler  |
| Ordinal     | `tumor_size_category`     | OrdinalEncoder  |
| Nominal     | `texture_type`            | OneHotEncoder   |

Everything is wrapped in a `Pipeline` so preprocessing is applied consistently at both training and prediction time.

---

## Model Comparison

Trained 7 models with **5-fold cross-validation** on the training set, then evaluated each on a held-out test set.

| Model                       | CV Accuracy | Test Accuracy |
|-----------------------------|-------------|---------------|
| Logistic Regression         | 97.80%      | 97.37%        |
| Ridge (L2) Regression       | 97.80%      | 97.37%        |
| **Lasso (L1) Regression**   | **97.36%**  | **98.25%** ✅ |
| Support Vector Machine      | 97.14%      | 97.37%        |
| K-Nearest Neighbors         | 96.48%      | 97.37%        |
| Decision Tree               | 92.09%      | 91.23%        |
| Random Forest               | 95.82%      | 94.74%        |

**Winner: Lasso (L1) Logistic Regression — 98.25% test accuracy.**

The L1 penalty pushes less-useful weights toward zero, keeping the model simple and reducing overfitting. That likely explains why it edged out the others.

---

## Results

- Best test accuracy: **98.25%**
- Good generalization confirmed via 5-fold CV
- Final model saved with `joblib` as `best_breast_cancer_model.pkl`

The saved `.pkl` includes the full pipeline (preprocessor + model), so it works out of the box at inference time.

---

## Run Locally

```bash
git clone https://github.com/harisyar-ai/breast-cancer-detection.git
cd breast-cancer-detection
pip install -r requirements.txt
```

Open `cancer.ipynb` and run all cells to reproduce the training. Then launch the app:

```bash
streamlit run app.py
```

---

## Possible Improvements

- Add ROC curves and AUC scores to the notebook
- Use SHAP for feature importance and explainability
- Try stacking or ensemble methods
- Add more input controls to the web app

---

```
Developed by Muhammad Haris Afridi
February 2026

GitHub: github.com/harisyar-ai
Stars and feedback are appreciated!
```
