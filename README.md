# Splitting the Dataset into Training and Test Sets

This repository contains a single Jupyter notebook, **`main.ipynb`**, that walks through an end‑to‑end machine learning workflow on the California housing dataset: exploratory analysis, reproducible train/test splitting, feature engineering with pipelines, model training, and hyper‑parameter tuning.

> **Credit & Context**  
> This notebook is part of my study of the book _Hands‑On Machine Learning with Scikit‑Learn, Keras & TensorFlow_ by **Aurélien Géron** (O’Reilly). It closely follows the book’s housing example and related exercises. Full credit for the original approach and educational framing goes to the author and publisher. This repo is for learning purposes only.

## What’s inside

- **Reproducible data split** using `StratifiedShuffleSplit` (with a hash‑based approach on stable IDs, as discussed in the book) to avoid data leakage across reruns.
- **Exploratory data analysis & visualization** of the housing dataset (e.g., geographic scatter plots, correlations).
- **Data preparation pipelines** with Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder for numerical and categorical features.
- **Models trained**: LinearRegression, DecisionTreeRegressor, RandomForestRegressor.
- **Fine‑tuning** with GridSearchCV and evaluation using RMSE.

## File

- `main.ipynb` — the full, runnable notebook.

## Requirements

- **Python**: 3.8.19 (kernel: mlprojects)
- **Key libraries**: numpy, pandas, matplotlib, scikit-learn

Install the basics with:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas matplotlib scikit-learn jupyter
```

> The notebook also calls a helper `load_housing_data()` (from a local `data.py` module in the book’s materials).  
> If you don’t already have it, add a `data.py` that provides `load_housing_data()` and downloads/loads the housing dataset just like in the book.

## How to run

```bash
# 1) Activate your environment (see above)
# 2) Launch Jupyter
jupyter notebook

# 3) Open main.ipynb and run cells top to bottom
```

## Outline (from the notebook)

- Splitting the Dataset into Training and Test Sets
- Splitting a DataFrame
- visualiziation of the data
- prepareing data
- handeling text and categprical attributes
- custom transformer
- Transformation pipelines
- Select and Train a Model
- Fine-Tune the model

## Notes

- The train/test split uses a stable, deterministic strategy (hashing stable IDs) so that reruns don’t leak information between splits—matching the approach explained in the book.
- Evaluation focuses on **regression** metrics (primarily RMSE), since the target variable is continuous (`median_house_value`).
- Feel free to adapt the pipelines or try additional models.

## Attribution

This work is **inspired by and based on** examples from _Hands‑On Machine Learning with Scikit‑Learn, Keras & TensorFlow_ by **Aurélien Géron**.  
© The respective author and publisher for book content; this repo is for educational use only.

---

_Last updated: 2025-08-07_
