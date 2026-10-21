# Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The project compares three modelling approaches:

- Random Forest with SMOTE oversampling
- Isolation Forest for anomaly detection
- XGBoost with class weighting

The goal is not simply to maximise accuracy. Fraud represents only 0.17% of observations, so the analysis focuses on recall, F2-score, PR-AUC and ROC-AUC.

## Results

The final comparison used 10-fold stratified cross-validation and a separate hold-out test set.

| Model | Mean CV F2 | Test F2 | Test Recall | Test Precision |
| --- | ---: | ---: | ---: | ---: |
| Random Forest + SMOTE | 0.8380 | 0.7928 | 0.8378 | 0.6526 |
| XGBoost | 0.8360 | 0.7881 | 0.8243 | 0.6703 |
| Isolation Forest | 0.4125 | 0.3914 | 0.6622 | 0.1485 |

Random Forest with SMOTE was selected as the final model. It achieved the strongest F2-score on the hold-out test set while keeping recall high.

The Wilcoxon signed-rank test was applied to paired PR-AUC values from the 10-fold cross-validation:

- Random Forest + SMOTE vs. XGBoost: no statistically significant difference (`p = 0.0840`).
- Isolation Forest vs. either supervised model: statistically significant difference (`p = 0.0020`).

## Project Structure

```text
.
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── src/
│   ├── config.py
│   ├── features.py
│   ├── metrics.py
│   └── models.py
├── tests/
│   ├── test_features.py
│   └── test_metrics.py
├── data/
│   └── .gitkeep
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) published on Kaggle.

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it either in the repository root or in `data/`.

## Setup

```bash
git clone https://github.com/tretiar/credit-card-fraud-detection
cd credit-card-fraud-detection
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open `notebooks/credit_card_fraud_detection.ipynb` in Jupyter Notebook or VS Code and run the cells from top to bottom.

The notebook contains the analysis and experiments; reusable preprocessing, metric and model-building code is stored in `src/`.

Run the unit tests with:

```bash
python -m pytest
```

## Methodology

1. Explore the class imbalance, missing values, duplicates, transaction amounts and time-related patterns.
2. Create cyclical time features and scale `Amount` and `Time` with `RobustScaler`.
3. Split data with stratification before fitting transformations to avoid leakage.
4. Compare Random Forest + SMOTE, Isolation Forest and XGBoost.
5. Tune several manual configurations and select the best candidate from each model family.
6. Evaluate candidates with 10-fold stratified cross-validation.
7. Compare paired PR-AUC scores with the Wilcoxon signed-rank test.
8. Refit the selected candidates and evaluate them once on the hold-out test set.

## Limitations

- The data covers roughly two days of transactions, so it does not assess temporal drift.
- The features `V1`–`V28` are PCA-transformed and have limited business interpretability.
- Decision thresholds are selected to optimise F2, which prioritises recall over precision.
- A production deployment would require time-based validation, monitoring and periodic retraining.

## Skills Demonstrated

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, feature engineering, imbalanced classification, cross-validation, statistical testing and model evaluation.
