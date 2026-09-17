# 📉 Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn, based on account and usage data. Includes a full model comparison in a notebook and a deployable Streamlit app for live predictions.

## Overview

The project trains and compares three classification models on a customer churn dataset, then saves the best-performing one for use in an interactive web app.

- **Data cleaning & EDA** on the raw customer dataset
- **Model training**: Logistic Regression, Random Forest, and XGBoost
- **Evaluation**: accuracy, precision, recall, F1, ROC AUC, confusion matrices, and 5-fold cross-validation
- **Deployment**: a Streamlit form where you enter a customer's details and get a churn prediction

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.827 | 0.814 | 0.823 | 0.819 | 0.903 |
| Random Forest | 0.998 | 0.999 | 0.997 | 0.998 | 1.000 |
| **XGBoost (selected)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

XGBoost was selected as the best model (ranked by ROC AUC, then recall, then F1) and is the model used in the app. 5-fold cross-validation confirmed the result (mean ROC AUC = 1.0000).

## Project structure

```
.
├── app.py                                 # Streamlit app for live predictions
├── main.ipynb                             # Data cleaning, EDA, model training & comparison
├── best_model.pkl                         # Saved model bundle (model + scaler + column info)
├── customer_churn_dataset-master.csv      # Training/reference dataset
└── requirements.txt                       # Python dependencies
```

## Dataset

Each row represents one customer, with the following fields:

- `Age`, `Gender`
- `Tenure` (months as a customer)
- `Usage Frequency`, `Support Calls`, `Payment Delay`
- `Subscription Type` (Basic / Standard / Premium)
- `Contract Length` (Monthly / Quarterly / Annual)
- `Total Spend`, `Last Interaction`
- `Churn` (target: 1 = churned, 0 = retained)

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook (optional — to retrain)

Open `main.ipynb` to see the full data cleaning, EDA, and model comparison workflow, or to retrain the model on new data. Retraining will overwrite `best_model.pkl`.

### 3. Launch the app

```bash
streamlit run app.py
```

This opens a browser form where you can enter a customer's numerical and categorical details and click **Predict** to see whether they're likely to churn.

## How the app works

1. Loads the saved model bundle (`best_model.pkl`) — the trained XGBoost model, the fitted `StandardScaler`, and the expected feature columns.
2. Builds an input form from the reference dataset's columns (numeric fields default to the column median; categorical fields are dropdowns of observed values).
3. One-hot encodes the categorical inputs and aligns them to the model's expected feature columns.
4. Scales the numerical inputs with the saved scaler.
5. Runs the prediction and displays the result: ✅ likely to stay, or ⚠️ likely to churn.

## Notes

- A high-cardinality ID column (`CustomerID`) is dropped before use — it doesn't hold predictive information.
- The near-perfect scores on this dataset suggest it may be synthetic or have low label noise; treat the metrics as a demonstration of the pipeline rather than a benchmark for real-world churn data.
