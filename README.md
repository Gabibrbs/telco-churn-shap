# Customer churn prediction explained with SHAP

Predicting which telecom customers will cancel, and explaining why, with a logistic regression and SHAP. Built for Module 3 (Learning and its libraries) of the AI Specialisation at Tokio School.

> The notebook is written in European Portuguese, as it was submitted for the course. All numbers below come from the saved outputs in the notebook (single run, `random_state=90`).

## Data

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle): 7,043 customers and 21 columns about contract, services and charges. The target, `Churn`, says whether the customer left.

The CSV is not included in this repository. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and put it in the same folder as the notebook.

## What the notebook does

1. **pandas and Matplotlib:** loads and explores the data. `TotalCharges` came as text because 11 customers had it blank. All 11 had 0 months of tenure, so they were removed (7,032 customers left). 26.6% of customers churned, so the classes are imbalanced. Churn is 42.7% on monthly contracts against 2.8% on two-year contracts.
2. **scikit-learn:** one-hot encoding (30 features), a stratified 80/20 split, and a logistic regression evaluated with accuracy, precision, recall and F1-score.
3. **SHAP:** a beeswarm plot for the global view and force plots to explain individual customers.

## Results (test set)

| Metric | Value |
|---|---|
| Accuracy | 0.802 |
| Accuracy of a model that always says "stays" | 0.734 |
| Precision | 0.640 |
| Recall | 0.588 |
| F1-score | 0.613 |

Train and test accuracy are close (0.807 and 0.802), so there is no clear overfitting. Accuracy alone is misleading here: the model misses about four in ten churners (recall 0.588), which is the costliest error for a company trying to keep customers.

**What SHAP shows:** tenure is the most important feature (the longer the customer, the more the model pushes towards staying), followed by total charges, the two-year contract and fibre optic internet. Two-year and one-year contracts, tech support and online security push towards staying. Fibre optic, paperless billing and electronic cheque payments push towards churn.

Some features overlap: tenure and total charges have a correlation of 0.82. The model splits importance between them, so the sign of one feature alone should not be read as a cause. SHAP explains the model, not what happens in real life.

## Limitations and next steps

- A simple linear model, one train/test split, and the default 50% threshold.
- Next steps: try `class_weight` or a lower threshold to catch more churners, compare other models with K-fold cross-validation, and drop redundant features.

## How to run

```
pip install -r requirements.txt
```

Put the CSV next to `telco_churn_shap.ipynb` and run all cells.
