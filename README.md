# Financial Fraud Detection

Detecting fraudulent credit card transactions with machine learning, with a focus on **business cost impact** rather than accuracy.

## Problem

A bank wants to flag fraudulent credit card transactions in near real-time.

- **Missing a fraud** (false negative) costs the transaction amount.
- **A false alarm** (false positive) costs customer trust / support time, assumed at a fixed **$10** per alert (configurable in `src/config.py`).

Only **0.17%** of transactions are fraud, so accuracy is meaningless (a model that flags nothing is 99.8% "accurate"). Instead, models are compared on **precision-recall AUC** and on **total business cost**, and the decision threshold is chosen to minimise cost instead of defaulting to 0.5.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group): 284,807 European card transactions over two days, 492 of them fraud. Features `V1`-`V28` are anonymised PCA components, plus `Time`, `Amount` and the label `Class`. The data is not stored in this repo - a script downloads it.

## Approach

1. **EDA** (`01_eda`): class imbalance, amount/time patterns, which features separate fraud.
2. **Cleaning & splitting** (`02_cleaning`): remove duplicates, engineer features, then a stratified **train / validation / test** split *before* any resampling to avoid leakage.
3. **Modelling** (`03_modeling`): four models compare imbalance strategies - logistic regression and XGBoost, each with class weights or SMOTE. The winner is chosen on validation **cost**, then evaluated once on the untouched test set. Includes a sensitivity analysis on the false-alarm cost assumption.
4. **Insights** (`04_insights`): SHAP explanations, analysis of the frauds the model still misses, limitations.
5. **Dashboard** (`dashboard/app.py`): Streamlit app to explore how the optimal threshold and total cost change with the false-alarm cost.

## Results

> Fill this in after running the notebooks on the real data (numbers come from `03_modeling` and `04_insights`).

| Strategy | Test-set cost | Frauds caught |
|---|---|---|
| Do nothing | _..._ | 0 |
| Best model (_name_) | _..._ | _.. / .._ |

Key findings: _(2-3 bullets in your own words: best model, loss reduction vs doing nothing, top SHAP features)_

![Cost vs threshold](reports/figures/03_cost_vs_threshold.png)
![SHAP summary](reports/figures/04_shap_summary.png)

## Project structure

```
data/raw/          raw data (downloaded, git-ignored)
data/processed/    train / val / test splits (git-ignored)
notebooks/         01_eda, 02_cleaning, 03_modeling, 04_insights
src/               config, data_prep, features, evaluate (cost model), model
dashboard/         Streamlit cost explorer
models/            saved model (git-ignored)
reports/figures/   plots used in the notebooks and README
```

## How to run

```bash
pip install -r requirements.txt

# 1. get the data (needs a Kaggle API token in ~/.kaggle/access_token)
python src/download_data.py

# 2. either run the notebooks in order (01 -> 04) ...
# 3. ... or run the pipeline from the command line:
python -m src.data_prep      # clean + split -> data/processed/
python -m src.model          # train, compare, evaluate, save best model

# 4. explore interactively
streamlit run dashboard/app.py
```

## Limitations

- Features are anonymised, so SHAP can rank them but not translate them into business rules.
- Only two days of data: no seasonality, and no check of how the model ages (a time-based split would be stricter).
- The false-alarm cost is an assumption; the sensitivity analysis shows how much the decision depends on it.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, Streamlit, Matplotlib / Seaborn.
