"""Model definitions, training, comparison and saving.

Run the whole pipeline from the repo root:
    python -m src.model
"""
import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import FALSE_ALARM_COST, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET
from src.data_prep import load_splits, prepare
from src.evaluate import (baseline_costs, best_threshold, savings_vs_no_model,
                          total_cost)
from src.features import get_feature_columns


def build_models(scale_pos_weight: float) -> dict:
    """Four ways of dealing with 0.17% fraud: class weights vs SMOTE,
    on a simple linear model and on gradient boosting."""

    def xgb():
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE,
        )

    models = {
        "logreg_balanced": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "logreg_smote": ImbPipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "xgb_weighted": Pipeline([("clf", xgb().set_params(scale_pos_weight=scale_pos_weight))]),
        "xgb_smote": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", xgb()),
        ]),
    }
    return models


def split_xy(df: pd.DataFrame, feature_cols=None):
    feature_cols = feature_cols or get_feature_columns(df)
    return df[feature_cols], df[TARGET], df["Amount"]


def compare_models(train, val, false_alarm_cost=FALSE_ALARM_COST):
    """Fit every model on `train`, score on `val`, pick each model's cost-optimal
    threshold on `val`. Returns (results_df, fitted_models, val_scores)."""
    X_tr, y_tr, _ = split_xy(train)
    X_va, y_va, amt_va = split_xy(val, X_tr.columns.tolist())

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    fitted, val_scores, rows = {}, {}, []
    base = baseline_costs(y_va, amt_va, false_alarm_cost)

    for name, model in build_models(spw).items():
        print(f"training {name} ...")
        model.fit(X_tr, y_tr)
        scores = model.predict_proba(X_va)[:, 1]
        thr, cost = best_threshold(y_va, scores, amt_va, false_alarm_cost)
        fitted[name], val_scores[name] = model, scores
        rows.append({
            "model": name,
            "pr_auc": average_precision_score(y_va, scores),
            "roc_auc": roc_auc_score(y_va, scores),
            "best_threshold": thr,
            "val_cost": cost,
            "savings_vs_no_model": savings_vs_no_model(cost, y_va, amt_va, false_alarm_cost),
        })

    results = pd.DataFrame(rows).sort_values("val_cost").reset_index(drop=True)
    results.attrs["baselines"] = base
    return results, fitted, val_scores


def evaluate_on_test(model, threshold, test, feature_cols, false_alarm_cost=FALSE_ALARM_COST):
    """Final, one-shot evaluation using the threshold chosen on validation."""
    X_te, y_te, amt_te = split_xy(test, feature_cols)
    scores = model.predict_proba(X_te)[:, 1]
    pred = (scores >= threshold).astype(int)
    cost = total_cost(y_te, pred, amt_te, false_alarm_cost)
    return {
        "scores": scores,
        "pred": pred,
        "cost": cost,
        "baselines": baseline_costs(y_te, amt_te, false_alarm_cost),
        "savings_vs_no_model": savings_vs_no_model(cost, y_te, amt_te, false_alarm_cost),
        "pr_auc": average_precision_score(y_te, scores),
        "roc_auc": roc_auc_score(y_te, scores),
    }


def save_model(model, name, threshold, feature_cols, false_alarm_cost=FALSE_ALARM_COST):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    meta = {
        "model_name": name,
        "threshold": threshold,
        "feature_columns": feature_cols,
        "false_alarm_cost": false_alarm_cost,
    }
    (MODELS_DIR / "best_model.json").write_text(json.dumps(meta, indent=2))


def load_model():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    meta = json.loads((MODELS_DIR / "best_model.json").read_text())
    return model, meta


def main():
    if not (PROCESSED_DIR / "train.csv").exists():
        print("No processed splits found - running data preparation first.")
        prepare()
    train, val, test = load_splits()

    results, fitted, _ = compare_models(train, val)
    print("\nValidation results (sorted by cost):")
    print(results.round(4).to_string(index=False))

    best = results.iloc[0]
    feature_cols = get_feature_columns(train)
    final = evaluate_on_test(fitted[best["model"]], best["best_threshold"], test, feature_cols)
    print(f"\nBest model: {best['model']} (threshold {best['best_threshold']:.3f})")
    print(f"Test cost: ${final['cost']:,.0f}  |  no-model cost: ${final['baselines']['no_model']:,.0f}"
          f"  |  saved {final['savings_vs_no_model']:.1%}")

    save_model(fitted[best["model"]], best["model"], float(best["best_threshold"]), feature_cols)
    print("Saved models/best_model.joblib")


if __name__ == "__main__":
    main()
