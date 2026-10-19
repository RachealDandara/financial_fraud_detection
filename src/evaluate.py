"""Cost-based evaluation: accuracy is meaningless at 0.17% fraud, money isn't.

Cost model
----------
  missed fraud (false negative) -> we lose the transaction amount
  false alarm  (false positive) -> fixed cost `false_alarm_cost`
  correct decisions             -> cost 0
"""
import numpy as np
import pandas as pd

from src.config import FALSE_ALARM_COST


def total_cost(y_true, y_pred, amounts, false_alarm_cost=FALSE_ALARM_COST) -> float:
    y_true, y_pred, amounts = map(np.asarray, (y_true, y_pred, amounts))
    missed_fraud = ((y_true == 1) & (y_pred == 0))
    false_alarms = ((y_true == 0) & (y_pred == 1))
    return float(amounts[missed_fraud].sum() + false_alarm_cost * false_alarms.sum())


def baseline_costs(y_true, amounts, false_alarm_cost=FALSE_ALARM_COST) -> dict:
    """Two dumb strategies every model must beat."""
    y_true, amounts = np.asarray(y_true), np.asarray(amounts)
    return {
        "no_model": float(amounts[y_true == 1].sum()),                    # never flag anything
        "flag_everything": float(false_alarm_cost * (y_true == 0).sum()),  # flag every transaction
    }


def cost_curve(y_true, scores, amounts, false_alarm_cost=FALSE_ALARM_COST,
               thresholds=None) -> pd.DataFrame:
    """Total cost (plus precision/recall) at every candidate threshold."""
    y_true, scores, amounts = map(np.asarray, (y_true, scores, amounts))
    if thresholds is None:
        thresholds = np.arange(0.005, 1.0, 0.005)
    rows = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        rows.append({
            "threshold": float(t),
            "cost": total_cost(y_true, pred, amounts, false_alarm_cost),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "flagged": tp + fp,
        })
    return pd.DataFrame(rows)


def best_threshold(y_true, scores, amounts, false_alarm_cost=FALSE_ALARM_COST):
    """Threshold with the lowest total cost -> (threshold, cost)."""
    curve = cost_curve(y_true, scores, amounts, false_alarm_cost)
    best = curve.loc[curve["cost"].idxmin()]
    return float(best["threshold"]), float(best["cost"])


def savings_vs_no_model(cost, y_true, amounts, false_alarm_cost=FALSE_ALARM_COST) -> float:
    """Fraction of the 'do nothing' fraud loss that the model saves."""
    base = baseline_costs(y_true, amounts, false_alarm_cost)["no_model"]
    return 1 - cost / base if base else 0.0
