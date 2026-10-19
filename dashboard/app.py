"""Interactive cost dashboard.

Run from the repo root (after running notebooks 02 and 03, or `python -m src.model`):
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.data_prep import load_splits
from src.evaluate import baseline_costs, cost_curve, total_cost
from src.model import load_model, split_xy

st.set_page_config(page_title="Fraud Detection - Cost Dashboard", layout="wide")


@st.cache_resource
def load_everything():
    model, meta = load_model()
    _, _, test = load_splits()
    X, y, amounts = split_xy(test, meta["feature_columns"])
    scores = model.predict_proba(X)[:, 1]
    return meta, y.to_numpy(), amounts.to_numpy(), scores


try:
    meta, y, amounts, scores = load_everything()
except FileNotFoundError:
    st.error("No trained model or processed data found. Run the notebooks "
             "02_cleaning and 03_modeling (or `python -m src.model`) first.")
    st.stop()

st.title("Credit Card Fraud Detection - Business Cost Explorer")
st.caption(f"Model: **{meta['model_name']}** | evaluated on the held-out test set "
           f"({len(y):,} transactions, {int(y.sum())} frauds)")

st.sidebar.header("Business assumptions")
fac = st.sidebar.slider("Cost of a false alarm", 1.0, 200.0, float(meta["false_alarm_cost"]), 1.0,
                        help="Support time / customer friction when a legitimate transaction is flagged.")
curve = cost_curve(y, scores, amounts, fac)
optimal = float(curve.loc[curve["cost"].idxmin(), "threshold"])
threshold = st.sidebar.slider("Decision threshold", 0.005, 0.995, optimal, 0.005,
                              help="Transactions with a fraud score at or above this are flagged.")
st.sidebar.caption(f"Cost-optimal threshold for this false-alarm cost: **{optimal:.3f}**  \n"
                   f"(computed on the test data for exploration; the saved model uses "
                   f"**{meta['threshold']:.3f}**, chosen on the validation set)")

pred = (scores >= threshold).astype(int)
cost = total_cost(y, pred, amounts, fac)
base = baseline_costs(y, amounts, fac)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total cost with model", f"{cost:,.0f}")
c2.metric("Cost doing nothing", f"{base['no_model']:,.0f}")
c3.metric("Loss avoided", f"{(1 - cost / base['no_model']):.1%}" if base["no_model"] else "n/a")
c4.metric("Frauds caught", f"{tp} / {tp + fn}")

left, right = st.columns([2, 1])
with left:
    st.subheader("Total cost vs decision threshold")
    chart = curve.set_index("threshold")[["cost"]].rename(columns={"cost": "total cost"})
    st.line_chart(chart)
    st.caption("The cost-optimal threshold shifts as the false-alarm cost changes - drag the sidebar sliders.")
with right:
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["actually legitimate", "actually fraud"],
        columns=["predicted legitimate", "predicted fraud"],
    ))
    st.write(f"Missed-fraud loss: **{amounts[(y == 1) & (pred == 0)].sum():,.0f}**")
    st.write(f"False-alarm cost: **{fp * fac:,.0f}** ({fp:,} alarms)")

st.subheader("Highest-risk transactions in the test set")
top = pd.DataFrame({"fraud_score": scores, "amount": amounts, "is_fraud": y}).sort_values(
    "fraud_score", ascending=False).head(15)
st.dataframe(top.style.format({"fraud_score": "{:.3f}", "amount": "{:,.2f}"}))
