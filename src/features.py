"""Feature engineering. Everything here is row-wise, so it cannot leak
information between train and test."""
import numpy as np
import pandas as pd

from src.config import TARGET


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # `Time` = seconds since the first transaction in the 2-day window.
    # The absolute clock time is unknown, so `hour` is hour-of-day *relative
    # to the start of the data*. It still captures day/night style cycles.
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    df["is_zero_amount"] = (df["Amount"] == 0).astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Model inputs. Raw `Time` is dropped: it only identifies *when in this
    particular 2-day window* a transaction happened, which would not generalize
    to new data (`hour` carries the useful cyclical part)."""
    return [c for c in df.columns if c not in (TARGET, "Time")]
