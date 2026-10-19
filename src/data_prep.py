"""Loading, cleaning and splitting the raw credit card data.

Run as a script from the repo root:
    python -m src.data_prep
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_STATE, RAW_PATH, TARGET
from src.features import add_features


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (the raw file contains ~1,000 of them)."""
    return df.drop_duplicates().reset_index(drop=True)


def split(df: pd.DataFrame, val_size=0.2, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train / validation / test split.

    The split happens BEFORE any resampling (SMOTE) or scaling so that no
    information from validation/test rows leaks into training.
      - train: fit models
      - validation: compare models and choose the decision threshold
      - test: touched once, for the final honest estimate
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val[TARGET],
        random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train, val, test, out_dir=PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)


def load_splits(in_dir=PROCESSED_DIR):
    return (
        pd.read_csv(in_dir / "train.csv"),
        pd.read_csv(in_dir / "val.csv"),
        pd.read_csv(in_dir / "test.csv"),
    )


def prepare():
    """Raw CSV -> cleaned -> feature-engineered -> split -> saved to data/processed."""
    df = add_features(clean(load_raw()))
    train, val, test = split(df)
    save_splits(train, val, test)
    return train, val, test


if __name__ == "__main__":
    tr, va, te = prepare()
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"{name:5s}: {len(d):>7,} rows, {int(d[TARGET].sum()):>4} frauds "
              f"({d[TARGET].mean() * 100:.3f}%)")
