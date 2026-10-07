"""Dataset loaders producing (X, y) with consistent dtypes for federated GBDT.

Each loader returns a pandas DataFrame X (numeric-encoded features) and a
numpy array y (binary labels), plus a `feature_groups` dict used to split
features across parties for vertical FL.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_adult():
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    df = pd.read_csv(
        DATA_DIR / "adult" / "adult.data", names=cols, na_values=" ?",
        skipinitialspace=True,
    ).dropna()

    y = (df["income"].str.strip() == ">50K").astype(int).to_numpy()
    df = df.drop(columns=["income"])

    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    feature_groups = {
        # party 0: demographic features, party 1: employment/financial features
        "party0": ["age", "sex", "race", "marital_status", "relationship", "native_country"],
        "party1": ["workclass", "fnlwgt", "education", "education_num",
                   "occupation", "capital_gain", "capital_loss", "hours_per_week"],
    }
    return df.reset_index(drop=True), y, feature_groups


def load_heart_disease():
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target",
    ]
    df = pd.read_csv(
        DATA_DIR / "heart_disease" / "processed.cleveland.data",
        names=cols, na_values="?",
    ).dropna()
    df = df.astype(float)

    y = (df["target"] > 0).astype(int).to_numpy()
    df = df.drop(columns=["target"])

    feature_groups = {
        # party 0: patient demographics/history, party 1: clinical test results
        "party0": ["age", "sex", "cp", "exang", "fbs"],
        "party1": ["trestbps", "chol", "restecg", "thalach", "oldpeak", "slope", "ca", "thal"],
    }
    return df.reset_index(drop=True), y, feature_groups


def load_credit_default():
    df = pd.read_excel(
        DATA_DIR / "credit_default" / "default of credit card clients.xls",
        header=1, index_col=0,
    )
    df = df.rename(columns={"default payment next month": "target"})

    y = df["target"].astype(int).to_numpy()
    df = df.drop(columns=["target"])

    feature_groups = {
        # party 0: demographic/credit-limit features, party 1: payment/bill history
        "party0": ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"],
        "party1": [c for c in df.columns if c not in
                   {"LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"}],
    }
    return df.reset_index(drop=True), y, feature_groups


LOADERS = {
    "adult": load_adult,
    "heart_disease": load_heart_disease,
    "credit_default": load_credit_default,
}


if __name__ == "__main__":
    for name, fn in LOADERS.items():
        X, y, groups = fn()
        print(f"{name}: X={X.shape}, y positive rate={y.mean():.3f}, "
              f"party0 features={len(groups['party0'])}, party1 features={len(groups['party1'])}")
