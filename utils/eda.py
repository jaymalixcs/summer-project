"""
utils/eda.py
============
Exploratory Data Analysis (EDA) functions.

Each function returns clean, JSON-serialisable Python objects so Flask
can pass them directly into Jinja2 templates.
"""

import pandas as pd
import numpy as np


# ─── 1. Statistical Summary ───────────────────────────────────────────────────
def get_describe(df: pd.DataFrame) -> list[dict]:
    """
    Return df.describe() transposed as a list of dicts for table rendering.

    Each dict has keys: feature, count, mean, std, min, 25%, 50%, 75%, max
    """
    desc = df.describe().T.round(2).reset_index()
    desc.columns = ["Feature"] + list(desc.columns[1:])
    return desc.to_dict(orient="records")


# ─── 2. Missing Values ────────────────────────────────────────────────────────
def get_missing(df: pd.DataFrame) -> list[dict]:
    """
    Return missing value count and percentage per column.

    Returns list of dicts:  [{"column": ..., "missing": ..., "pct": ...}, ...]
    """
    total = len(df)
    missing = df.isnull().sum()
    pct     = (missing / total * 100).round(2)
    return [
        {"column": col, "missing": int(missing[col]), "pct": float(pct[col])}
        for col in df.columns
    ]


# ─── 3. Data Types ────────────────────────────────────────────────────────────
def get_dtypes(df: pd.DataFrame) -> list[dict]:
    """Return column → dtype as a list of dicts for table rendering."""
    return [
        {"column": col, "dtype": str(df[col].dtype), "sample": str(df[col].iloc[0])}
        for col in df.columns
    ]


# ─── 4. Outlier Detection (IQR Method) ───────────────────────────────────────
def detect_outliers(df: pd.DataFrame, cols: list[str] = None) -> dict:
    """
    Use the Inter-Quartile Range (IQR) method to detect outliers.

    For each numeric column:
        IQR  = Q3 − Q1
        Lower fence = Q1 − 1.5 × IQR
        Upper fence = Q3 + 1.5 × IQR
        Any value outside [lower, upper] is an outlier.

    Returns:
        dict  {col: {"Q1": ..., "Q3": ..., "IQR": ...,
                     "lower": ..., "upper": ..., "outlier_count": ...}}
    """
    numeric = df.select_dtypes(include=[np.number])
    if cols:
        numeric = numeric[cols]

    result = {}
    for col in numeric.columns:
        Q1  = float(numeric[col].quantile(0.25))
        Q3  = float(numeric[col].quantile(0.75))
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        out_count = int(((numeric[col] < lower) | (numeric[col] > upper)).sum())
        result[col] = {
            "Q1"           : round(Q1, 2),
            "Q3"           : round(Q3, 2),
            "IQR"          : round(IQR, 2),
            "lower_fence"  : round(lower, 2),
            "upper_fence"  : round(upper, 2),
            "outlier_count": out_count,
        }
    return result


# ─── 5. Gender Distribution ───────────────────────────────────────────────────
def get_gender_dist(df: pd.DataFrame) -> dict:
    """Return gender counts as a dict: {"Male": N, "Female": M}."""
    if "Gender" in df.columns:
        counts = df["Gender"].value_counts().to_dict()
    else:
        counts = {}
    return counts


# ─── 6. Value Ranges ─────────────────────────────────────────────────────────
def get_value_ranges(df: pd.DataFrame) -> list[dict]:
    """Return min/max/mean for all numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        rows.append({
            "column" : col,
            "min"    : round(float(numeric[col].min()), 2),
            "max"    : round(float(numeric[col].max()), 2),
            "mean"   : round(float(numeric[col].mean()), 2),
            "median" : round(float(numeric[col].median()), 2),
            "std"    : round(float(numeric[col].std()), 2),
        })
    return rows


# ─── 7. Correlation Matrix ────────────────────────────────────────────────────
def get_correlation(df: pd.DataFrame) -> dict:
    """
    Return the Pearson correlation matrix as a nested dict.

    Example: {"Age": {"Age": 1.0, "Annual_Income": 0.03, ...}, ...}
    """
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().round(3)
    return corr.to_dict()


# ─── 8. Full EDA Summary ──────────────────────────────────────────────────────
def run_full_eda(df: pd.DataFrame) -> dict:
    """
    Run all EDA functions and return a unified dictionary for template rendering.
    """
    return {
        "describe"     : get_describe(df),
        "missing"      : get_missing(df),
        "dtypes"       : get_dtypes(df),
        "outliers"     : detect_outliers(df, ["Age", "Annual_Income", "Spending_Score"]),
        "gender_dist"  : get_gender_dist(df),
        "value_ranges" : get_value_ranges(df),
        "correlation"  : get_correlation(df),
        "shape"        : list(df.shape),
        "duplicates"   : int(df.duplicated().sum()),
        "head"         : df.head(10).to_dict(orient="records"),
    }
