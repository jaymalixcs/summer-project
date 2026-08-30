"""
utils/preprocessing.py
======================
Data loading, cleaning, and feature scaling utilities.

Every function is documented step-by-step so the code can be used directly
in project documentation.

NOTE ON SIMPLICITY: earlier versions of this file saved the fitted scaler
to a .pkl file on disk with joblib, so the /predict route could reload it
later. That added complexity for no real benefit — fitting a StandardScaler
on 200 rows takes a fraction of a millisecond, and serverless hosts like
Vercel have a READ-ONLY filesystem (except /tmp), so writing model files
next to the code breaks in production. Instead, we simply keep the fitted
scaler object in memory (inside the pipeline dict) and reuse it directly.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Mall_Customers.csv")


# ─── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load the raw Mall Customers CSV file into a Pandas DataFrame.

    Steps:
        1. Build the absolute path to the CSV file.
        2. Read the file using pd.read_csv().
        3. Return the raw DataFrame unchanged.

    Returns:
        pd.DataFrame: Raw dataset with 200 rows and 5 columns.
    """
    df = pd.read_csv(DATA_PATH)          # Read CSV into DataFrame
    return df                             # Return the raw data


# ─── 2. Inspect Data ──────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform an initial inspection of the dataset and return a summary dict.

    Returns a dictionary containing:
        - shape        : (rows, cols) tuple
        - columns      : list of column names
        - dtypes       : column → dtype mapping
        - null_counts  : column → null count mapping
        - duplicates   : number of duplicate rows
        - head         : first 5 rows as list of dicts (JSON-serialisable)
        - tail         : last 5 rows as list of dicts
        - describe     : statistical summary as list of dicts
    """
    return {
        "shape"       : list(df.shape),                           # (rows, cols)
        "columns"     : df.columns.tolist(),                      # column names
        "dtypes"      : df.dtypes.astype(str).to_dict(),          # dtype per col
        "null_counts" : df.isnull().sum().to_dict(),              # nulls per col
        "duplicates"  : int(df.duplicated().sum()),               # dup row count
        "head"        : df.head(5).to_dict(orient="records"),     # first 5 rows
        "tail"        : df.tail(5).to_dict(orient="records"),     # last 5 rows
        "describe"    : df.describe().round(2).to_dict(),         # stats summary
    }


# ─── 3. Clean Data ────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw DataFrame and return the cleaned DataFrame plus a log.

    Cleaning steps:
        1. Drop duplicate rows.
        2. Drop the 'CustomerID' column (not useful for clustering).
        3. Encode 'Gender' as a numeric binary column (Male=1, Female=0).
        4. Rename columns for brevity and consistency.
        5. Verify no null values remain.
        6. Verify data types are correct.

    Returns:
        (cleaned_df, cleaning_log)
    """
    log = {}                                  # Track what was done

    # Step 1 — Remove duplicates
    before = len(df)
    df = df.drop_duplicates()                 # Remove exact duplicate rows
    log["duplicates_removed"] = before - len(df)

    # Step 2 — Drop CustomerID (identifier, not a feature)
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
        log["dropped_columns"] = ["CustomerID"]

    # Step 3 — Rename columns for cleaner code
    df = df.rename(columns={
        "Annual Income (k$)" : "Annual_Income",
        "Spending Score (1-100)": "Spending_Score",
    })
    log["renamed_columns"] = {
        "Annual Income (k$)"       : "Annual_Income",
        "Spending Score (1-100)"   : "Spending_Score",
    }

    # Step 4 — Encode Gender  (Male → 1, Female → 0)
    df["Gender_Encoded"] = df["Gender"].map({"Male": 1, "Female": 0})
    log["gender_encoded"] = "Male=1, Female=0"

    # Step 5 — Confirm nulls
    log["null_after_cleaning"] = df.isnull().sum().to_dict()

    # Step 6 — Confirm dtypes
    log["dtypes_after_cleaning"] = df.dtypes.astype(str).to_dict()

    return df, log


# ─── 4. Scale Features ────────────────────────────────────────────────────────
FEATURE_COLS = ["Annual_Income", "Spending_Score"]   # Features used for K-Means

def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Apply StandardScaler to the clustering features.

    Why scaling?
        K-Means uses Euclidean distance. If features have different ranges
        (e.g. Income: 15–137 vs. SpendingScore: 1–99), the larger-range
        feature dominates the distance calculation and biases the clusters.
        StandardScaler transforms each feature to mean=0, std=1.

    Steps:
        1. Select FEATURE_COLS from the cleaned DataFrame.
        2. Instantiate StandardScaler.
        3. Fit the scaler on the selected data.
        4. Transform the data into a scaled NumPy array.

    The fitted `scaler` object is returned (not saved to disk) so that the
    /predict route can reuse the exact same fitted scaler later, straight
    from memory, via the cached pipeline dict.

    Returns:
        (X_scaled, scaler)
    """
    X = df[FEATURE_COLS].values               # Extract feature matrix (numpy)

    scaler = StandardScaler()                 # Instantiate the scaler
    X_scaled = scaler.fit_transform(X)        # Fit and transform in one step

    return X_scaled, scaler


# ─── 5. Full Pipeline (convenience wrapper) ───────────────────────────────────
def run_pipeline() -> dict:
    """
    Execute the complete preprocessing pipeline and return all artefacts.

    Returns a dict with keys:
        raw_df, cleaned_df, cleaning_log, X_scaled, scaler, inspect_info
    """
    raw_df              = load_data()
    inspect_info        = inspect_data(raw_df)
    cleaned_df, log     = clean_data(raw_df)
    X_scaled, scaler    = scale_features(cleaned_df)

    return {
        "raw_df"       : raw_df,
        "cleaned_df"   : cleaned_df,
        "cleaning_log" : log,
        "X_scaled"     : X_scaled,
        "scaler"       : scaler,
        "inspect_info" : inspect_info,
    }
