"""
utils/clustering.py
===================
K-Means clustering, Elbow Method, and Silhouette Score utilities.

All functions return JSON-serialisable Python objects suitable for
direct use in Flask routes and Jinja2 templates.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── Optimal K (determined by Elbow + Silhouette analysis) ───────────────────
OPTIMAL_K = 5


# ─── 1. Elbow Method ──────────────────────────────────────────────────────────
def compute_elbow(X_scaled: np.ndarray, k_range: range = range(1, 11)) -> dict:
    """
    Calculate the Within-Cluster Sum of Squares (WCSS) for k = 1 … 10.

    How it works:
        For each value of k:
            1. Fit a KMeans model with k clusters.
            2. Record the model's inertia_ (= WCSS).
        The WCSS decreases as k increases.
        The 'elbow' point — where the rate of decrease slows — is the
        optimal number of clusters.

    Args:
        X_scaled  : Scaled feature matrix (numpy array).
        k_range   : Range of k values to test (default 1–10).

    Returns:
        dict with keys "k_values" and "wcss" (both lists).
    """
    wcss     = []                                   # Store WCSS for each k
    k_values = list(k_range)                        # [1, 2, 3, …, 10]

    for k in k_values:
        km = KMeans(
            n_clusters    = k,
            init          = "k-means++",            # Smart centroid initialisation
            n_init        = 10,                     # Run 10 times, pick best
            random_state  = 42,
            max_iter      = 300,
        )
        km.fit(X_scaled)                            # Fit on scaled features
        wcss.append(float(km.inertia_))            # inertia_ = WCSS

    return {"k_values": k_values, "wcss": [round(w, 4) for w in wcss]}


# ─── 2. Silhouette Scores ─────────────────────────────────────────────────────
def compute_silhouette(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Compute the Silhouette Score for k = 2 … 10.

    How it works:
        For each k:
            1. Fit KMeans and get cluster labels.
            2. Silhouette score = (b − a) / max(a, b)
               where a = mean intra-cluster distance,
                     b = mean nearest-cluster distance.
            3. Score ranges from -1 (bad) to +1 (perfect).
        The k with the highest silhouette score is the best.

    Returns:
        dict with keys "k_values" and "scores" (both lists).
    """
    scores   = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=42, max_iter=300)
        labels = km.fit_predict(X_scaled)           # Cluster labels for each row
        score  = silhouette_score(X_scaled, labels) # Compute silhouette
        scores.append(round(float(score), 4))

    return {"k_values": k_values, "scores": scores}


# ─── 3. Fit Final K-Means ─────────────────────────────────────────────────────
def fit_kmeans(X_scaled: np.ndarray, k: int = OPTIMAL_K) -> dict:
    """
    Fit the final K-Means model with k = OPTIMAL_K and save it to disk.

    Steps:
        1. Instantiate KMeans with k clusters and k-means++ initialisation.
        2. Fit the model on the scaled feature matrix.
        3. Extract cluster labels (0 … k-1) for every customer.
        4. Extract cluster centroids (in scaled space).
        5. Compute the silhouette score of the final model.
        6. Save the fitted model as models/kmeans_model.pkl.

    Returns:
        dict: {labels, centroids, inertia, silhouette, k}
    """
    km = KMeans(
        n_clusters   = k,
        init         = "k-means++",
        n_init       = 10,
        random_state = 42,
        max_iter     = 300,
    )
    km.fit(X_scaled)                                # Train the model

    labels     = km.labels_.tolist()               # Cluster ID for each customer
    centroids  = km.cluster_centers_.tolist()      # Centroid coordinates
    inertia    = round(float(km.inertia_), 4)      # Final WCSS
    sil_score  = round(float(silhouette_score(X_scaled, km.labels_)), 4)

    # Persist model so /predict route can reuse it
    os.makedirs(MODEL_DIR, exist_ok=True)
    # joblib.dump(km, os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    return {
        "labels"     : labels,
        "centroids"  : centroids,
        "inertia"    : inertia,
        "silhouette" : sil_score,
        "k"          : k,
    }


# ─── 4. Cluster Summary Stats ────────────────────────────────────────────────
def get_cluster_summary(df: pd.DataFrame, labels: list) -> list[dict]:
    """
    Attach cluster labels to the cleaned DataFrame and compute per-cluster stats.

    Returns:
        list of dicts, one per cluster:
        [{cluster_id, size, pct, mean_age, mean_income, mean_spending, gender_M, gender_F}, ...]
    """
    df = df.copy()
    df["Cluster"] = labels                          # Add cluster column

    total    = len(df)
    summary  = []

    for cid in sorted(df["Cluster"].unique()):
        grp = df[df["Cluster"] == cid]

        gender_counts = grp["Gender"].value_counts().to_dict() if "Gender" in df.columns else {}
        summary.append({
            "cluster_id"    : int(cid),
            "size"          : int(len(grp)),
            "pct"           : round(len(grp) / total * 100, 1),
            "mean_age"      : round(float(grp["Age"].mean()), 1),
            "mean_income"   : round(float(grp["Annual_Income"].mean()), 1),
            "mean_spending" : round(float(grp["Spending_Score"].mean()), 1),
            "gender_M"      : int(gender_counts.get("Male", 0)),
            "gender_F"      : int(gender_counts.get("Female", 0)),
        })

    return summary


# ─── 5. Predict Cluster for New Customer ─────────────────────────────────────
def predict_cluster(annual_income: float, spending_score: float) -> int:
    """
    Predict the cluster for a new customer given their income and spending score.

    Loads the saved scaler and KMeans model from disk, scales the input,
    and returns the cluster ID (0-indexed integer).
    """
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    model_path  = os.path.join(MODEL_DIR, "kmeans_model.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(model_path)):
        # Models not trained yet — run pipeline first
        from utils.preprocessing import run_pipeline
        from utils.clustering    import fit_kmeans as _fit
        pipe    = run_pipeline()
        _fit(pipe["X_scaled"])

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    X_new    = np.array([[annual_income, spending_score]])
    X_scaled = scaler.transform(X_new)
    cluster  = int(model.predict(X_scaled)[0])
    return cluster
