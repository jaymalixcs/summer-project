"""
app.py — Customer Segmentation Dashboard
=========================================
Main Flask application. All routes are defined here.

Run:
    python app.py

Then open: http://127.0.0.1:5000
"""

import os
import json
from flask import (Flask, render_template, request,
                   redirect, url_for, jsonify, flash)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = "segmentiq-secret-2024"

# ─── Global pipeline cache ────────────────────────────────────────────────────
# Run once on startup so every page loads fast.
_cache = {}

def get_pipeline():
    """Return cached pipeline data; run it if not yet computed."""
    if "pipeline" not in _cache:
        from utils.preprocessing import run_pipeline
        _cache["pipeline"] = run_pipeline()
    return _cache["pipeline"]

def get_elbow():
    if "elbow" not in _cache:
        from utils.clustering import compute_elbow
        pipe = get_pipeline()
        _cache["elbow"] = compute_elbow(pipe["X_scaled"])
    return _cache["elbow"]

def get_silhouette():
    if "silhouette" not in _cache:
        from utils.clustering import compute_silhouette
        pipe = get_pipeline()
        _cache["silhouette"] = compute_silhouette(pipe["X_scaled"])
    return _cache["silhouette"]

def get_kmeans():
    if "kmeans" not in _cache:
        from utils.clustering import fit_kmeans
        import joblib
        pipe = get_pipeline()
        kmeans_data = fit_kmeans(pipe["X_scaled"])
        
        # Load the scaler to inverse transform centroids (use absolute path)
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
        scaler = joblib.load(scaler_path)
        centroids_orig = scaler.inverse_transform(kmeans_data["centroids"])
        kmeans_data["centroids_orig"] = centroids_orig.tolist()
        
        _cache["kmeans"] = kmeans_data
    return _cache["kmeans"]

def get_cluster_summary():
    if "cluster_summary" not in _cache:
        from utils.clustering import get_cluster_summary
        pipe   = get_pipeline()
        kmeans = get_kmeans()
        _cache["cluster_summary"] = get_cluster_summary(
            pipe["cleaned_df"], kmeans["labels"]
        )
    return _cache["cluster_summary"]


# ─── Initialise DB ────────────────────────────────────────────────────────────
from utils.database import init_db
init_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Home ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    pipe    = get_pipeline()
    summary = get_cluster_summary()
    return render_template("home.html",
                           total_customers=pipe["inspect_info"]["shape"][0],
                           n_clusters=5,
                           n_features=3,
                           silhouette=round(get_kmeans()["silhouette"], 3))


# ── 2. Dataset Overview ───────────────────────────────────────────────────────
@app.route("/dataset")
def dataset():
    pipe  = get_pipeline()
    info  = pipe["inspect_info"]
    return render_template("dataset.html",
                           info=info,
                           head=info["head"],
                           tail=info["tail"])


# ── 3. EDA Workflow ───────────────────────────────────────────────────────────
@app.route("/eda")
def eda():
    from utils.eda import run_full_eda
    pipe     = get_pipeline()
    eda_data = run_full_eda(pipe["cleaned_df"])
    return render_template("eda.html", eda=eda_data)


# ── 4. Data Cleaning ──────────────────────────────────────────────────────────
@app.route("/cleaning")
def cleaning():
    pipe = get_pipeline()
    return render_template("cleaning.html",
                           log=pipe["cleaning_log"],
                           head=pipe["cleaned_df"].head(10).to_dict(orient="records"),
                           columns=pipe["cleaned_df"].columns.tolist(),
                           shape=list(pipe["cleaned_df"].shape))


# ── 5. Feature Scaling ────────────────────────────────────────────────────────
@app.route("/scaling")
def scaling():
    pipe = get_pipeline()
    df   = pipe["cleaned_df"]
    # Show before/after scaling values for first 8 rows
    raw_vals    = df[["Annual_Income", "Spending_Score"]].head(8).values.tolist()
    scaled_vals = pipe["X_scaled"][:8].tolist()
    return render_template("scaling.html",
                           raw_vals=raw_vals,
                           scaled_vals=[[round(v, 4) for v in row] for row in scaled_vals],
                           feature_cols=["Annual_Income", "Spending_Score"])


# ── 6. EDA Visualizations ─────────────────────────────────────────────────────
@app.route("/visualizations")
def visualizations():
    from utils import visualization as viz
    pipe = get_pipeline()
    df   = pipe["cleaned_df"]
    charts = {
        "age_hist"       : viz.plot_age_hist(df),
        "income_hist"    : viz.plot_income_hist(df),
        "spending_hist"  : viz.plot_spending_hist(df),
        "gender_pie"     : viz.plot_gender_pie(pipe["raw_df"]),
        "boxplots"       : viz.plot_boxplots(df),
        "age_vs_spending": viz.plot_age_vs_spending(df),
        "income_vs_spending": viz.plot_income_vs_spending(df),
        "heatmap"        : viz.plot_correlation_heatmap(df),
    }
    return render_template("visualizations.html", charts=charts)


# ── 7. Elbow Method ───────────────────────────────────────────────────────────
@app.route("/elbow")
def elbow():
    from utils.visualization import plot_elbow
    elbow_data  = get_elbow()
    chart_json  = plot_elbow(elbow_data)
    return render_template("elbow.html",
                           elbow_data=elbow_data,
                           chart_json=chart_json)


# ── 8. Silhouette Score ───────────────────────────────────────────────────────
@app.route("/silhouette")
def silhouette():
    return redirect(url_for("clustering") + "#silhouette-analysis")


# ── 9. K-Means Clustering ────────────────────────────────────────────────────
@app.route("/clustering")
def clustering():
    from utils.visualization import plot_silhouette
    kmeans  = get_kmeans()
    summary = get_cluster_summary()
    sil_data = get_silhouette()
    sil_chart_json = plot_silhouette(sil_data)
    return render_template("clustering.html",
                           kmeans=kmeans,
                           summary=summary,
                           sil_data=sil_data,
                           sil_chart_json=sil_chart_json)


# ── 10. Cluster Visualization ────────────────────────────────────────────────
@app.route("/cluster-visualization")
def cluster_visualization():
    from utils.visualization import plot_clusters, plot_cluster_sizes, plot_cluster_radar
    pipe    = get_pipeline()
    kmeans  = get_kmeans()
    summary = get_cluster_summary()

    # Centroids are in scaled space — convert back to original for display
    scaler = pipe["scaler"]
    centroids_orig = scaler.inverse_transform(kmeans["centroids"]).tolist()

    charts = {
        "clusters"      : plot_clusters(pipe["cleaned_df"], kmeans["labels"], centroids_orig),
        "cluster_sizes" : plot_cluster_sizes(summary),
        "radar"         : plot_cluster_radar(summary),
    }
    return render_template("cluster_viz.html",
                           charts=charts,
                           summary=summary,
                           centroids=centroids_orig)


# ── 11. Cluster Interpretation ───────────────────────────────────────────────
@app.route("/interpretation")
def interpretation():
    from utils.recommendations import get_all_profiles
    summary  = get_cluster_summary()
    profiles = get_all_profiles()
    return render_template("interpretation.html",
                           summary=summary,
                           profiles=profiles)


# ── 12. Marketing Recommendations ────────────────────────────────────────────
@app.route("/recommendations")
def recommendations():
    from utils.recommendations import get_all_profiles, get_all_recommendations
    profiles = get_all_profiles()
    recs     = get_all_recommendations()
    return render_template("recommendations.html",
                           profiles=profiles,
                           recs=recs)


# ── 13. Dashboard ─────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    from utils.visualization import plot_cluster_sizes, plot_cluster_radar
    from utils.database import get_all_customers, get_total_count
    pipe    = get_pipeline()
    kmeans  = get_kmeans()
    summary = get_cluster_summary()
    charts  = {
        "cluster_sizes": plot_cluster_sizes(summary),
        "radar"        : plot_cluster_radar(summary),
    }
    db_count = get_total_count()
    return render_template("dashboard.html",
                           summary=summary,
                           kmeans=kmeans,
                           charts=charts,
                           total=pipe["inspect_info"]["shape"][0],
                           db_count=db_count)


# ── 14. Predict New Customer ──────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    from utils.database import save_customer, get_all_customers
    from utils.recommendations import get_all_profiles
    from utils.clustering import predict_cluster

    result   = None
    profiles = get_all_profiles()
    customers = get_all_customers()

    if request.method == "POST":
        try:
            name           = request.form.get("name", "Anonymous").strip() or "Anonymous"
            gender         = request.form.get("gender", "Male")
            age            = int(request.form.get("age", 30))
            annual_income  = float(request.form.get("annual_income", 60))
            spending_score = int(request.form.get("spending_score", 50))

            cluster_id = predict_cluster(annual_income, spending_score)
            db_id      = save_customer(name, gender, age, annual_income,
                                       spending_score, cluster_id)
            customers  = get_all_customers()  # refresh

            result = {
                "name"          : name,
                "gender"        : gender,
                "age"           : age,
                "annual_income" : annual_income,
                "spending_score": spending_score,
                "cluster_id"    : cluster_id,
                "profile"       : profiles.get(cluster_id, {}),
                "db_id"         : db_id,
            }
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template("predict.html",
                           result=result,
                           profiles=profiles,
                           customers=customers)


# ── DELETE Customer (AJAX) ────────────────────────────────────────────────────
@app.route("/delete-customer/<int:cid>", methods=["POST"])
def delete_customer(cid):
    from utils.database import delete_customer as _del
    _del(cid)
    return jsonify({"status": "ok"})


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-warm the pipeline cache so first page load is instant
    print("[*] Pre-computing pipeline (EDA + Clustering)...")
    get_pipeline()
    get_elbow()
    get_silhouette()
    get_kmeans()
    get_cluster_summary()
    print("[OK] Pipeline ready! Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
