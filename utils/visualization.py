"""
utils/visualization.py
=======================
All Plotly chart generation functions.

Each function returns a Plotly figure as a JSON string that can be
passed to Jinja2 and rendered client-side with plotly.js.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Global layout defaults ───────────────────────────────────────────────────
LAYOUT_DEFAULTS = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(15,15,40,0.6)",
    font          = dict(family="Inter, sans-serif", color="#E2E8F0", size=12),
    margin        = dict(l=40, r=20, t=50, b=40),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0")),
    xaxis         = dict(gridcolor="rgba(255,255,255,0.07)", color="#9CA3AF"),
    yaxis         = dict(gridcolor="rgba(255,255,255,0.07)", color="#9CA3AF"),
)

# 5-cluster colour palette
CLUSTER_COLORS = ["#6C63FF", "#00D4AA", "#FF6B6B", "#FFD166", "#06D6A0"]


def _to_rgba(color: str, alpha: float) -> str:
    """Convert hex/rgb/rgba strings to a valid rgba() string for Plotly."""
    color = color.strip()
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")
    if color.startswith("rgba("):
        return color
    return f"rgba(108,99,255,{alpha})"


def _fig_to_json(fig) -> str:
    """Convert a Plotly figure to a JSON string for template embedding."""
    return fig.to_json()


import plotly


# ─── 1. Age Distribution Histogram ───────────────────────────────────────────
def plot_age_hist(df: pd.DataFrame) -> str:
    fig = px.histogram(
        df, x="Age", nbins=20,
        title="Age Distribution",
        color_discrete_sequence=["#6C63FF"],
    )
    fig.update_layout(**LAYOUT_DEFAULTS, title_font_color="#A78BFA")
    fig.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1)
    return _fig_to_json(fig)


# ─── 2. Annual Income Histogram ──────────────────────────────────────────────
def plot_income_hist(df: pd.DataFrame) -> str:
    fig = px.histogram(
        df, x="Annual_Income", nbins=20,
        title="Annual Income Distribution (k$)",
        color_discrete_sequence=["#00D4AA"],
    )
    fig.update_layout(**LAYOUT_DEFAULTS, title_font_color="#34D399")
    fig.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1)
    return _fig_to_json(fig)


# ─── 3. Spending Score Histogram ─────────────────────────────────────────────
def plot_spending_hist(df: pd.DataFrame) -> str:
    fig = px.histogram(
        df, x="Spending_Score", nbins=20,
        title="Spending Score Distribution (1–100)",
        color_discrete_sequence=["#FF6B6B"],
    )
    fig.update_layout(**LAYOUT_DEFAULTS, title_font_color="#F87171")
    fig.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1)
    return _fig_to_json(fig)


# ─── 4. Gender Distribution Pie ──────────────────────────────────────────────
def plot_gender_pie(df: pd.DataFrame) -> str:
    gender_counts = df["Gender"].value_counts()
    fig = go.Figure(go.Pie(
        labels=gender_counts.index.tolist(),
        values=gender_counts.values.tolist(),
        marker=dict(colors=["#6C63FF", "#FF6B6B"], line=dict(color="#0A0B1A", width=2)),
        hole=0.45,
        textfont=dict(color="#E2E8F0"),
    ))
    fig.update_layout(
        title="Gender Distribution",
        title_font_color="#A78BFA",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
    )
    return _fig_to_json(fig)


# ─── 5. Box Plots for Outliers ────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> str:
    cols   = ["Age", "Annual_Income", "Spending_Score"]
    colors = ["#6C63FF", "#00D4AA", "#FF6B6B"]
    fig = go.Figure()
    for col, color in zip(cols, colors):
        fig.add_trace(go.Box(
            y=df[col], name=col,
            marker_color=color,
            line=dict(color=color),
            fillcolor=_to_rgba(color, 0.2),
        ))
    fig.update_layout(
        title="Boxplots — Outlier Detection",
        title_font_color="#A78BFA",
        **LAYOUT_DEFAULTS,
    )
    return _fig_to_json(fig)


# ─── 6. Age vs Spending Score Scatter ────────────────────────────────────────
def plot_age_vs_spending(df: pd.DataFrame) -> str:
    fig = px.scatter(
        df, x="Age", y="Spending_Score", color="Gender",
        title="Age vs Spending Score",
        color_discrete_map={"Male": "#6C63FF", "Female": "#FF6B6B"},
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#0A0B1A")))
    fig.update_layout(**LAYOUT_DEFAULTS, title_font_color="#A78BFA")
    return _fig_to_json(fig)


# ─── 7. Income vs Spending Score Scatter ─────────────────────────────────────
def plot_income_vs_spending(df: pd.DataFrame) -> str:
    fig = px.scatter(
        df, x="Annual_Income", y="Spending_Score", color="Gender",
        title="Annual Income vs Spending Score",
        color_discrete_map={"Male": "#6C63FF", "Female": "#FF6B6B"},
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#0A0B1A")))
    fig.update_layout(**LAYOUT_DEFAULTS, title_font_color="#A78BFA")
    return _fig_to_json(fig)


# ─── 8. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=[np.number])
    corr    = numeric.corr().round(3)
    cols    = corr.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=corr.values.tolist(),
        x=cols, y=cols,
        colorscale="Viridis",
        text=corr.round(2).values.tolist(),
        texttemplate="%{text}",
        textfont=dict(color="#E2E8F0", size=11),
        colorbar=dict(tickfont=dict(color="#E2E8F0")),
    ))
    fig.update_layout(
        title="Correlation Heatmap",
        title_font_color="#A78BFA",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(color="#9CA3AF"),
        yaxis=dict(color="#9CA3AF", autorange="reversed"),
    )
    return _fig_to_json(fig)


# ─── 9. Elbow Method Plot ─────────────────────────────────────────────────────
def plot_elbow(elbow_data: dict) -> str:
    k_vals = elbow_data["k_values"]
    wcss   = elbow_data["wcss"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_vals, y=wcss,
        mode="lines+markers",
        line=dict(color="#6C63FF", width=3),
        marker=dict(color="#FF6B6B", size=10, symbol="circle",
                    line=dict(color="#E2E8F0", width=1.5)),
        name="WCSS",
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.12)",
    ))
    # Highlight elbow at k=5
    fig.add_vline(x=5, line=dict(color="#FFD166", dash="dash", width=2))
    fig.add_annotation(x=5, y=max(wcss)*0.5, text="Optimal K=5",
                       font=dict(color="#FFD166", size=13),
                       showarrow=True, arrowcolor="#FFD166")
    fig.update_layout(
        title="Elbow Method — WCSS vs Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="WCSS (Inertia)",
        title_font_color="#A78BFA",
        **LAYOUT_DEFAULTS,
    )
    return _fig_to_json(fig)


# ─── 10. Silhouette Score Plot ────────────────────────────────────────────────
def plot_silhouette(sil_data: dict) -> str:
    k_vals = sil_data["k_values"]
    scores = sil_data["scores"]
    best_k = k_vals[scores.index(max(scores))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_vals, y=scores,
        marker=dict(
            color=scores,
            colorscale="Viridis",
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        name="Silhouette Score",
    ))
    fig.add_vline(x=best_k, line=dict(color="#FFD166", dash="dash", width=2))
    fig.update_layout(
        title="Silhouette Score vs Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        title_font_color="#A78BFA",
        **LAYOUT_DEFAULTS,
    )
    return _fig_to_json(fig)


# ─── 11. Cluster Scatter (main visualisation) ─────────────────────────────────
def plot_clusters(df: pd.DataFrame, labels: list, centroids: list = None) -> str:
    df = df.copy()
    df["Cluster"] = [str(l) for l in labels]

    cluster_names = {
        "0": "Careful Spenders",
        "1": "Budget Shoppers",
        "2": "Impulsive Buyers",
        "3": "Target Customers",
        "4": "Average Customers",
    }

    color_map = {str(i): CLUSTER_COLORS[i] for i in range(5)}

    fig = px.scatter(
        df, x="Annual_Income", y="Spending_Score",
        color="Cluster",
        color_discrete_map=color_map,
        title="Customer Clusters — Income vs Spending Score",
        hover_data={"Age": True, "Gender": True, "Annual_Income": True, "Spending_Score": True},
        opacity=0.85,
        labels={"Annual_Income": "Annual Income (k$)", "Spending_Score": "Spending Score"},
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.8, color="#0A0B1A")))

    # Add centroids
    if centroids:
        cx = [c[0] for c in centroids]
        cy = [c[1] for c in centroids]
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="markers",
            marker=dict(symbol="x", size=18, color="#FFFFFF",
                        line=dict(color="#0A0B1A", width=2)),
            name="Centroids",
        ))

    fig.update_layout(**LAYOUT_DEFAULTS, title_font_color="#A78BFA")

    # Rename legend entries
    for trace in fig.data:
        if trace.name in cluster_names:
            trace.name = f"Cluster {trace.name}: {cluster_names[trace.name]}"

    return _fig_to_json(fig)


# ─── 12. Cluster Size Bar Chart ───────────────────────────────────────────────
def plot_cluster_sizes(summary: list) -> str:
    names = [f"C{s['cluster_id']}" for s in summary]
    sizes = [s["size"] for s in summary]

    fig = go.Figure(go.Bar(
        x=names, y=sizes,
        marker=dict(color=CLUSTER_COLORS[:len(summary)],
                    line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=sizes, textposition="outside",
        textfont=dict(color="#E2E8F0"),
    ))
    fig.update_layout(
        title="Customer Count per Cluster",
        xaxis_title="Cluster", yaxis_title="Customer Count",
        title_font_color="#A78BFA",
        **LAYOUT_DEFAULTS,
    )
    return _fig_to_json(fig)


# ─── 13. Cluster Radar Chart (mean features per cluster) ──────────────────────
def plot_cluster_radar(summary: list) -> str:
    categories = ["Age", "Annual Income", "Spending Score"]

    # Proper RGBA colors (hex converted to RGB integers)
    RADAR_FILL_COLORS = [
        "rgba(108,99,255,0.15)",   # violet
        "rgba(255,138,101,0.15)",  # orange
        "rgba(206,147,216,0.15)",  # purple
        "rgba(129,199,132,0.15)",  # green
        "rgba(255,213,79,0.15)",   # yellow
    ]

    fig = go.Figure()
    for s in summary:
        cid = s["cluster_id"]
        # Normalise to 0–1 for radar
        vals = [
            s["mean_age"]      / 70,    # max age ~70
            s["mean_income"]   / 140,   # max income ~140
            s["mean_spending"] / 100,   # max score 100
        ]
        vals += [vals[0]]  # close the radar
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Cluster {cid}",
            marker=dict(color=CLUSTER_COLORS[cid]),
            line=dict(color=CLUSTER_COLORS[cid], width=2),
            fillcolor=RADAR_FILL_COLORS[cid],
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15,15,40,0.6)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)",
                            tickfont=dict(color="#9CA3AF")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#9CA3AF"),
        ),
        title="Cluster Feature Comparison (Radar Chart)",
        title_font_color="#A78BFA",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return _fig_to_json(fig)
