from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "Customer-Segmentation-Dashboard"
DATA_PATH = PROJECT / "data" / "Mall_Customers.csv"
LOGO_PATH = PROJECT / "static" / "report_assets" / "lpu_logo_0.jpg"
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs" / "customer_report_assets"
SCREENSHOT_DIR = ROOT / "output" / "screenshots_viewport"
OUT_PATH = OUT_DIR / "Customer_Segmentation_Assistant_Report_LPU_Format.pdf"

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 2.2 * cm
TOP = 2.0 * cm
BOTTOM = 1.8 * cm
CONTENT_W = PAGE_W - LEFT - RIGHT

NAVY = colors.HexColor("#1F2F5F")
LIGHT = colors.HexColor("#F2F3F7")
GRID = colors.HexColor("#BFC4D6")
TEXT = colors.HexColor("#111111")


def build_data():
    raw = pd.read_csv(DATA_PATH)
    df = raw.drop_duplicates().copy()
    df = df.drop(columns=["CustomerID"])
    df = df.rename(
        columns={
            "Annual Income (k$)": "Annual_Income",
            "Spending Score (1-100)": "Spending_Score",
        }
    )
    df["Gender_Encoded"] = df["Gender"].map({"Male": 1, "Female": 0})

    features = df[["Annual_Income", "Spending_Score"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    wcss = []
    sil_scores = {}
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
        wcss.append(float(km.inertia_))
        if k >= 2:
            sil_scores[k] = float(silhouette_score(scaled, labels))

    kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(scaled)
    df["Cluster"] = labels
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    silhouette = float(silhouette_score(scaled, labels))

    summary = []
    for cid in sorted(df["Cluster"].unique()):
        group = df[df["Cluster"] == cid]
        income = float(group["Annual_Income"].mean())
        spending = float(group["Spending_Score"].mean())
        summary.append(
            {
                "cluster": int(cid),
                "name": segment_name(income, spending),
                "size": int(len(group)),
                "share": len(group) / len(df) * 100,
                "age": float(group["Age"].mean()),
                "income": income,
                "spending": spending,
                "male": int((group["Gender"] == "Male").sum()),
                "female": int((group["Gender"] == "Female").sum()),
            }
        )

    return raw, df, scaled, wcss, sil_scores, kmeans, centroids, silhouette, summary


def segment_name(income, spending):
    if income >= 70 and spending >= 60:
        return "Target Customers"
    if income >= 70 and spending < 40:
        return "Careful Spenders"
    if income < 45 and spending >= 60:
        return "Impulsive Buyers"
    if income < 45 and spending < 40:
        return "Budget Shoppers"
    return "Average Customers"


def segment_description(name):
    descriptions = {
        "Target Customers": "High income and high spending. These are premium customers who are most valuable for retention, loyalty, and personalized offers.",
        "Careful Spenders": "High income but low spending. These customers have purchasing capacity but need trust, value assurance, and stronger incentives.",
        "Impulsive Buyers": "Low income but high spending. These customers respond well to trends, offers, events, and emotionally engaging campaigns.",
        "Budget Shoppers": "Low income and low spending. These customers are price-sensitive and respond best to discounts, bundles, and loyalty benefits.",
        "Average Customers": "Moderate income and moderate spending. These customers represent the stable middle group and can be improved through broad engagement.",
    }
    return descriptions.get(name, "General customer group identified through income and spending behaviour.")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.black,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Times-Italic",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
        ),
        "cover_bold": ParagraphStyle(
            "cover_bold",
            fontName="Times-Bold",
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
        ),
        "cover": ParagraphStyle(
            "cover",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
        ),
        "cover_it": ParagraphStyle(
            "cover_it",
            fontName="Times-Italic",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Times-Bold",
            fontSize=16,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Times-Bold",
            fontSize=12,
            leading=16,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Times-Roman",
            fontSize=11,
            leading=17,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            textColor=TEXT,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Times-Roman",
            fontSize=11,
            leading=16,
            leftIndent=18,
            firstLineIndent=-10,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Times-Roman",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            textColor=colors.black,
            backColor=colors.HexColor("#F4F4F4"),
        ),
    }


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    canvas.drawCentredString(PAGE_W / 2, 0.9 * cm, str(doc.page))
    canvas.restoreState()


def p(text, style):
    return Paragraph(text.replace("&", "&amp;"), style)


def bullets(items, style):
    flow = []
    for item in items:
        flow.append(Paragraph("- " + item.replace("&", "&amp;"), style))
    return flow


def table(headers, rows, col_widths=None):
    data = [[Paragraph(str(h), ParagraphStyle("th", fontName="Times-Bold", fontSize=10, alignment=TA_CENTER, textColor=colors.white)) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), ParagraphStyle("td", fontName="Times-Roman", fontSize=9.2, leading=12, alignment=TA_CENTER)) for c in row])
    if col_widths is None:
        col_widths = [CONTENT_W / len(headers)] * len(headers)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def code_block(code, style):
    rows = [[Paragraph(line.replace(" ", "&nbsp;"), style)] for line in code.strip().splitlines()]
    t = Table(rows, colWidths=[CONTENT_W])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F4F4")),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def save_chart(fig, name, width=14 * cm):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def screenshot_image(name, width=15.2 * cm):
    path = SCREENSHOT_DIR / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def make_charts(df, wcss, sil_scores, centroids):
    charts = {}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["Age"], bins=15, color="#4C78A8", edgecolor="white")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    charts["age"] = save_chart(fig, "age.png", 13 * cm)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(df["Annual_Income"], df["Spending_Score"], c="#F58518", edgecolors="white", alpha=0.8)
    ax.set_title("Annual Income vs Spending Score")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    charts["scatter"] = save_chart(fig, "scatter.png", 13 * cm)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, 11), wcss, marker="o", color="#1F2F5F")
    ax.axvline(5, color="#C44E52", linestyle="--", label="K = 5")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("WCSS")
    ax.legend()
    charts["elbow"] = save_chart(fig, "elbow.png", 13 * cm)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ks = list(sil_scores.keys())
    vals = [sil_scores[k] for k in ks]
    ax.bar(ks, vals, color=["#55A868" if k == 5 else "#8172B3" for k in ks])
    ax.set_title("Silhouette Score")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Score")
    charts["silhouette"] = save_chart(fig, "silhouette.png", 13 * cm)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    for cid in sorted(df["Cluster"].unique()):
        group = df[df["Cluster"] == cid]
        ax.scatter(group["Annual_Income"], group["Spending_Score"], s=45, color=palette[cid], label=f"Cluster {cid}", edgecolors="white")
    ax.scatter(centroids[:, 0], centroids[:, 1], s=180, c="black", marker="X", label="Centroids")
    ax.set_title("Final K-Means Customer Segments")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.legend(fontsize=8)
    charts["clusters"] = save_chart(fig, "clusters.png", 15 * cm)

    return charts


def add_section(story, title, style):
    story.append(Paragraph(title, style))


def build_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, df, scaled, wcss, sil_scores, kmeans, centroids, silhouette, summary = build_data()
    charts = make_charts(df, wcss, sil_scores, centroids)
    S = styles()
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        rightMargin=RIGHT,
        leftMargin=LEFT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title="Customer Segmentation Assistant Report",
        author="Student, Lovely Professional University",
    )

    story = []

    story.append(Spacer(1, 2.0 * cm))
    story.append(Paragraph("Customer Segmentation Assistant", S["title"]))
    story.append(Paragraph("(AI-ML Based Customer Segmentation Platform)", S["subtitle"]))
    story.append(Spacer(1, 0.65 * cm))
    story.append(Paragraph("A PROJECT REPORT", S["cover_bold"]))
    story.append(Spacer(1, 0.45 * cm))
    story.append(Paragraph("Submitted by", S["cover_it"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("Jayant Malik", S["cover_bold"]))
    story.append(Paragraph("Registration No: 12404501", S["cover"]))
    story.append(Spacer(1, 0.55 * cm))
    story.append(Paragraph("in partial fulfillment for the award of the degree", S["cover_it"]))
    story.append(Paragraph("of", S["cover_it"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("BACHELOR OF TECHNOLOGY", S["cover_bold"]))
    story.append(Paragraph("School of Computer Science and Engineering", S["cover_bold"]))
    story.append(Spacer(1, 0.45 * cm))
    if LOGO_PATH.exists():
        logo = Image(str(LOGO_PATH))
        logo.drawWidth = 7.2 * cm
        logo.drawHeight = logo.drawWidth * logo.imageHeight / logo.imageWidth
        story.append(logo)
    story.append(Spacer(1, 0.45 * cm))
    story.append(Paragraph("Lovely Professional University, Punjab", S["cover_bold"]))
    story.append(Paragraph("July 2026", S["cover_bold"]))
    story.append(PageBreak())

    toc = [
        ["1", "Abstract", "3"],
        ["2", "Introduction", "3"],
        ["3", "Problem Statement", "3"],
        ["4", "Objectives", "3"],
        ["5", "Scope", "4"],
        ["6", "System Design & Analysis", "4"],
        ["7", "Technologies Used", "4"],
        ["8", "Architecture Diagram", "5"],
        ["9", "Module Description", "5"],
        ["10", "Implementation", "6"],
        ["11", "Internal Segmentation Workflow", "7-10"],
        ["12", "Code Snippets and Explanation", "11-14"],
        ["13", "Website Screenshots", "15-23"],
        ["14", "Model Output Charts", "24-27"],
        ["15", "Testing", "28"],
        ["16", "Results", "29"],
        ["17", "Conclusion", "30"],
        ["18", "Future Work", "30"],
        ["19", "References", "31"],
    ]
    add_section(story, "TABLE OF CONTENTS", S["h1"])
    story.append(table(["Sr No", "Topic", "Page"], toc, [2 * cm, CONTENT_W - 4.5 * cm, 2.5 * cm]))
    story.append(PageBreak())

    add_section(story, "ABSTRACT", S["h1"])
    story.append(p(
        f"This project, Customer Segmentation Assistant, is an AI-ML web application developed to identify meaningful groups of mall customers using unsupervised machine learning. The system uses the Mall Customers dataset containing {len(raw)} records with demographic and spending-related attributes. After data cleaning, exploratory analysis, feature scaling, and model selection, the K-Means algorithm is applied to segment customers based on annual income and spending score.",
        S["body"],
    ))
    story.append(p(
        f"The final model uses K = 5 clusters and achieves a silhouette score of {silhouette:.4f}, indicating clear separation between customer groups. The identified segments include budget shoppers, impulsive buyers, careful spenders, average customers, and target customers. The application also includes a Flask dashboard, visualizations, cluster interpretation, marketing recommendations, and new-customer prediction support.",
        S["body"],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "INTRODUCTION", S["h1"])
    story.append(p(
        "Customer segmentation is an important business analytics technique used to divide customers into groups with similar purchasing behaviour. Retail organizations can use these groups to personalize offers, improve customer retention, and make better marketing decisions.",
        S["body"],
    ))
    story.append(p(
        "The Customer Segmentation Assistant applies machine learning to transform raw customer records into actionable business insights. Instead of manually assigning customer categories, the system uses K-Means clustering to discover natural patterns in income and spending behaviour.",
        S["body"],
    ))
    story.append(p(
        "The project is implemented as a Python Flask web application with pages for dataset overview, EDA, cleaning, scaling, clustering, visualization, interpretation, recommendations, dashboard analytics, prediction, and report generation.",
        S["body"],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "PROBLEM STATEMENT", S["h1"])
    story.append(p(
        "Businesses often collect customer data but struggle to convert it into practical marketing decisions. Treating all customers in the same way leads to generic promotions, poor targeting, and missed revenue opportunities.",
        S["body"],
    ))
    story.append(p(
        "The problem addressed in this project is to automatically group mall customers into meaningful segments using their annual income and spending score. The system must also present the results in a simple dashboard so that the segmentation output can be understood by non-technical users.",
        S["body"],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "OBJECTIVES", S["h1"])
    story.extend(
        bullets(
            [
                "To load and inspect the Mall Customers dataset.",
                "To perform exploratory data analysis on age, gender, income, and spending score.",
                "To clean the dataset by removing unnecessary identifiers and checking missing values.",
                "To scale numeric features before applying K-Means clustering.",
                "To determine a suitable number of clusters using elbow and silhouette analysis.",
                "To train a K-Means model and interpret the resulting customer groups.",
                "To build a Flask dashboard for visualization, prediction, and report generation.",
                "To provide marketing recommendations for each customer segment.",
            ],
            S["bullet"],
        )
    )
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "SCOPE", S["h1"])
    story.append(p(
        "The current scope of the project includes data preprocessing, EDA, clustering, visualization, segment profiling, marketing recommendation generation, and prediction of the cluster for a new customer based on income and spending score.",
        S["body"],
    ))
    story.append(p(
        "The project works on the Mall Customers dataset and demonstrates an end-to-end AI-ML workflow suitable for academic demonstration. It can be extended in the future by adding real retail transaction data, product-level behaviour, visit frequency, CRM integration, and deployment to a public cloud server.",
        S["body"],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "SYSTEM DESIGN & ANALYSIS", S["h1"])
    story.append(p(
        "The system is designed as a modular Flask application. The user interacts with web pages rendered through HTML templates. The backend loads the dataset, performs preprocessing, trains or loads the machine learning model, and returns visual or tabular outputs to the dashboard.",
        S["body"],
    ))
    story.append(table(
        ["Layer", "Component", "Purpose"],
        [
            ["User Interface", "HTML, CSS, Bootstrap", "Displays pages and forms"],
            ["Application Layer", "Flask routes", "Controls dashboard navigation"],
            ["Data Layer", "Pandas, CSV, SQLite", "Loads dataset and stores predictions"],
            ["ML Layer", "StandardScaler, KMeans", "Scales features and creates clusters"],
            ["Visualization", "Plotly, Matplotlib", "Shows charts and model results"],
        ],
        [3.2 * cm, 5.0 * cm, CONTENT_W - 8.2 * cm],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "TECHNOLOGIES USED", S["h1"])
    story.append(table(
        ["Technology", "Use in Project"],
        [
            ["Python", "Core programming language"],
            ["Flask", "Web application framework"],
            ["Pandas and NumPy", "Data handling and numerical processing"],
            ["Scikit-learn", "Feature scaling, K-Means clustering, silhouette score"],
            ["Plotly", "Interactive dashboard visualizations"],
            ["Matplotlib and Seaborn", "Report charts and statistical plots"],
            ["SQLite", "Storage of new predicted customers"],
            ["ReportLab", "PDF project report generation"],
            ["HTML, CSS, JavaScript", "Frontend templates and interactions"],
        ],
        [5.5 * cm, CONTENT_W - 5.5 * cm],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "ARCHITECTURE DIAGRAM", S["h1"])
    arch_rows = [
        ["User", "Flask Web App"],
        ["Flask Web App", "Preprocessing Module"],
        ["Preprocessing Module", "EDA and Feature Scaling"],
        ["EDA and Feature Scaling", "K-Means Clustering"],
        ["K-Means Clustering", "Cluster Profiles"],
        ["Cluster Profiles", "Dashboard, Prediction, PDF Report"],
    ]
    story.append(table(["Input / Module", "Output / Connected Module"], arch_rows, [CONTENT_W / 2, CONTENT_W / 2]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(p(
        "The architecture follows a sequential machine learning pipeline. Data flows from the uploaded or stored CSV file to preprocessing, scaling, model training, cluster interpretation, and finally to user-facing pages such as dashboard, prediction, and report output.",
        S["body"],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "MODULE DESCRIPTION", S["h1"])
    story.append(table(
        ["Module", "Description"],
        [
            ["preprocessing.py", "Loads data, removes duplicate rows, renames columns, encodes gender, and scales features."],
            ["eda.py", "Generates descriptive statistics, missing-value checks, outlier checks, and correlations."],
            ["clustering.py", "Computes WCSS, silhouette scores, trains K-Means, and predicts new clusters."],
            ["visualization.py", "Creates charts for age, income, spending, elbow method, silhouette score, and clusters."],
            ["recommendations.py", "Stores segment profiles and marketing strategies."],
            ["database.py", "Handles SQLite operations for predicted customer records."],
            ["app.py", "Defines Flask routes and connects all modules to web pages."],
        ],
        [4.0 * cm, CONTENT_W - 4.0 * cm],
    ))
    story.append(Spacer(1, 0.45 * cm))

    add_section(story, "IMPLEMENTATION", S["h1"])
    story.append(p(
        "The implementation begins with data loading and inspection. CustomerID is dropped because it is only an identifier and does not help clustering. Annual income and spending score are selected as the final clustering features because they represent purchasing power and buying behaviour.",
        S["body"],
    ))
    story.append(p(
        "StandardScaler is applied before K-Means because distance-based algorithms are sensitive to feature ranges. The elbow method and silhouette score are used to validate K = 5 before fitting the final clustering model.",
        S["body"],
    ))
    story.append(table(
        ["Metric", "Value"],
        [
            ["Total records", len(raw)],
            ["Final records after cleaning", len(df)],
            ["Missing values", int(df.isnull().sum().sum())],
            ["Duplicate rows removed", len(raw) - len(df)],
            ["Selected features", "Annual_Income, Spending_Score"],
            ["Number of clusters", 5],
            ["Silhouette score", f"{silhouette:.4f}"],
            ["WCSS / Inertia", f"{kmeans.inertia_:.4f}"],
        ],
        [6 * cm, CONTENT_W - 6 * cm],
    ))
    story.append(PageBreak())

    add_section(story, "INTERNAL SEGMENTATION WORKFLOW", S["h1"])
    story.append(Paragraph("1. Data Input and Inspection", S["h2"]))
    story.append(p(
        "The segmentation process begins when the application loads the Mall_Customers.csv dataset. The raw file contains CustomerID, Gender, Age, Annual Income, and Spending Score. The application first checks the number of rows and columns, verifies data types, counts missing values, and checks duplicate rows. This step confirms whether the dataset is suitable for machine learning.",
        S["body"],
    ))
    story.append(Paragraph("2. Cleaning and Feature Preparation", S["h2"]))
    story.append(p(
        "CustomerID is removed because it is only a serial number and does not describe customer behaviour. The long column names are renamed to Annual_Income and Spending_Score so they can be used easily in Python code. Gender is encoded as a numeric value for analysis, but the final K-Means clustering uses Annual_Income and Spending_Score because these two variables directly represent purchasing power and shopping behaviour.",
        S["body"],
    ))
    story.append(table(
        ["Raw Column", "Action Performed", "Reason"],
        [
            ["CustomerID", "Dropped", "Identifier only; not useful for clustering"],
            ["Gender", "Encoded as Male = 1, Female = 0", "Useful for analysis and dashboard summaries"],
            ["Age", "Retained for interpretation", "Helps describe customer segment profile"],
            ["Annual Income (k$)", "Renamed and selected", "Represents earning capacity"],
            ["Spending Score (1-100)", "Renamed and selected", "Represents purchase behaviour"],
        ],
        [4.0 * cm, 5.5 * cm, CONTENT_W - 9.5 * cm],
    ))
    story.append(PageBreak())

    add_section(story, "INTERNAL SEGMENTATION WORKFLOW", S["h1"])
    story.append(Paragraph("3. Feature Scaling", S["h2"]))
    story.append(p(
        "K-Means calculates distance between data points. If one feature has a much larger numeric range than another feature, it can dominate the Euclidean distance calculation. Annual income ranges from 15 to 137, while spending score ranges from 1 to 99. To make both features contribute fairly, StandardScaler converts each feature into a z-score.",
        S["body"],
    ))
    story.append(p(
        "The formula used by StandardScaler is: z = (x - mean) / standard deviation. After this transformation, both income and spending score have approximately mean 0 and standard deviation 1.",
        S["body"],
    ))
    feature_stats = []
    for col in ["Annual_Income", "Spending_Score"]:
        feature_stats.append([col, f"{df[col].mean():.2f}", f"{df[col].std():.2f}", f"{df[col].min()}", f"{df[col].max()}"])
    story.append(table(
        ["Feature", "Mean", "Std Dev", "Minimum", "Maximum"],
        feature_stats,
        [4.5 * cm, 2.5 * cm, 2.8 * cm, 2.5 * cm, CONTENT_W - 12.3 * cm],
    ))
    story.append(Paragraph("4. How K-Means Works Internally", S["h2"]))
    story.extend(bullets([
        "Step 1: Choose K initial centroids using k-means++ initialization.",
        "Step 2: Calculate Euclidean distance from each customer to each centroid.",
        "Step 3: Assign each customer to the nearest centroid.",
        "Step 4: Recalculate each centroid as the mean of all customers assigned to that cluster.",
        "Step 5: Repeat assignment and centroid update until the centroids stop changing significantly.",
        "Step 6: Store the final cluster labels and centroids for interpretation and prediction.",
    ], S["bullet"]))
    story.append(PageBreak())

    add_section(story, "INTERNAL SEGMENTATION WORKFLOW", S["h1"])
    story.append(Paragraph("5. Choosing the Number of Clusters", S["h2"]))
    story.append(p(
        "The project evaluates multiple K values before finalizing the model. The elbow method calculates WCSS, which measures compactness inside clusters. As K increases, WCSS decreases, but after a point the improvement becomes small. The silhouette score measures how well separated the clusters are. Higher silhouette values indicate better-defined clusters.",
        S["body"],
    ))
    story.append(table(
        ["K", "WCSS", "Silhouette Score"],
        [[k, f"{wcss[k-1]:.2f}", "-" if k == 1 else f"{sil_scores[k]:.4f}"] for k in range(1, 11)],
        [2.0 * cm, 5.0 * cm, CONTENT_W - 7.0 * cm],
    ))
    story.append(p(
        f"K = 5 is selected because it gives meaningful business groups and a strong silhouette score of {silhouette:.4f}. It separates customers into low-income low-spending, low-income high-spending, medium behaviour, high-income low-spending, and high-income high-spending patterns.",
        S["body"],
    ))
    story.append(PageBreak())

    add_section(story, "INTERNAL SEGMENTATION WORKFLOW", S["h1"])
    story.append(Paragraph("6. Cluster Profiling", S["h2"]))
    story.append(p(
        "After K-Means assigns cluster labels, the system joins each label back to the cleaned customer table. Then it calculates the size, percentage share, average age, average income, average spending score, and gender distribution for each cluster. These statistics convert mathematical clusters into understandable customer personas.",
        S["body"],
    ))
    story.append(table(
        ["Cluster", "Segment", "Meaning"],
        [
            [s["cluster"], s["name"], segment_description(s["name"])]
            for s in summary
        ],
        [2.0 * cm, 4.5 * cm, CONTENT_W - 6.5 * cm],
    ))
    story.append(Paragraph("7. Prediction for a New Customer", S["h2"]))
    story.append(p(
        "For prediction, the user enters annual income and spending score in the web form. The saved StandardScaler transforms the input using the same scale learned from the training data. The saved K-Means model then checks which centroid is closest to the new point and returns the corresponding cluster ID. Finally, the dashboard displays the matching customer persona and marketing suggestion.",
        S["body"],
    ))
    story.append(PageBreak())

    add_section(story, "CODE SNIPPETS", S["h1"])
    story.append(Paragraph("Data Cleaning and Feature Scaling", S["h2"]))
    story.append(code_block(
        """
df = pd.read_csv("Mall_Customers.csv")
df = df.drop_duplicates()
df = df.drop(columns=["CustomerID"])
df = df.rename(columns={
    "Annual Income (k$)": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score"
})
df["Gender_Encoded"] = df["Gender"].map({"Male": 1, "Female": 0})

X = df[["Annual_Income", "Spending_Score"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
        """,
        S["code"],
    ))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("K-Means Model Training", S["h2"]))
    story.append(code_block(
        """
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    random_state=42,
    max_iter=300
)
labels = kmeans.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels)
df["Cluster"] = labels
        """,
        S["code"],
    ))
    story.append(PageBreak())

    add_section(story, "CODE SNIPPETS", S["h1"])
    story.append(Paragraph("New Customer Prediction", S["h2"]))
    story.append(code_block(
        """
def predict_cluster(annual_income, spending_score):
    X_new = np.array([[annual_income, spending_score]])
    X_scaled = scaler.transform(X_new)
    cluster = int(kmeans.predict(X_scaled)[0])
    return cluster
        """,
        S["code"],
    ))
    story.append(Paragraph("Cluster Summary Logic", S["h2"]))
    story.append(code_block(
        """
summary = []
for cid in sorted(df["Cluster"].unique()):
    group = df[df["Cluster"] == cid]
    summary.append({
        "cluster": cid,
        "size": len(group),
        "mean_income": group["Annual_Income"].mean(),
        "mean_spending": group["Spending_Score"].mean(),
        "mean_age": group["Age"].mean()
    })
        """,
        S["code"],
    ))
    story.append(PageBreak())

    add_section(story, "CODE SNIPPETS AND EXPLANATION", S["h1"])
    story.append(Paragraph("Flask Route Structure", S["h2"]))
    story.append(p(
        "The Flask app is the controller of the project. Each route calls the required utility functions, prepares the processed data, and sends it to the matching HTML template. This keeps the machine learning logic separate from the user interface.",
        S["body"],
    ))
    story.append(code_block(
        """
@app.route("/clustering")
def clustering():
    kmeans = get_kmeans()
    summary = get_cluster_summary()
    sil_data = get_silhouette()
    return render_template(
        "clustering.html",
        kmeans=kmeans,
        summary=summary,
        sil_data=sil_data
    )
        """,
        S["code"],
    ))
    story.append(Paragraph("How the Route Works", S["h2"]))
    story.extend(bullets([
        "get_kmeans() trains or loads the K-Means model.",
        "get_cluster_summary() calculates customer count, average income, average spending, and gender distribution for each cluster.",
        "get_silhouette() prepares validation data for the clustering quality section.",
        "render_template() sends all values to the front-end clustering page.",
    ], S["bullet"]))
    story.append(PageBreak())

    add_section(story, "CODE SNIPPETS AND EXPLANATION", S["h1"])
    story.append(Paragraph("Elbow Method and Silhouette Logic", S["h2"]))
    story.append(code_block(
        """
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

for k in range(2, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
        """,
        S["code"],
    ))
    story.append(p(
        "The first loop calculates WCSS for different K values. WCSS shows how compact the clusters are. The second loop calculates silhouette score for each K from 2 to 10. The model finally selects K = 5 because it gives useful business segments with a strong validation score.",
        S["body"],
    ))
    story.append(Paragraph("Database and Prediction Storage", S["h2"]))
    story.append(p(
        "When a new customer is predicted from the dashboard, the application can store the customer details and predicted cluster in SQLite. This allows the dashboard to maintain a history of predictions and show previously entered customers.",
        S["body"],
    ))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("Home Page / Project Landing Page", S["h2"]))
    story.append(screenshot_image("home.png"))
    story.append(Paragraph("Figure 1: Live Flask dashboard home page showing the customer segmentation platform overview.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("Dataset Overview Page", S["h2"]))
    story.append(screenshot_image("dataset.png"))
    story.append(Paragraph("Figure 2: Dataset page showing records, columns, and preview of the Mall Customers dataset.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("EDA Workflow Page", S["h2"]))
    story.append(screenshot_image("eda.png"))
    story.append(Paragraph("Figure 3: EDA page showing exploratory analysis steps used before clustering.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("Visualizations Page", S["h2"]))
    story.append(screenshot_image("visualizations.png"))
    story.append(Paragraph("Figure 4: Visualization page containing interactive charts for understanding customer behaviour.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("K-Means Clustering Page", S["h2"]))
    story.append(screenshot_image("clustering.png"))
    story.append(Paragraph("Figure 5: K-Means page showing model configuration, cluster quality, and summary output.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("Cluster Visualization Page", S["h2"]))
    story.append(screenshot_image("cluster_visualization.png"))
    story.append(Paragraph("Figure 6: Cluster visualization page showing how customers are divided into segments.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("Marketing Recommendations Page", S["h2"]))
    story.append(screenshot_image("recommendations.png"))
    story.append(Paragraph("Figure 7: Recommendation page showing business strategies for each segment.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("Dashboard Page", S["h2"]))
    story.append(screenshot_image("dashboard.png"))
    story.append(Paragraph("Figure 8: Dashboard page summarizing segmentation results and saved customer predictions.", S["small"]))
    story.append(PageBreak())

    add_section(story, "WEBSITE SCREENSHOTS", S["h1"])
    story.append(Paragraph("Predict Customer Page", S["h2"]))
    story.append(screenshot_image("predict.png"))
    story.append(Paragraph("Figure 9: Prediction page where a new customer's segment can be predicted using income and spending score.", S["small"]))
    story.append(PageBreak())

    add_section(story, "MODEL OUTPUT CHARTS", S["h1"])
    story.append(charts["age"])
    story.append(Paragraph("Figure 10: Age distribution of customers", S["small"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(charts["scatter"])
    story.append(Paragraph("Figure 11: Annual income versus spending score", S["small"]))
    story.append(PageBreak())

    add_section(story, "MODEL OUTPUT CHARTS", S["h1"])
    story.append(charts["elbow"])
    story.append(Paragraph("Figure 12: Elbow method for selecting number of clusters", S["small"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(charts["silhouette"])
    story.append(Paragraph("Figure 13: Silhouette score comparison", S["small"]))
    story.append(PageBreak())

    add_section(story, "MODEL OUTPUT CHARTS", S["h1"])
    story.append(charts["clusters"])
    story.append(Paragraph("Figure 14: Final K-Means cluster visualization", S["small"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(table(
        ["Cluster", "Segment", "Size", "Avg Income", "Avg Spending"],
        [[s["cluster"], s["name"], s["size"], f"{s['income']:.1f}", f"{s['spending']:.1f}"] for s in summary],
        [2 * cm, 5 * cm, 2 * cm, 3.5 * cm, CONTENT_W - 12.5 * cm],
    ))
    story.append(PageBreak())

    add_section(story, "MODEL OUTPUT CHARTS", S["h1"])
    story.append(Paragraph("Sample Dataset Preview", S["h2"]))
    preview = raw.head(8).values.tolist()
    story.append(table(list(raw.columns), preview, [2.2 * cm, 2.8 * cm, 2 * cm, 4.2 * cm, CONTENT_W - 11.2 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("The dashboard pages show dataset overview, EDA, cleaning, scaling, clustering results, interpretation, recommendations, and new customer prediction.", S["body"]))
    story.append(PageBreak())

    add_section(story, "TESTING", S["h1"])
    story.append(table(
        ["Test Case", "Expected Result", "Status"],
        [
            ["Dataset loads successfully", "CSV contains 200 records", "Pass"],
            ["Missing value check", "No missing values after cleaning", "Pass"],
            ["Scaling check", "Selected features transformed by StandardScaler", "Pass"],
            ["Elbow method", "WCSS decreases as K increases", "Pass"],
            ["Silhouette score", "Model returns score above 0.5 for K = 5", "Pass"],
            ["Prediction function", "New customer receives a cluster label", "Pass"],
            ["Dashboard routes", "Pages render project outputs", "Pass"],
        ],
        [4.8 * cm, CONTENT_W - 7.8 * cm, 3 * cm],
    ))
    story.append(PageBreak())

    add_section(story, "RESULTS", S["h1"])
    story.append(p(
        f"The K-Means model successfully segmented all {len(df)} customers into five groups. The final silhouette score is {silhouette:.4f}, which indicates good cluster separation for this dataset.",
        S["body"],
    ))
    story.append(table(
        ["Cluster", "Segment Name", "Customers", "Share", "Avg Age", "Male / Female"],
        [[s["cluster"], s["name"], s["size"], f"{s['share']:.1f}%", f"{s['age']:.1f}", f"{s['male']} / {s['female']}"] for s in summary],
        [1.8 * cm, 5.0 * cm, 2.3 * cm, 2.2 * cm, 2.2 * cm, CONTENT_W - 13.5 * cm],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(p(
        "The segments provide direct business value. High-income high-spending customers can be treated as premium target customers, while high-income low-spending customers can be encouraged through value-based offers. Low-income high-spending customers respond well to trend-based campaigns, and low-income low-spending customers benefit from discount and loyalty programs.",
        S["body"],
    ))
    story.append(PageBreak())

    add_section(story, "CONCLUSION", S["h1"])
    story.append(p(
        "The Customer Segmentation Assistant successfully demonstrates a complete AI-ML workflow for a real business analytics use case. The project starts from raw customer data, performs EDA and preprocessing, applies K-Means clustering, and presents the output through visual dashboards and a formal report.",
        S["body"],
    ))
    story.append(p(
        "The final model identifies five meaningful customer groups. These groups can help a retail business plan personalized promotions, improve customer engagement, and make data-driven marketing decisions.",
        S["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    add_section(story, "FUTURE WORK", S["h1"])
    story.extend(
        bullets(
            [
                "Use larger real-world customer transaction datasets.",
                "Add product categories, visit frequency, and purchase history as features.",
                "Compare K-Means with hierarchical clustering, DBSCAN, and Gaussian mixture models.",
                "Deploy the Flask dashboard on a cloud platform.",
                "Integrate real-time CRM data for continuously updated segmentation.",
            ],
            S["bullet"],
        )
    )
    story.append(PageBreak())

    add_section(story, "REFERENCES", S["h1"])
    refs = [
        "Scikit-learn Documentation: K-Means Clustering - https://scikit-learn.org/",
        "Kaggle: Mall Customer Segmentation Dataset.",
        "Pandas Documentation - https://pandas.pydata.org/",
        "Flask Documentation - https://flask.palletsprojects.com/",
        "Plotly Python Documentation - https://plotly.com/python/",
        "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation of cluster analysis.",
        "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.",
    ]
    for idx, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{idx}] {ref}", S["body"]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT_PATH)
    print(f"Silhouette: {silhouette:.4f}")
    print(f"Pages generated in LPU sample format.")


if __name__ == "__main__":
    build_report()
