# Customer Segmentation Dashboard

A **premium Flask web application** for K-Means customer segmentation on the Mall Customers dataset.
Built as a Final Year Project for **Lovely Professional University**.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn plotly matplotlib seaborn joblib
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
Customer-Segmentation-Dashboard/
├── app.py                        ← Flask app with all routes
├── requirements.txt              ← Python dependencies
├── data/
│   └── Mall_Customers.csv        ← Dataset
├── static/
│   ├── css/main.css              ← Design system
│   └── js/main.js                ← Interactive JS
├── templates/
│   ├── base.html                 ← Navbar + footer layout
│   ├── home.html                 ← Landing page
│   ├── dataset.html              ← Dataset overview
│   ├── eda.html                  ← Step-by-step EDA
│   ├── cleaning.html             ← Data cleaning
│   ├── scaling.html              ← Feature scaling
│   ├── visualizations.html       ← EDA charts (Plotly)
│   ├── elbow.html                ← Elbow method
│   ├── silhouette.html           ← Silhouette score
│   ├── clustering.html           ← K-Means model
│   ├── cluster_viz.html          ← Cluster plots
│   ├── interpretation.html       ← Cluster profiles
│   ├── recommendations.html      ← Marketing strategies
│   ├── dashboard.html            ← Analytics dashboard
│   └── predict.html              ← New customer prediction
├── utils/
│   ├── preprocessing.py          ← Load, clean, scale
│   ├── eda.py                    ← EDA functions
│   ├── clustering.py             ← Elbow, Silhouette, KMeans
│   ├── visualization.py          ← All Plotly charts
│   ├── recommendations.py        ← Cluster profiles & strategies
│   └── database.py               ← SQLite operations
└── models/                       ← Auto-created: scaler.pkl, kmeans_model.pkl, customers.db
```

---

## 🔗 Pages

| URL | Page |
|-----|------|
| `/` | Home |
| `/dataset` | Dataset Overview |
| `/eda` | EDA Workflow |
| `/cleaning` | Data Cleaning |
| `/scaling` | Feature Scaling |
| `/visualizations` | EDA Charts |
| `/elbow` | Elbow Method |
| `/silhouette` | Silhouette Score |
| `/clustering` | K-Means Clustering |
| `/cluster-visualization` | Cluster Plots |
| `/interpretation` | Cluster Interpretation |
| `/recommendations` | Marketing Recommendations |
| `/dashboard` | Analytics Dashboard |
| `/predict` | Predict New Customer |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11, Flask 3.0
- **ML**: Scikit-learn (KMeans, StandardScaler, Silhouette Score)
- **Data**: Pandas, NumPy
- **Charts**: Plotly (interactive), Matplotlib, Seaborn
- **Database**: SQLite
- **Frontend**: Bootstrap 5, Vanilla CSS, Vanilla JS

---

## 📊 Dataset

The **Mall Customers Dataset** (Kaggle) — 200 customers, 5 features:
- `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

---


