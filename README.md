# 🛒 Store Sales Time Series Forecasting

A machine learning project forecasting daily sales for Corporación Favorita — a large Ecuadorian grocery retailer — using Prophet and LightGBM with lag-based feature engineering.

---

## 📌 Project Overview

Time series forecasting is a common real-world problem where naive approaches fail quickly. This project builds a full forecasting pipeline — from EDA and decomposition to production-ready models — and compares three approaches by MAE.

**Key insight:** LightGBM with lag features outperforms Facebook Prophet by ~25%, and both beat the naive baseline by 2x.

---

## 📁 Project Structure

```
retail-sales-time-series/
│
├── data/                           # Empty — see Dataset section below
│   └── .gitkeep
│
├── models/
│   └── lgbm_store_sales.pkl        # Saved best model (LightGBM+)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis & Decomposition
│   └── 02_modeling.ipynb           # Baseline, Prophet, LightGBM, comparison
│
├── .gitignore
├── README.md
└── pyproject.toml
```

---

## 📊 Dataset

> The `data/` folder is empty. Download all required files from Kaggle and place them there:
> 👉 [Store Sales — Time Series Forecasting (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

| File | Description |
|------|-------------|
| `train.csv` | Historical sales by store and product family |
| `holidays_events.csv` | Holidays and events in Ecuador |
| `stores.csv` | Store metadata (city, type, cluster) — not used |
| `oil.csv` | Daily oil prices — not used |
| `transactions.csv` | Daily transaction counts — not used |


| Property | Value |
|----------|-------|
| Total rows | 3,000,888 |
| Date range | 2013-01-01 → 2017-08-15 |
| Stores | 54 |
| Product families | 33 |
| Target variable | `sales` — units sold per store/family/day |

---

## 🔍 Exploratory Data Analysis

- Upward sales trend from 2013 to 2017
- Clear weekly seasonality — weekends significantly outperform weekdays, Thursday is the weakest day
- Anomalies at the start of each year — stores closed during national holidays
- Seasonal decomposition confirmed trend + weekly pattern + residual outliers on holidays

---

## ⚙️ Feature Engineering

LightGBM has no built-in sense of time — all temporal information was manually encoded:

| Feature | Description |
|---------|-------------|
| `lag_1` | Sales yesterday |
| `lag_7` | Sales 7 days ago |
| `lag_14` | Sales 14 days ago |
| `lag_30` | Sales 30 days ago |
| `day_of_week` | 0=Monday … 6=Sunday |
| `month` | Month of year |

---

## 🤖 Models Compared

| Model | MAE |
|-------|-----|
| Naive (yesterday's value) | 143,714 |
| Prophet (Meta) | 70,890 |
| **LightGBM+ ✅** | **54,452** |

> Split: train `2013-01-01 → 2017-05-14`, test `2017-05-15 → 2017-08-15` (~93 days)
> Walk-forward validation — no shuffle, no data leakage.

---

## 💡 Key Decisions

**Why not random train/test split?**
Time series data has temporal order — shuffling causes data leakage where the model sees future data during training, resulting in falsely optimistic metrics that collapse in production.

**Why LightGBM over Prophet?**
Prophet is great out-of-the-box but treats all series the same. LightGBM with hand-crafted lag features better captures recent sales momentum — `lag_1` was the single most important feature.

**Why holidays didn't help?**
Ecuador's holidays are regional — adding them as a binary feature slightly hurt performance (MAE 58,549 vs 54,452) because they don't uniformly affect all stores.

---

## 📈 Feature Importance

`lag_1` (yesterday) dominates — short-term momentum is the strongest signal. `day_of_week` and `month` add moderate value on top of the lag structure.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Pandas / NumPy** — data manipulation
- **Matplotlib** — visualization
- **statsmodels** — seasonal decomposition
- **Prophet** — trend + seasonality forecasting
- **LightGBM** — gradient boosting regressor
- **scikit-learn** — metrics
- **Joblib** — model serialization
- **Poetry** — dependency management
- **Jupyter Notebook** — analysis & modeling

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ete9nal/retail-sales-time-series.git
cd retail-sales-time-series

# Install dependencies
poetry install

# Download data from Kaggle and place CSVs in data/

# Run notebooks
poetry run jupyter notebook
```
