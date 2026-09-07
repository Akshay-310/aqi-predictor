# Karachi AQI Predictor

A serverless, end-to-end machine learning system that forecasts the Air Quality Index (AQI) for Karachi, Pakistan, three days ahead — built as the capstone project for the **10Pearls Shine Internship Program** (Data Science track, 8 weeks).

The system covers the full pipeline: automated hourly data collection, feature engineering, model training and selection, SHAP-based explainability, and a live public dashboard — with a resilience layer engineered to survive real third-party platform outages.

---

## 🔴 Live Dashboard

**https://karachi-air-quality-ak310.streamlit.app/**

---

## 🏗️ Project Architecture

<img width="1672" height="941" alt="github image" src="https://github.com/user-attachments/assets/a3a2a746-851f-45ec-8868-48d8b909be2a" />


*(Pipeline flow: Open-Meteo API → hourly/backfill fetch scripts → Hopsworks Feature Store → feature engineering → model training & registry → live dashboard, with an hourly GitHub Actions-driven local backup layer for resilience.)*

---

## 📋 Overview

This project forecasts AQI for Karachi three days ahead using three independent regression models — one per horizon (day+1, day+2, day+3) — trained on two years of historical hourly weather and air-quality data. It goes beyond a working model to include:

- A fully automated CI/CD pipeline (hourly data collection, daily retraining)
- A live, publicly deployed interactive dashboard
- Full model interpretability via SHAP
- An automated test suite
- A resilience architecture that keeps the dashboard accurate even during data-platform outages

## ✨ Key Features

- **3-day AQI forecast** — independent models per horizon, each using the target day's forecasted weather as an input feature
- **Live dashboard** — current AQI, key pollutants, 3-day forecast, 2-year historical trend, year-over-year comparison, and a severity-based health advisory with an interactive alert indicator
- **Model interpretability** — a SHAP notebook explaining global and per-prediction feature importance for every horizon
- **Automated pipeline** — GitHub Actions run hourly data collection and daily model retraining without manual intervention
- **Resilience by design** — a local backup layer keeps the dashboard's live data accurate even if the feature store's sync layer is degraded
- **Tested** — an automated pytest suite covering feature-column consistency, data cleaning, category logic, and model sanity

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data source | [Open-Meteo API](https://open-meteo.com/) (weather + air quality, no key required) |
| Feature store / model registry | [Hopsworks](https://www.hopsworks.ai/) |
| Modelling | Python, scikit-learn, LightGBM, XGBoost |
| Interpretability | SHAP |
| Dashboard | Streamlit (deployed on Streamlit Community Cloud) |
| Automation | GitHub Actions |
| Testing | pytest |

## 📊 Model Results

| Horizon | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| Day +1 | Ridge Regression | 0.819 | 5.16 | 3.54 |
| Day +2 | Ridge Regression | 0.410 | 9.34 | 7.13 |
| Day +3 | ElasticNet | 0.221 | 10.54 | 7.99 |

All three models are verified against a naive persistence baseline. Day+3 was tested against five candidate algorithms (Ridge, Random Forest, LightGBM, ElasticNet, XGBoost); ElasticNet was selected as the only candidate that generalises cleanly rather than the one with the marginally highest raw score.

## 📁 Project Structure

```
aqi-predictor/
├── .github/workflows/       # Hourly fetch + daily retrain automation
├── app/
│   └── app.py                # Streamlit dashboard
├── src/
│   ├── backfill.py            # One-time historical data backfill
│   ├── fetch_current_data.py  # Hourly live data collection
│   ├── preprocessing.py       # Data quality checks & cleaning
│   ├── feature_engineering.py # Daily feature aggregation
│   ├── train_models.py        # Model training, comparison, registration
│   └── utils/                 # Diagnostic and maintenance scripts
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis
│   └── shap_analysis.ipynb     # Model interpretability
├── tests/                    # pytest suite
├── data/raw_backup/          # Local resilience backup (auto-managed)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started (Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/Akshay-310/aqi-predictor.git
cd aqi-predictor
```

### 2. Set up the environment

```bash
conda create -n aqi-predictor python=3.10
conda activate aqi-predictor
pip install -r requirements.txt
```

### 3. Configure credentials

This project uses [Hopsworks](https://www.hopsworks.ai/) as its feature store and model registry. To run it yourself, create a **free Hopsworks account and project**, then create a `.env` file in the project root:

```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT=your_project_name
HOPSWORKS_HOST=your_hopsworks_host
```

> **Note:** the live dashboard link above runs against the author's own Hopsworks project. To fully reproduce the pipeline (backfill → training → your own live dashboard), you'll need your own Hopsworks project and to re-run the setup steps below against it.

### 4. Run the data pipeline (first-time setup)

```bash
# One-time historical backfill (2 years of hourly data)
python src/backfill.py

# Build the daily feature table
python src/feature_engineering.py

# Train and register models
python src/train_models.py
```

### 5. Run the dashboard locally

```bash
streamlit run app/app.py
```

### 6. Run the test suite

```bash
pytest tests/ -v
```

## 🤖 Automation

Two GitHub Actions workflows keep the system current without manual intervention:

- **`fetch_data_hourly.yml`** — fetches the latest weather + air-quality data every hour and updates the feature store (with a local backup fallback for resilience)
- **`retrain_models_daily.yml`** — rebuilds the daily feature table and retrains/re-evaluates all models once a day

To run these on your own fork, add `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, and `HOPSWORKS_HOST` as repository secrets under **Settings → Secrets and variables → Actions**.

## ⚠️ Known Limitations

- Day+3 forecast accuracy is likely constrained primarily by training data volume, not model choice
- The system depends on Hopsworks' free tier, whose reliability was a recurring challenge during development (documented in the accompanying technical report)
- GitHub Actions' scheduled-workflow timing is best-effort, not guaranteed, per GitHub's own documentation

## 📄 Documentation

A detailed technical report covering methodology, EDA findings, model selection, SHAP analysis, and a full engineering incident log is available in the project repository.

## 👤 Author

**Akshay Kumar**
10Pearls Shine Internship Program — Data Science
