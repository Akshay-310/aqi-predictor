# src/train_models.py
"""
Trains and compares 3 algorithms (Ridge, Random Forest, LightGBM) for each
of the 3 horizons. Each horizon's model uses base features PLUS the
actual/forecasted weather for that specific target day. Evaluates with
RMSE/MAE/R² on both train and test sets to detect overfitting, prefers
non-overfit models during selection, and registers the best-performing
algorithm per horizon in the Hopsworks Model Registry.
"""
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
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


def get_candidate_models():
    return {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
        ),
        "lightgbm": lgb.LGBMRegressor(
            objective="regression",
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            min_child_samples=15,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            verbose=-1,
        ),
    }


def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
    }


def train_and_compare(train_df, test_df, target_col, feature_cols, horizon_name):
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    results = {}
    print(f"\n=== {horizon_name} (using {len(feature_cols)} features) ===")
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        train_metrics = evaluate(model, X_train, y_train)
        test_metrics = evaluate(model, X_test, y_test)
        gap = train_metrics["r2"] - test_metrics["r2"]
        results[name] = {"model": model, "metrics": test_metrics, "gap": gap}

        overfit_flag = "  possible overfitting" if gap > OVERFIT_GAP_THRESHOLD else ""
        print(f"  {name:15s} Train R2={train_metrics['r2']:.3f}  Test R2={test_metrics['r2']:.3f}  Gap={gap:.3f}{overfit_flag}")
        print(f"  {'':15s} Test RMSE={test_metrics['rmse']:.2f}  Test MAE={test_metrics['mae']:.2f}")

    # Prefer models that aren't badly overfit; among those, pick highest test R2.
    # Only fall back to an overfit model if every candidate is flagged.
    healthy = {k: v for k, v in results.items() if v["gap"] <= OVERFIT_GAP_THRESHOLD}
    pool = healthy if healthy else results
    best_name = max(pool, key=lambda k: pool[k]["metrics"]["r2"])

    reason = " (chosen from non-overfit candidates)" if healthy else " (all candidates overfit, picked least-bad)"
    print(f"  --> Best for {horizon_name}: {best_name}{reason}")

    return best_name, results[best_name]["model"], results[best_name]["metrics"]


def register_model(project, model, horizon_name, algo_name, metrics):
    mr = project.get_model_registry()
    model_dir = f"models_local/{horizon_name}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")

    hw_model = mr.python.create_model(
        name=f"aqi_predictor_{horizon_name}",
        metrics=metrics,
        description=f"Best model ({algo_name}) for {horizon_name}, includes target-day weather features, overfit-checked",
    )
    hw_model.save(model_dir)
    print(f"Registered model: aqi_predictor_{horizon_name} (algorithm: {algo_name})")


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

        best_name, best_model, best_metrics = train_and_compare(
            train_df, test_df, target_col, feature_cols, horizon_name
        )
        register_model(project, best_model, horizon_name, best_name, best_metrics)
        summary.append((horizon_name, best_name, best_metrics))

    print("\n=== Final Summary ===")
    for horizon_name, algo, metrics in summary:
        print(f"{horizon_name}: {algo} | RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")


if __name__ == "__main__":
    main()