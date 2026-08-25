"""
Tests that feature column lists stay in sync across feature_engineering.py,
train_models.py, and app.py. A mismatch here has caused real bugs in this
project before (forecasts silently built from the wrong columns) — this
test exists specifically to catch that class of bug automatically, before
it reaches the live dashboard.
"""
import pandas as pd
import train_models
import app as dashboard_app
import feature_engineering


def test_base_features_match_between_train_and_app():
    assert train_models.BASE_FEATURES == dashboard_app.BASE_FEATURES, (
        "train_models.py and app.py's BASE_FEATURES have drifted apart — "
        "the dashboard's live inference must build the exact same columns "
        "the model was trained on."
    )


def test_weather_vars_match_between_train_and_app():
    assert train_models.WEATHER_VARS == dashboard_app.WEATHER_VARS


def test_weather_vars_match_feature_engineering():
    assert train_models.WEATHER_VARS == feature_engineering.WEATHER_VARS


def test_feature_engineering_output_matches_base_features():
    """Runs the real feature_engineering.py pipeline on synthetic hourly
    data and checks the resulting daily columns exactly match
    BASE_FEATURES — the strongest possible check that the two files agree."""
    hours = pd.date_range("2026-01-01", periods=24 * 10, freq="h")
    raw = pd.DataFrame({
        "time": hours,
        "us_aqi": range(len(hours)),
        "pm2_5": 20.0,
        "pm10": 40.0,
        "carbon_monoxide": 300.0,
        "nitrogen_dioxide": 15.0,
        "sulphur_dioxide": 10.0,
        "ozone": 50.0,
        "temperature_2m": 28.0,
        "relative_humidity_2m": 60.0,
        "wind_speed_10m": 10.0,
        "wind_direction_10m": 180.0,
        "surface_pressure": 1005.0,
        "precipitation": 0.0,
        "shortwave_radiation": 300.0,
    })

    daily = feature_engineering.aggregate_daily(raw)
    daily = feature_engineering.add_time_features(daily)
    daily = feature_engineering.add_derived_features(daily)
    daily = feature_engineering.add_target_day_weather(daily)
    daily = feature_engineering.add_targets(daily)
    daily_clean = daily.dropna().reset_index(drop=True)

    non_feature_cols = {"date"}
    non_feature_cols |= {f"target_day{h}" for h in (1, 2, 3)}
    non_feature_cols |= {
        f"{var}_h{h}" for var in feature_engineering.WEATHER_VARS for h in (1, 2, 3)
    }
    produced_base_features = set(daily_clean.columns) - non_feature_cols

    assert produced_base_features == set(train_models.BASE_FEATURES), (
        f"feature_engineering.py produces columns that don't match "
        f"train_models.py's BASE_FEATURES.\n"
        f"Only in feature_engineering output: "
        f"{produced_base_features - set(train_models.BASE_FEATURES)}\n"
        f"Only in BASE_FEATURES: "
        f"{set(train_models.BASE_FEATURES) - produced_base_features}"
    )