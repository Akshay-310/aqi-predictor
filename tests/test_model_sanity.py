"""
Sanity-checks that each horizon's currently trained model produces a
plausible AQI prediction. Loads directly from the local model files
(models_local/) rather than Hopsworks, so this test is fast and doesn't
depend on network/Hopsworks availability.
"""
import os
import joblib
import pandas as pd
import pytest
import train_models

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_BASE = {
    "us_aqi": 90, "pm2_5": 35, "pm10": 60, "carbon_monoxide": 400,
    "nitrogen_dioxide": 20, "sulphur_dioxide": 15, "ozone": 60,
    "temperature_2m": 28, "relative_humidity_2m": 55, "wind_speed_10m": 12,
    "wind_direction_10m": 200, "surface_pressure": 1005, "precipitation": 0,
    "shortwave_radiation": 350, "day_of_week": 2, "month": 1, "day_of_year": 15,
    "aqi_change_rate": 2.0, "aqi_roll3": 88, "aqi_roll7": 85, "pm25_roll3": 33,
}
SAMPLE_WEATHER = {
    "temperature_2m": 27, "relative_humidity_2m": 58, "wind_speed_10m": 10,
    "wind_direction_10m": 190, "surface_pressure": 1004, "precipitation": 0,
    "shortwave_radiation": 340,
}


@pytest.mark.parametrize("horizon,horizon_name", [(1, "day1"), (2, "day2"), (3, "day3")])
def test_model_prediction_is_plausible_aqi(horizon, horizon_name):
    model_path = os.path.join(PROJECT_ROOT, "models_local", horizon_name, "model.pkl")
    if not os.path.exists(model_path):
        pytest.skip(f"No local model file for {horizon_name} — run train_models.py first")

    model = joblib.load(model_path)

    row = dict(SAMPLE_BASE)
    for var, val in SAMPLE_WEATHER.items():
        row[f"{var}_h{horizon}"] = val

    feature_cols = train_models.BASE_FEATURES + [
        f"{v}_h{horizon}" for v in train_models.WEATHER_VARS
    ]
    X = pd.DataFrame([row])[feature_cols]

    prediction = model.predict(X)[0]

    assert prediction == prediction, "Prediction is NaN"  # NaN check
    assert -50 <= prediction <= 600, (
        f"{horizon_name} predicted {prediction:.1f}, well outside plausible AQI range"
    )