# src/feature_engineering.py
"""
Reads raw hourly data from the 'aqi_raw_hourly' Hopsworks feature group,
aggregates to daily, engineers features + horizon targets, and writes
the result into the 'aqi_daily_features' feature group.
"""
import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()


def connect_to_hopsworks():
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
    )
    return project.get_feature_store()


def load_raw_data(fs) -> pd.DataFrame:
    raw_fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
    df = raw_fg.read()
    df["time"] = pd.to_datetime(df["time"])
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df["time"].dt.date
    daily = df.groupby("date").agg({
        "us_aqi": "mean", "pm2_5": "mean", "pm10": "mean",
        "carbon_monoxide": "mean", "nitrogen_dioxide": "mean",
        "sulphur_dioxide": "mean", "ozone": "mean",
        "temperature_2m": "mean", "relative_humidity_2m": "mean",
        "wind_speed_10m": "mean", "wind_direction_10m": "mean",
        "surface_pressure": "mean", "precipitation": "sum",
        "shortwave_radiation": "mean",
    }).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["aqi_change_rate"] = df["us_aqi"].diff()
    df["aqi_roll3"] = df["us_aqi"].shift(1).rolling(window=3).mean()
    df["aqi_roll7"] = df["us_aqi"].shift(1).rolling(window=7).mean()
    df["pm25_roll3"] = df["pm2_5"].shift(1).rolling(window=3).mean()
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["target_day1"] = df["us_aqi"].shift(-1)
    df["target_day2"] = df["us_aqi"].shift(-2)
    df["target_day3"] = df["us_aqi"].shift(-3)
    return df


def main():
    fs = connect_to_hopsworks()

    raw = load_raw_data(fs)
    print(f"Loaded raw hourly rows: {raw.shape}")

    daily = aggregate_daily(raw)
    daily = add_time_features(daily)
    daily = add_derived_features(daily)
    daily = add_targets(daily)

    before = len(daily)
    daily_clean = daily.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(daily_clean)} edge rows (no target/history)")
    print(f"Final feature table shape: {daily_clean.shape}")

    features_fg = fs.get_or_create_feature_group(
        name="aqi_daily_features",
        version=1,
        description="Daily aggregated AQI features with time features, rolling stats, and day+1/2/3 targets",
        primary_key=["date"],
        event_time="date",
        time_travel_format="HUDI",
    )
    features_fg.insert(daily_clean)

    print("Successfully inserted into Hopsworks feature group: aqi_daily_features")


if __name__ == "__main__":
    main()