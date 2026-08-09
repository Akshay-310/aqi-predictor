# src/backfill.py
"""
Pulls 1 year of historical hourly air quality + weather data for Karachi
from Open-Meteo and writes it directly into the Hopsworks feature store
as the 'aqi_raw_hourly' feature group.
"""
import os
import requests
import pandas as pd
import hopsworks
from datetime import date, timedelta
from dotenv import load_dotenv
from preprocessing import clean_pipeline

load_dotenv()

LAT, LON = 24.8607, 67.0011  # Karachi
TIMEZONE = "Asia/Karachi"

END_DATE = date.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=730)

AIR_QUALITY_VARS = "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,shortwave_radiation"


def fetch_air_quality() -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": AIR_QUALITY_VARS,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])


def fetch_weather() -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": WEATHER_VARS,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])


def connect_to_hopsworks():
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
    )
    return project.get_feature_store()


def main():
    print(f"Fetching data from {START_DATE} to {END_DATE}...")

    aq_df = fetch_air_quality()
    weather_df = fetch_weather()

    merged = pd.merge(aq_df, weather_df, on="time", how="inner")
    merged["time"] = pd.to_datetime(merged["time"])
    merged = clean_pipeline(merged)

    print(f"Merged shape: {merged.shape}")
    print(f"Missing values:\n{merged.isna().sum()}")

    fs = connect_to_hopsworks()

    raw_fg = fs.get_or_create_feature_group(
        name="aqi_raw_hourly",
        version=1,
        description="Raw hourly weather + air quality data for Karachi from Open-Meteo",
        primary_key=["time"],
        event_time="time",
        time_travel_format="HUDI",
    )
    raw_fg.insert(merged)

    print("Successfully inserted into Hopsworks feature group: aqi_raw_hourly")


if __name__ == "__main__":
    main()