# src/fetch_current_data.py
"""
Hourly live data collection: fetches the most recent 1-2 hours of
weather + air quality data from Open-Meteo, cleans it, checks for
staleness (flags if a value repeats identically across consecutive
hours, which can happen if the upstream source hasn't refreshed),
and inserts new rows into the aqi_raw_hourly feature group. Designed
to run hourly via GitHub Actions.
"""
import os
import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from preprocessing import clean_pipeline

load_dotenv()

LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"

AIR_QUALITY_VARS = "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,shortwave_radiation"

STALENESS_CHECK_HOURS = 5  # if this many consecutive us_aqi readings are identical, flag it


def fetch_recent_air_quality() -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": AIR_QUALITY_VARS,
        "past_days": 1,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])


def fetch_recent_weather() -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": WEATHER_VARS,
        "past_days": 1,
        "forecast_days": 1,
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


def check_staleness(fs, new_data: pd.DataFrame, column: str = "us_aqi") -> None:
    """
    Compares newly fetched values against the most recent rows already
    stored, to catch a value repeating unexpectedly across several
    hours (verified against Open-Meteo's live API: hourly us_aqi values
    do vary hour to hour under normal conditions, so a run of identical
    values is a genuine signal worth flagging, not expected behavior).
    Logs a warning only — doesn't block insertion, since we're
    single-source now and have no fallback to switch to.
    """
    try:
        raw_fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
        existing = raw_fg.read()
        existing["time"] = pd.to_datetime(existing["time"])
        existing = existing.sort_values("time").tail(STALENESS_CHECK_HOURS)

        recent_values = list(existing[column]) + list(new_data[column])
        recent_values = recent_values[-STALENESS_CHECK_HOURS:]

        if len(set(recent_values)) == 1 and len(recent_values) == STALENESS_CHECK_HOURS:
            print(f"WARNING: '{column}' has been identical ({recent_values[0]}) for the last "
                  f"{STALENESS_CHECK_HOURS} hours — possible stale CAMS update cycle. "
                  f"Data still being inserted, but flagging for awareness.")
        else:
            print(f"Staleness check passed — '{column}' shows variation across recent hours")
    except Exception as e:
        print(f"Staleness check skipped (couldn't read existing data): {e}")


def main():
    print("Fetching current hourly data...")

    aq_df = fetch_recent_air_quality()
    weather_df = fetch_recent_weather()

    merged = pd.merge(aq_df, weather_df, on="time", how="inner")
    merged["time"] = pd.to_datetime(merged["time"])

    # Keep only the last 2 completed hours — avoids inserting the
    # current, possibly-incomplete hour, and avoids re-inserting a huge
    # backlog every run.
    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")
    merged = merged[(merged["time"] < now) & (merged["time"] >= now - pd.Timedelta(hours=2))]

    if merged.empty:
        print("No new rows to insert for this run.")
        return

    merged = clean_pipeline(merged)

    fs = connect_to_hopsworks()
    check_staleness(fs, merged)

    raw_fg = fs.get_or_create_feature_group(
        name="aqi_raw_hourly",
        version=1,
        description="Raw hourly weather + air quality data for Karachi from Open-Meteo",
        primary_key=["time"],
        event_time="time",
        time_travel_format="HUDI",
    )
    raw_fg.insert(merged)

    print(f"Successfully inserted {len(merged)} new row(s) into aqi_raw_hourly")


if __name__ == "__main__":
    main()