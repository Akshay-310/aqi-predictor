# src/fetch_current_data.py
"""
Hourly live data collection: fetches the most recent 1-2 hours of
weather + air quality data from Open-Meteo, cleans it, and inserts new
rows into the aqi_raw_hourly feature group.

If the Hopsworks insert fails (e.g. spending-cap throttling), the
fetched rows are saved to a local pending-upload CSV instead of being
lost. The next time this script runs successfully, it retries any
pending rows first, then clears the pending file.
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

STALENESS_CHECK_HOURS = 5
PENDING_BACKUP_PATH = "data/raw_backup/pending_upload.csv"


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
    try:
        raw_fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
        existing = raw_fg.read()
        existing["time"] = pd.to_datetime(existing["time"])
        existing = existing.sort_values("time").tail(STALENESS_CHECK_HOURS)

        recent_values = list(existing[column]) + list(new_data[column])
        recent_values = recent_values[-STALENESS_CHECK_HOURS:]

        if len(set(recent_values)) == 1 and len(recent_values) == STALENESS_CHECK_HOURS:
            print(f"WARNING: '{column}' has been identical ({recent_values[0]}) for the last "
                  f"{STALENESS_CHECK_HOURS} hours — possible upstream staleness.")
        else:
            print(f"Staleness check passed — '{column}' shows variation across recent hours")
    except Exception as e:
        print(f"Staleness check skipped (couldn't read existing data): {e}")


def save_pending(df: pd.DataFrame) -> None:
    """Rows that failed to reach Hopsworks get parked here instead of lost."""
    os.makedirs(os.path.dirname(PENDING_BACKUP_PATH), exist_ok=True)
    if os.path.exists(PENDING_BACKUP_PATH):
        existing = pd.read_csv(PENDING_BACKUP_PATH, parse_dates=["time"])
        combined = pd.concat([existing, df]).drop_duplicates(subset="time").sort_values("time")
    else:
        combined = df.sort_values("time")
    combined.to_csv(PENDING_BACKUP_PATH, index=False)
    print(f"WARNING: Hopsworks insert failed — saved {len(df)} row(s) locally. "
          f"Pending backup now holds {len(combined)} row(s) waiting to sync.")


def flush_pending(raw_fg) -> None:
    """If a previous run failed and left rows pending, retry them now that
    Hopsworks is reachable, then clear the pending file."""
    if not os.path.exists(PENDING_BACKUP_PATH):
        return
    pending = pd.read_csv(PENDING_BACKUP_PATH, parse_dates=["time"])
    if pending.empty:
        os.remove(PENDING_BACKUP_PATH)
        return
    print(f"Found {len(pending)} pending row(s) from a previous failed run — retrying...")
    raw_fg.insert(pending)
    os.remove(PENDING_BACKUP_PATH)
    print("Pending backup synced successfully and cleared.")


def main():
    print("Fetching current hourly data...")

    aq_df = fetch_recent_air_quality()
    weather_df = fetch_recent_weather()

    merged = pd.merge(aq_df, weather_df, on="time", how="inner")
    merged["time"] = pd.to_datetime(merged["time"])

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")
    merged = merged[(merged["time"] < now) & (merged["time"] >= now - pd.Timedelta(hours=2))]

    if merged.empty:
        print("No new rows to insert for this run.")
        return

    merged = clean_pipeline(merged)

    try:
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

        # This run reached Hopsworks fine — also flush anything stranded earlier
        flush_pending(raw_fg)

    except Exception as e:
        print(f"WARNING: Hopsworks insert failed ({e}) — saving locally instead.")
        save_pending(merged)


if __name__ == "__main__":
    main()