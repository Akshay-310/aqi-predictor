"""
Hourly live data collection: fetches recent weather + air quality data
from Open-Meteo, cleans it, and:
  1) ALWAYS writes it to a local running CSV backup (data/raw_backup/
     aqi_raw_hourly_live.csv) — independent of Hopsworks entirely, so
     the dashboard always has a fresh source even if Hopsworks'
     materialization jobs are stuck/failing.
  2) Also tries to insert into Hopsworks as the long-term store. If
     that insert call itself fails, the rows are additionally saved to
     a separate pending-retry file and flushed on the next successful
     run — this is unrelated to materialization health, only covers
     the insert() call itself failing.

Fetches a wide (~60 hour) trailing window rather than just the last 2
hours, deliberately — GitHub Actions' scheduled triggers have proven
unreliable (observed firing only 5-7 times/day instead of hourly), so
each successful run needs to be able to catch up everything missed
since the last one, not just the most recent couple of hours.
Deduplication (by timestamp) makes re-fetching overlapping hours safe.
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
LIVE_BACKUP_PATH = "data/raw_backup/aqi_raw_hourly_live.csv"
LIVE_BACKUP_MAX_ROWS = 24 * 14  # keep ~2 weeks locally — plenty for live
                                  # forecasting, keeps the file small

FETCH_PAST_DAYS = 3          # widened from 1 — gives ~72h of API-side history to draw from
CATCH_UP_WINDOW_HOURS = 60   # widened from 2 — safely covers even a full day of missed runs


def fetch_recent_air_quality() -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": AIR_QUALITY_VARS,
        "past_days": FETCH_PAST_DAYS,
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
        "past_days": FETCH_PAST_DAYS,
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


def update_live_backup(df: pd.DataFrame) -> None:
    """Always writes the fetched rows to a running local CSV, regardless
    of what happens with Hopsworks below. This is what makes the
    dashboard's 'current AQI' and forecast independent of Hopsworks'
    materialization health entirely."""
    os.makedirs(os.path.dirname(LIVE_BACKUP_PATH), exist_ok=True)
    if os.path.exists(LIVE_BACKUP_PATH):
        existing = pd.read_csv(LIVE_BACKUP_PATH, parse_dates=["time"])
        combined = pd.concat([existing, df]).drop_duplicates(subset="time").sort_values("time")
    else:
        combined = df.sort_values("time")

    if len(combined) > LIVE_BACKUP_MAX_ROWS:
        combined = combined.tail(LIVE_BACKUP_MAX_ROWS)

    combined.to_csv(LIVE_BACKUP_PATH, index=False)
    print(f"Live backup updated: {len(combined)} rows, latest = {combined['time'].max()}")


def save_pending(df: pd.DataFrame) -> None:
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
    print("Fetching recent hourly data (wide catch-up window)...")

    aq_df = fetch_recent_air_quality()
    weather_df = fetch_recent_weather()

    merged = pd.merge(aq_df, weather_df, on="time", how="inner")
    merged["time"] = pd.to_datetime(merged["time"])

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")
    merged = merged[(merged["time"] < now) & (merged["time"] >= now - pd.Timedelta(hours=CATCH_UP_WINDOW_HOURS))]

    if merged.empty:
        print("No new rows to insert for this run.")
        return

    merged = clean_pipeline(merged)

    # ALWAYS update the live backup, independent of Hopsworks' status below.
    update_live_backup(merged)

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

        flush_pending(raw_fg)

    except Exception as e:
        print(f"WARNING: Hopsworks insert failed ({e}) — saving locally instead.")
        save_pending(merged)


if __name__ == "__main__":
    main()