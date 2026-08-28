import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

LIVE_BACKUP_PATH = "data/raw_backup/aqi_raw_hourly_live.csv"
KARACHI_TZ = "Asia/Karachi"
DAILY_AGG = {
    "us_aqi": "mean", "pm2_5": "mean", "pm10": "mean",
    "carbon_monoxide": "mean", "nitrogen_dioxide": "mean",
    "sulphur_dioxide": "mean", "ozone": "mean",
    "temperature_2m": "mean", "relative_humidity_2m": "mean",
    "wind_speed_10m": "mean", "wind_direction_10m": "mean",
    "surface_pressure": "mean", "precipitation": "sum",
    "shortwave_radiation": "mean",
}

frames = []
try:
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
    )
    fs = project.get_feature_store()
    hw_df = fs.get_feature_group(name="aqi_raw_hourly", version=1).read()
    hw_df["time"] = pd.to_datetime(hw_df["time"])
    if hw_df["time"].dt.tz is not None:
        hw_df["time"] = hw_df["time"].dt.tz_localize(None)
    frames.append(hw_df)
    print(f"Hopsworks raw_hourly: {len(hw_df)} rows, latest = {hw_df['time'].max()}")
except Exception as e:
    print(f"Could not read Hopsworks: {e}")

if os.path.exists(LIVE_BACKUP_PATH):
    local_df = pd.read_csv(LIVE_BACKUP_PATH, parse_dates=["time"])
    frames.append(local_df)
    print(f"Local live backup: {len(local_df)} rows, latest = {local_df['time'].max()}")
else:
    print("No local live backup file found")

combined = pd.concat(frames, ignore_index=True)
combined = combined.drop_duplicates(subset="time", keep="last")
combined = combined.sort_values("time").reset_index(drop=True)
print(f"\nMerged: {len(combined)} rows, latest = {combined['time'].max()}")

combined["date"] = combined["time"].dt.date
daily = combined.groupby("date").agg(DAILY_AGG).reset_index()
daily["date"] = pd.to_datetime(daily["date"])

today_local = pd.Timestamp.now(tz=KARACHI_TZ).tz_localize(None).normalize()
daily_complete = daily[daily["date"] < today_local]

print(f"\nDays of complete history available (before today): {len(daily_complete)}")
print(f"Need at least 8 for a forecast (7-day rolling window)")
if len(daily_complete) > 0:
    print(f"Date range: {daily_complete['date'].min().date()} to {daily_complete['date'].max().date()}")