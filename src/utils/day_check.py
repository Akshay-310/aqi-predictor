import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

LIVE_BACKUP_PATH = "data/raw_backup/aqi_raw_hourly_live.csv"
KARACHI_TZ = "Asia/Karachi"

frames = []
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

if os.path.exists(LIVE_BACKUP_PATH):
    local_df = pd.read_csv(LIVE_BACKUP_PATH, parse_dates=["time"])
    frames.append(local_df)

combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time", keep="last")
combined = combined.sort_values("time").reset_index(drop=True)

today_local = pd.Timestamp.now(tz=KARACHI_TZ).tz_localize(None).normalize()
yesterday = today_local - pd.Timedelta(days=1)
combined["date"] = combined["time"].dt.date

y_rows = combined[combined["date"] == yesterday.date()]
print(f"Hours of data for 'yesterday' ({yesterday.date()}): {len(y_rows)} (should be close to 24)")
print(f"Yesterday's average us_aqi: {y_rows['us_aqi'].mean():.1f}")
print(f"\nToday's forecasted weather (h1) vs tomorrow's (h2):")
print(y_rows[["wind_speed_10m", "temperature_2m", "relative_humidity_2m"]].mean())