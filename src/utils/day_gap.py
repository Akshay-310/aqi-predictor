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
print(f"Hopsworks: {len(hw_df)} rows, {hw_df['time'].min()} to {hw_df['time'].max()}")

if os.path.exists(LIVE_BACKUP_PATH):
    local_df = pd.read_csv(LIVE_BACKUP_PATH, parse_dates=["time"])
    frames.append(local_df)
    print(f"Local backup: {len(local_df)} rows, {local_df['time'].min()} to {local_df['time'].max()}")
else:
    print("No local backup file found")

combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time", keep="last")
combined["date"] = combined["time"].dt.date
counts = combined.groupby("date").size().sort_index()

print("\nHourly coverage, last 14 days:")
for date, count in counts.tail(14).items():
    flag = "  <-- INCOMPLETE" if count < 18 else ""
    print(f"  {date}: {count:2d} hours{flag}")