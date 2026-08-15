import pandas as pd
import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(
    host=os.getenv("HOPSWORKS_HOST"),
    project=os.getenv("HOPSWORKS_PROJECT"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    engine="python",
)
fs = project.get_feature_store()
df = fs.get_feature_group(name="aqi_raw_hourly", version=1).read()
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

print(df[["time", "us_aqi", "pm2_5"]].tail(10).to_string())