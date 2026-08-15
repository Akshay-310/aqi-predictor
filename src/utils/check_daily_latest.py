import os
import pandas as pd
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
df = fs.get_feature_group(name="aqi_daily_features", version=2).read()
df["date"] = pd.to_datetime(df["date"])
print(df[["date"]].sort_values("date").tail(5))