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
fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
job = fg.materialization_job

print("Current state before:", job.get_state())
job.stop()
print("Stopped. Triggering a fresh run...")
job.run(await_termination=False)
print("New state:", job.get_state())