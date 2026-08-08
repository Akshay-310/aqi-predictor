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
print("Connected successfully to project:", project.name)
print("Feature store:", fs.name)