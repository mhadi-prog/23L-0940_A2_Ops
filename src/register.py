"""Log two runs' models to the MLflow registry, set alias 'champion'."""
import sys
import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
NAME = "VisionGuard"
# usage: python src/register.py <run_id_v1> <run_id_champion> <run_name_v1> <run_name_champion>
run_ids, run_names = sys.argv[1:3], sys.argv[3:5]
client = MlflowClient()
mlflow.set_experiment("23L-0940_A2_Ops")

versions = []
for rid, rname in zip(run_ids, run_names):
    local = mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=f"{rname}_final.keras")
    model = tf.keras.models.load_model(local)
    with mlflow.start_run(run_name=f"register_{rname}"):
        info = mlflow.keras.log_model(model, artifact_path="model", registered_model_name=NAME)
    versions.append(info)
    print("registered", rname)

latest = max(int(v.version) for v in client.search_model_versions(f"name='{NAME}'"))
client.set_registered_model_alias(NAME, "champion", str(latest))
print("champion ->", latest)