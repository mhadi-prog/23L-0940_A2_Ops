"""Download the champion once, save it locally so the app never calls DagsHub per request."""
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()
Path("models").mkdir(exist_ok=True)
m = mlflow.keras.load_model("models:/VisionGuard@champion")
m.save("models/champion.keras")
r = mlflow.get_run("4434eed03f5b4e018db5af2d2a16d13c")
print("threshold", r.data.params.get("threshold"))
