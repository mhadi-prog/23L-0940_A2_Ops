"""3.6: load the model by alias and classify one image."""
import sys
import mlflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
model = mlflow.keras.load_model("models:/VisionGuard@champion")
img = Image.open(sys.argv[1]).convert("RGB").resize((160, 160))
x = np.asarray(img, dtype="float32")[None]          # 0..255, scaling is inside the model
p = float(model.predict(x, verbose=0)[0, 0])
print(f"P(AI) = {p:.3f} -> {'AI-generated' if p >= 0.5 else 'real'}")