"""VisionGuard model API: POST /predict (needs X-API-Key), GET /health."""
import hmac
import os

import tensorflow as tf
from flask import Flask, jsonify, request

IMG_SIZE = 160
MODEL_PATH = os.getenv("MODEL_PATH", "champion.keras")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "VisionGuard-transfer_main2")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2")

model = tf.keras.models.load_model(MODEL_PATH)   # loaded once at start-up

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, version=MODEL_VERSION, threshold=THRESHOLD)


@app.post("/predict")
def predict():
    if not API_KEY:                                           # fail closed
        return jsonify(error="Server has no API key configured."), 500
    sent = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(sent.encode(), API_KEY.encode()):
        return jsonify(error="Invalid or missing API key."), 401
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Send the file in a form field named 'image'."), 400
    try:
        img = tf.io.decode_image(f.read(), channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError:
        return jsonify(error="File could not be decoded as an image."), 422
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)
    p = float(model.predict(tf.cast(img, tf.float32)[None], verbose=0)[0, 0])
    is_ai = p >= THRESHOLD
    return jsonify(label="AI-generated" if is_ai else "Real", prob_ai=p,
                   confidence=p if is_ai else 1 - p, threshold=THRESHOLD,
                   model_name=MODEL_NAME, model_version=MODEL_VERSION)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large (max 10 MB)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860)