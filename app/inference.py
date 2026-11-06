"""Loads the champion once at start-up; preprocessing matches training exactly."""
import os
import time

import requests
import tensorflow as tf

IMG_SIZE = 160
MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
MODEL_API_URL = os.getenv("MODEL_API_URL", "").rstrip("/")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "VisionGuard-transfer_main2")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2")

# Loaded ONCE at import (start-up), never per request.
_model = None if MODEL_API_URL else tf.keras.models.load_model(MODEL_PATH)


class BadImage(ValueError):
    pass


def preprocess(data: bytes):
    """Same steps as src/data.py: decode to 3-channel RGB, antialiased resize, 0..255 floats.
    Scaling to the model's expected range happens inside the model."""
    try:
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError as e:
        raise BadImage("File could not be decoded as an image.") from e
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)
    return tf.cast(img, tf.float32)[None]


def classify(data: bytes) -> dict:
    t0 = time.perf_counter()
    if MODEL_API_URL:                                   # hosted API mode (Task 8)
        r = requests.post(f"{MODEL_API_URL}/predict", files={"image": ("upload", data)},
                          headers={"X-API-Key": MODEL_API_KEY}, timeout=60)
        if r.status_code == 422:
            raise BadImage(r.json().get("error", "Image rejected by the model API."))
        r.raise_for_status()
        out = r.json()
        out["mode"] = "remote"
    else:                                               # local mode (Task 4)
        p = float(_model.predict(preprocess(data), verbose=0)[0, 0])
        is_ai = p >= THRESHOLD
        out = {"label": "AI-generated" if is_ai else "Real",
               "prob_ai": p, "confidence": p if is_ai else 1 - p,
               "threshold": THRESHOLD, "model_name": MODEL_NAME,
               "model_version": MODEL_VERSION, "mode": "local"}
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return out