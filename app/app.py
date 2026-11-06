import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

load_dotenv()
from app import inference  # noqa: E402  (after env is loaded)

ALLOWED = {"jpg", "jpeg", "png", "webp"}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
db = SQLAlchemy(app)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    filename = db.Column(db.String(255))
    label = db.Column(db.String(32))
    confidence = db.Column(db.Float)
    prob_ai = db.Column(db.Float)
    threshold = db.Column(db.Float)
    model_name = db.Column(db.String(128))
    model_version = db.Column(db.String(32))
    mode = db.Column(db.String(16))
    latency_ms = db.Column(db.Integer)


with app.app_context():
    db.create_all()


def error(msg, code=400):
    return render_template("error.html", message=msg, code=code), code


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None or not f.filename:
        return error("Please choose an image file.")
    name = secure_filename(f.filename) or "upload"
    if name.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return error(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED))}.")
    data = f.read()                       # kept in memory only, never written to disk
    try:
        res = inference.classify(data)
    except inference.BadImage as e:
        return error(str(e))
    except Exception:
        app.logger.exception("inference failed")
        return error("The model service is unavailable. Please try again.", 502)
    db.session.add(Prediction(filename=name, **{k: res[k] for k in (
        "label", "confidence", "prob_ai", "threshold", "model_name",
        "model_version", "mode", "latency_ms")}))
    db.session.commit()
    return render_template("result.html", r=res, filename=name)


@app.get("/history")
def history():
    rows = Prediction.query.order_by(Prediction.id.desc()).limit(200).all()
    return render_template("history.html", rows=rows)


@app.get("/health")
def health():
    return jsonify(status="ok", mode="remote" if inference.MODEL_API_URL else "local",
                   model=inference.MODEL_NAME, version=inference.MODEL_VERSION,
                   threshold=inference.THRESHOLD, debug=app.debug)


@app.errorhandler(413)
def too_large(_):
    return error(f"File too large. The limit is {MAX_MB} MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return error("Page not found.", 404)


@app.errorhandler(500)
def server_error(_):
    return error("Something went wrong on our side.", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5001")), debug=False)