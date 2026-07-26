"""
app_tfserving.py
----------------
Step 3 variant of app.py:
  - Removes in-process TensorFlow / Keras model loading
  - Replaces model.predict() with an HTTP call to TF Serving REST API
  - Preserves the exact same JSON response schema as the original app.py
  - Original app.py is left untouched (used by Render deployment)

Environment variables:
  TF_SERVING_URL  URL of the TF Serving service (default: http://tf-serving:8501)
  SECRET_KEY      Flask secret key
  PORT            Server port (default: 8080)
"""

import logging
import os
import uuid

import numpy as np
import requests as http_requests
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# TF Serving configuration
# ---------------------------------------------------------------------------
TF_SERVING_URL = os.environ.get("TF_SERVING_URL", "http://tf-serving:8501")
TF_SERVING_PREDICT_URL = f"{TF_SERVING_URL}/v1/models/waste_classifier:predict"

WASTE_CLASSES = ["Biodegradable", "Recyclable", "Trash"]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Return True if *filename* has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: str, target_size: tuple = (224, 224)):
    """Load and preprocess an image for VGG16 inference."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as exc:
        logger.error("Image preprocessing error: %s", exc)
        return None


def call_tf_serving(preprocessed_image) -> list | None:
    """
    Send preprocessed image tensor to TF Serving and return raw predictions.

    TF Serving REST payload (column / instances format):
        POST /v1/models/<name>:predict
        {
          "signature_name": "serve",   ← Keras 3 export() uses 'serve', not 'serving_default'
          "instances": [ [[...pixel data...]] ]
        }

    Response:
        { "predictions": [ [p0, p1, p2] ] }
    """
    try:
        payload = {
            "signature_name": "serve",     # Keras 3 model.export() signature name
            "instances": preprocessed_image.tolist(),
        }
        resp = http_requests.post(TF_SERVING_PREDICT_URL, json=payload, timeout=30)
        resp.raise_for_status()
        predictions = resp.json()["predictions"]
        return predictions[0]   # first (and only) sample
    except Exception as exc:
        logger.error("TF Serving call failed: %s", exc)
        return None


def get_prediction_details(prediction_array, classes: list) -> dict:
    """Extract human-readable prediction details from raw model output."""
    predicted_class_idx = int(np.argmax(prediction_array))
    predicted_class = (
        classes[predicted_class_idx]
        if predicted_class_idx < len(classes)
        else "Unknown"
    )
    confidence = float(prediction_array[predicted_class_idx])
    all_probabilities = {
        classes[i] if i < len(classes) else f"Class_{i}": round(float(prediction_array[i]), 4)
        for i in range(len(prediction_array))
    }
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "all_probabilities": all_probabilities,
    }


RECYCLING_INSTRUCTIONS = {
    "Biodegradable": {
        "disposal": "Use compost bin or biodegradable waste service.",
        "tips": "Can be composted at home or through city programs.",
        "environmental_impact": "Low — breaks down naturally.",
    },
    "Recyclable": {
        "disposal": "Use local recycling bin or station.",
        "tips": "Clean before disposal and sort properly.",
        "environmental_impact": "Medium — recyclable but must be cleaned.",
    },
    "Trash": {
        "disposal": "Place in general waste.",
        "tips": "Avoid if recyclable options exist.",
        "environmental_impact": "High — contributes to landfill.",
    },
}

DEFAULT_INSTRUCTIONS = {
    "disposal": "Check local waste guidelines.",
    "tips": "Sort based on material.",
    "environmental_impact": "Variable",
}


def get_recycling_instructions(waste_class: str) -> dict:
    """Return disposal instructions for *waste_class*."""
    return RECYCLING_INSTRUCTIONS.get(waste_class, DEFAULT_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    if "file" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("predict"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Invalid or no file selected. Allowed: png, jpg, jpeg, bmp", "error")
        return redirect(url_for("predict"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    processed_image = preprocess_image(filepath)
    if processed_image is None:
        flash("Error processing image.", "error")
        return redirect(url_for("predict"))

    raw_prediction = call_tf_serving(processed_image)
    if raw_prediction is None:
        flash("Inference service unavailable. Please try again later.", "error")
        return redirect(url_for("predict"))

    prediction_details = get_prediction_details(raw_prediction, WASTE_CLASSES)
    recycling_info = get_recycling_instructions(prediction_details["predicted_class"])

    result_data = {
        "filename": filename,
        "filepath": url_for("static", filename=f"uploads/{filename}"),
        "prediction": prediction_details,
        "recycling_info": recycling_info,
    }
    return render_template("portfolio.html", result=result_data)


@app.route("/blog")
def blog():
    return render_template("blog.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ---------------------------------------------------------------------------
# Routes — REST API (JSON) — synchronous
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint — returns JSON prediction results (synchronous)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, bmp"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    processed_image = preprocess_image(filepath)
    if processed_image is None:
        return jsonify({"error": "Failed to process image"}), 422

    raw_prediction = call_tf_serving(processed_image)
    if raw_prediction is None:
        return jsonify({"error": "Inference service unavailable"}), 503

    prediction_details = get_prediction_details(raw_prediction, WASTE_CLASSES)
    recycling_info = get_recycling_instructions(prediction_details["predicted_class"])

    # Identical response schema to original app.py
    return jsonify({
        "prediction": prediction_details,
        "recycling_info": recycling_info,
    }), 200


# ---------------------------------------------------------------------------
# Routes — REST API (JSON) — async (Step 4)
# ---------------------------------------------------------------------------

@app.route("/api/predict/async", methods=["POST"])
def api_predict_async():
    """
    Async prediction endpoint (Step 4).
    Publishes image reference to Kafka topic 'waste-jobs'.
    Returns request_id immediately; poll /api/result/<request_id> for the result.
    """
    # Lazy import so the sync path works even without kafka-python installed
    try:
        from kafka import KafkaProducer  # noqa: F401
        import json as _json
    except ImportError:
        return jsonify({"error": "Async queue not configured (kafka-python not installed)"}), 501

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, bmp"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    request_id = str(uuid.uuid4())
    kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092")

    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: _json.dumps(v).encode("utf-8"),
        )
        producer.send("waste-jobs", {"request_id": request_id, "filepath": filepath})
        producer.flush()
    except Exception as exc:
        logger.error("Kafka publish failed: %s", exc)
        return jsonify({"error": "Failed to queue job"}), 503

    return jsonify({"request_id": request_id, "status": "queued"}), 202


@app.route("/api/result/<request_id>", methods=["GET"])
def api_result(request_id: str):
    """
    Poll for async prediction result (Step 4).
    Returns result JSON if ready, or { "status": "pending" } if still processing.
    """
    import sqlite3

    db_path = os.environ.get("RESULTS_DB", "results.db")
    if not os.path.exists(db_path):
        return jsonify({"status": "pending", "request_id": request_id}), 202

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT result FROM results WHERE request_id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
    except Exception as exc:
        logger.error("DB read error: %s", exc)
        return jsonify({"error": "Database error"}), 500

    if row is None:
        return jsonify({"status": "pending", "request_id": request_id}), 202

    import json as _json
    result = _json.loads(row[0])
    return jsonify({"status": "complete", "request_id": request_id, **result}), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint — pings TF Serving to verify it's reachable."""
    tf_serving_ok = False
    try:
        resp = http_requests.get(
            f"{TF_SERVING_URL}/v1/models/waste_classifier",
            timeout=5,
        )
        tf_serving_ok = resp.status_code == 200
    except Exception:
        pass

    return jsonify({
        "status": "healthy",
        "model_loaded": tf_serving_ok,
        "tf_serving_url": TF_SERVING_URL,
        "tf_serving_reachable": tf_serving_ok,
        "waste_classes": WASTE_CLASSES,
    }), 200


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=debug_mode, port=port)
