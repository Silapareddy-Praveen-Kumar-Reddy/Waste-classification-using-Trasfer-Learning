import logging
import os

import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
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
# Model Loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "vgg16.h5")

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
    model = None

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

    if model is None:
        flash("Model is not available. Please try again later.", "error")
        return redirect(url_for("predict"))

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

    prediction = model.predict(processed_image, verbose=0)
    prediction_details = get_prediction_details(prediction[0], WASTE_CLASSES)
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
# Routes — REST API (JSON)
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint — returns JSON prediction results."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

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

    prediction = model.predict(processed_image, verbose=0)
    prediction_details = get_prediction_details(prediction[0], WASTE_CLASSES)
    recycling_info = get_recycling_instructions(prediction_details["predicted_class"])

    return jsonify({
        "prediction": prediction_details,
        "recycling_info": recycling_info,
    }), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint with model status."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "waste_classes": WASTE_CLASSES,
    }), 200


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 2222))
    app.run(debug=debug_mode, port=port)
