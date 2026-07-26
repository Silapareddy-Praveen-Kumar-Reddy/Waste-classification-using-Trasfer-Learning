"""
worker.py — Async inference worker (Step 4)
-------------------------------------------
Consumes image jobs from the Kafka topic 'waste-jobs',
calls TF Serving for each job, and writes results to SQLite (results.db).

Run with:
    python worker.py

Or via docker-compose (see docker-compose.yml).
"""

import json
import logging
import os
import sqlite3
import time

import numpy as np
import requests as http_requests
from kafka import KafkaConsumer
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] worker %(message)s",
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP  = os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092")
TF_SERVING_URL   = os.environ.get("TF_SERVING_URL", "http://tf-serving:8501")
TF_SERVING_PREDICT_URL = f"{TF_SERVING_URL}/v1/models/waste_classifier:predict"
RESULTS_DB       = os.environ.get("RESULTS_DB", "results.db")
TOPIC            = "waste-jobs"

WASTE_CLASSES = ["Biodegradable", "Recyclable", "Trash"]

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

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create the results table if it doesn't exist."""
    conn = sqlite3.connect(RESULTS_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            request_id TEXT PRIMARY KEY,
            result     TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info("SQLite DB ready at: %s", RESULTS_DB)


def write_result(request_id: str, result: dict):
    """Persist prediction result keyed by request_id."""
    conn = sqlite3.connect(RESULTS_DB)
    conn.execute(
        "INSERT OR REPLACE INTO results (request_id, result, created_at) VALUES (?, ?, ?)",
        (request_id, json.dumps(result), time.time()),
    )
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, target_size: tuple = (224, 224)):
    """Load and preprocess image for VGG16."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def call_tf_serving(preprocessed_image) -> list | None:
    """Call TF Serving and return raw probability list."""
    try:
        payload = {
            "signature_name": "serve",   # Keras 3 model.export() uses 'serve'
            "instances": preprocessed_image.tolist(),
        }
        resp = http_requests.post(TF_SERVING_PREDICT_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["predictions"][0]
    except Exception as exc:
        logger.error("TF Serving call failed: %s", exc)
        return None


def get_prediction_details(prediction_array) -> dict:
    """Build structured prediction dict from raw probabilities."""
    idx = int(np.argmax(prediction_array))
    cls = WASTE_CLASSES[idx] if idx < len(WASTE_CLASSES) else "Unknown"
    conf = float(prediction_array[idx])
    return {
        "predicted_class": cls,
        "confidence": conf,
        "confidence_percentage": round(conf * 100, 2),
        "all_probabilities": {
            WASTE_CLASSES[i] if i < len(WASTE_CLASSES) else f"Class_{i}": round(float(prediction_array[i]), 4)
            for i in range(len(prediction_array))
        },
    }

# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def wait_for_kafka(bootstrap: str, retries: int = 20, delay: float = 3.0):
    """Poll until Kafka/Redpanda is reachable (gives compose time to start)."""
    from kafka import KafkaAdminClient
    from kafka.errors import NoBrokersAvailable

    for attempt in range(1, retries + 1):
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=5000)
            admin.close()
            logger.info("Kafka reachable at %s", bootstrap)
            return
        except NoBrokersAvailable:
            logger.info("Waiting for Kafka... (%d/%d)", attempt, retries)
            time.sleep(delay)
    raise RuntimeError(f"Kafka not reachable after {retries} attempts")


def main():
    init_db()
    wait_for_kafka(KAFKA_BOOTSTRAP)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        group_id="waste-workers",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    logger.info("Worker listening on topic '%s' ...", TOPIC)

    for message in consumer:
        job = message.value
        request_id = job.get("request_id")
        filepath    = job.get("filepath")

        logger.info("Processing job: request_id=%s, file=%s", request_id, filepath)

        try:
            img = preprocess_image(filepath)
            raw = call_tf_serving(img)

            if raw is None:
                write_result(request_id, {"error": "TF Serving unavailable"})
                continue

            pred   = get_prediction_details(raw)
            recycle = RECYCLING_INSTRUCTIONS.get(
                pred["predicted_class"],
                {"disposal": "Check local guidelines.", "tips": "Sort by material.", "environmental_impact": "Variable"},
            )

            write_result(request_id, {"prediction": pred, "recycling_info": recycle})
            logger.info("Done: request_id=%s -> %s (%.1f%%)",
                        request_id, pred["predicted_class"], pred["confidence_percentage"])

        except Exception as exc:
            logger.error("Job failed: request_id=%s, error=%s", request_id, exc)
            write_result(request_id, {"error": str(exc)})


if __name__ == "__main__":
    main()
