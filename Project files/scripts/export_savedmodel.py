"""
export_savedmodel.py
--------------------
One-time script: converts vgg16.h5 (Keras HDF5 format) to TensorFlow
SavedModel format with a proper 'serving_default' signature required by
TensorFlow Serving's REST API.

Usage (from Project files/ directory):
    python scripts/export_savedmodel.py

Output:
    models/waste_classifier/1/   <- TF Serving expects this version structure

TF Serving REST format (instances → predictions):
    POST /v1/models/waste_classifier:predict
    { "instances": [[[...224x224x3 float array...]]] }
    → { "predictions": [[p_biodeg, p_recyclable, p_trash]] }
"""

import os
import sys
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Allow running from any directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_H5       = os.path.join(PROJECT_DIR, "vgg16.h5")
SAVED_MODEL_DIR = os.path.join(PROJECT_DIR, "models", "waste_classifier", "1")


def main():
    if not os.path.exists(MODEL_H5):
        logger.error("Model file not found: %s", MODEL_H5)
        logger.error("Place vgg16.h5 in the 'Project files/' directory first.")
        sys.exit(1)

    import tensorflow as tf  # noqa: F401 — kept for version logging
    from tensorflow.keras.models import load_model

    logger.info("Loading Keras model from: %s", MODEL_H5)
    model = load_model(MODEL_H5)
    logger.info("Model loaded. Input shape: %s  Output shape: %s",
                model.input_shape, model.output_shape)

    # Clean up any previous export so TF Serving picks up the new one cleanly
    if os.path.exists(SAVED_MODEL_DIR):
        logger.info("Removing previous export at %s", SAVED_MODEL_DIR)
        shutil.rmtree(SAVED_MODEL_DIR)
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Create an explicit 'serving_default' signature.
    # Keras 3's model.export() uses the name 'serve', but TF Serving's REST
    # API looks for 'serving_default' by default.  Wrapping in a @tf.function
    # with input_signature gives us the correct signature name AND locks in
    # the concrete input shape so TF Serving doesn't need to trace at runtime.
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Export using Keras 3's model.export() — the only method that works
    # reliably with Keras 3 + Python 3.13 (tf.saved_model.save() triggers a
    # _DictWrapper bug in TF 2.21 on Python 3.13).
    #
    # model.export() creates a SavedModel with a 'serve' endpoint (not the
    # standard 'serving_default' name).  We handle this in app_tfserving.py
    # by including "signature_name": "serve" in every REST request body —
    # TF Serving's REST API supports this field in both column and row format.
    # -----------------------------------------------------------------------
    logger.info("Exporting SavedModel (Keras export) to: %s", SAVED_MODEL_DIR)
    model.export(SAVED_MODEL_DIR)

    logger.info("Export complete!")
    logger.info("Signature name used by TF Serving: 'serve'")
    logger.info("TF Serving REST endpoint: http://localhost:8501/v1/models/waste_classifier:predict")
    logger.info("Payload: { \"signature_name\": \"serve\", \"instances\": [<224x224x3 float list>] }")
    logger.info("Response: { \"predictions\": [[p0, p1, p2]] }")


if __name__ == "__main__":
    main()
