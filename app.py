"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Backend: app.py (Flask Web Application)
Description: Production Flask server exposing REST API endpoints for
             model inference, health check, and rendering the web dashboard.
=============================================================================
"""

import os
import json
import logging
import traceback
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application Initialization & Constants
# ---------------------------------------------------------------------------
app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "coating_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Mandatory feature order matching training schema exactly
INPUT_FEATURES = [
    "Resin_Content",
    "Pigment_Content",
    "Solvent_Content",
    "Additive_Content",
    "Curing_Agent_Content",
    "Curing_Temperature",
    "Curing_Time"
]

TARGET_PROPERTIES = [
    "Hardness",
    "Adhesion",
    "Flexibility",
    "Gloss",
    "Drying_Time"
]

PROPERTY_UNITS = {
    "Hardness": "Index (0-100)",
    "Adhesion": "Index (0-100)",
    "Flexibility": "Index (0-100)",
    "Gloss": "GU @ 60°",
    "Drying_Time": "minutes"
}

# Physical bounds for sanity checks
FEATURE_BOUNDS = {
    "Resin_Content": (0.0, 100.0),
    "Pigment_Content": (0.0, 100.0),
    "Solvent_Content": (0.0, 100.0),
    "Additive_Content": (0.0, 50.0),
    "Curing_Agent_Content": (0.0, 50.0),
    "Curing_Temperature": (0.0, 300.0),
    "Curing_Time": (0.0, 600.0)
}

# ---------------------------------------------------------------------------
# Model & Pipeline Loading (Loaded once on startup)
# ---------------------------------------------------------------------------
ml_model = None
preprocessor = None
model_metadata = {}
is_demo_mode = True

def load_ml_assets():
    """Load machine learning model, preprocessor, and metadata at startup."""
    global ml_model, preprocessor, model_metadata, is_demo_mode

    # 1. Load Metadata if available
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
            is_demo_mode = model_metadata.get("is_demo", False)
            logger.info("Loaded model metadata successfully.")
        except Exception as e:
            logger.warning(f"Failed to parse metadata file: {e}")
            model_metadata = {}

    # 2. Load Preprocessor if available separately
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Loaded preprocessing pipeline from: {PREPROCESSOR_PATH}")
        except Exception as e:
            logger.warning(f"Could not load preprocessor: {e}")
            preprocessor = None

    # 3. Load Main Model
    if os.path.exists(MODEL_PATH):
        try:
            loaded_obj = joblib.load(MODEL_PATH)
            # Handle if model is wrapped in a dict bundle (e.g. from Stage 3)
            if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                ml_model = loaded_obj["model"]
                if "metrics" in loaded_obj and "overall_metrics" not in model_metadata:
                    model_metadata["overall_metrics"] = loaded_obj["metrics"]
                if "model_name" in loaded_obj:
                    model_metadata["model_name"] = loaded_obj["model_name"]
            else:
                ml_model = loaded_obj

            logger.info(f"Loaded trained ML model successfully from: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load ML model from {MODEL_PATH}: {e}")
            ml_model = None
    else:
        logger.warning(f"Model file not found at: {MODEL_PATH}. Application starting in fallback mode.")
        ml_model = None

# Execute asset loading at server boot
load_ml_assets()


# ---------------------------------------------------------------------------
# Routes & Endpoints
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the homepage dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming API status and model readiness."""
    return jsonify({
        "status": "running",
        "model_loaded": ml_model is not None,
        "is_demo": is_demo_mode,
        "model_name": model_metadata.get("model_name", "Multi-Output Regressor"),
        "features": INPUT_FEATURES,
        "targets": TARGET_PROPERTIES
    }), 200


@app.route("/api/info", methods=["GET"])
def info():
    """Return model architecture, parameters, and metadata."""
    return jsonify({
        "model_loaded": ml_model is not None,
        "is_demo": is_demo_mode,
        "model_name": model_metadata.get("model_name", "Multi-Output Machine Learning Model"),
        "task": model_metadata.get("task", "Multi-output Regression"),
        "features": INPUT_FEATURES,
        "targets": TARGET_PROPERTIES,
        "units": PROPERTY_UNITS,
        "metrics": model_metadata.get("overall_metrics", None),
        "per_target_metrics": model_metadata.get("per_target_metrics", None)
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint: takes formulation & baking parameters and outputs
    predicted coating properties via the actual trained ML model.
    """
    global ml_model, preprocessor

    # Check model readiness
    if ml_model is None:
        logger.error("Prediction requested but ML model is not loaded.")
        return jsonify({
            "success": False,
            "error": "ML model is not loaded on the server. Please ensure 'models/coating_model.joblib' exists."
        }), 503

    # Check JSON body
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request must be a valid JSON payload with Content-Type: application/json."
        }), 400

    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "error": "Empty request body received."
        }), 400

    # 1. Input Validation
    missing_fields = [f for f in INPUT_FEATURES if f not in data]
    if missing_fields:
        return jsonify({
            "success": False,
            "error": f"Missing required parameters: {', '.join(missing_fields)}"
        }), 400

    # Convert and validate numeric bounds
    parsed_inputs = {}
    for feat in INPUT_FEATURES:
        val = data.get(feat)
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": f"Invalid numerical value for '{feat.replace('_', ' ')}': {val}"
            }), 400

        # Boundary checks
        b_min, b_max = FEATURE_BOUNDS.get(feat, (0.0, 1000.0))
        if val_float < b_min:
            return jsonify({
                "success": False,
                "error": f"'{feat.replace('_', ' ')}' cannot be less than {b_min}."
            }), 400
        if val_float > b_max:
            return jsonify({
                "success": False,
                "error": f"'{feat.replace('_', ' ')}' exceeds physical maximum of {b_max}."
            }), 400

        parsed_inputs[feat] = val_float

    # Formulation composition sum check
    formulation_components = [
        "Resin_Content", "Pigment_Content", "Solvent_Content",
        "Additive_Content", "Curing_Agent_Content"
    ]
    comp_sum = sum(parsed_inputs[c] for c in formulation_components)

    try:
        # 2. Strict Feature Ordering into pandas DataFrame
        input_df = pd.DataFrame([parsed_inputs], columns=INPUT_FEATURES)

        # 3. Model Inference (handling pipeline or separate preprocessor)
        if preprocessor is not None and not hasattr(ml_model, "steps"):
            # Separate preprocessor applied only if ml_model is not already a Pipeline
            scaled_features = preprocessor.transform(input_df)
            raw_preds = ml_model.predict(scaled_features)
        else:
            raw_preds = ml_model.predict(input_df)

        # Handle predictions shape (single sample)
        if isinstance(raw_preds, pd.DataFrame):
            pred_vector = raw_preds.values[0]
        elif isinstance(raw_preds, np.ndarray):
            pred_vector = raw_preds[0] if raw_preds.ndim > 1 else raw_preds
        else:
            pred_vector = list(raw_preds)

        # 4. Map outputs to target property names & enforce physical limits
        predictions = {}
        for i, tgt in enumerate(TARGET_PROPERTIES):
            val = float(pred_vector[i])
            # Enforce physical bounds (e.g. non-negative indices and time)
            if tgt in ["Hardness", "Adhesion", "Flexibility", "Gloss"]:
                val = float(np.clip(val, 0.0, 100.0))
            elif tgt == "Drying_Time":
                val = float(max(0.0, val))

            predictions[tgt] = round(val, 2)

        # 5. Return JSON Response
        return jsonify({
            "success": True,
            "predictions": predictions,
            "units": PROPERTY_UNITS,
            "input_summary": parsed_inputs,
            "formulation_sum": round(comp_sum, 2),
            "is_demo": is_demo_mode
        }), 200

    except Exception as e:
        logger.error(f"Inference Exception: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "Internal prediction error occurred while processing the model inference."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Coating Prediction Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
