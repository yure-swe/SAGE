# =============================================================================
# app.py — Main Flask Application (Updated with Dashboard Structure)
# =============================================================================
# Thesis: Ensemble Learning for Predicting Game Success (Pre-Launch Focus)
#
# Routes:
#   GET/POST /           → main dashboard (form + results)
#   GET      /model-info → model evaluation metrics
#   GET      /guide      → user manual/documentation
#   GET      /about      → methodology & thesis info
#   POST     /api/predict → JSON endpoint
#   GET      /health     → server health check
# =============================================================================

import logging
import os
from logging.handlers import RotatingFileHandler

from flask import Flask, render_template, request, jsonify
from predictor   import predict, ALL_FEATURES, CLASS_RANGES, N_CLASSES
from recommender import get_recommendations
from validation  import validate_form_data, FIELD_SPECS

app = Flask(__name__)

# ── Hardening: cap request size (prevents memory-exhaustion attacks) ─────────
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024   # 256 KB — way more than we need

# ── Logging: rotating file + stderr ──────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "sage.log"),
    maxBytes=1_000_000,
    backupCount=3,
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
app.logger.addHandler(_handler)
app.logger.setLevel(logging.INFO)

# ── Feature metadata for the HTML form ───────────────────────────────────────
# Groups features into logical sections for the UI
FORM_SECTIONS = [
    {
        "title": "💰 Pricing",
        "fields": [
            {"name": "price",        "label": "Price (USD)",         "type": "number", "default": 9.99,  "min": 0,   "max": 200,  "step": 0.01},
            {"name": "initialprice", "label": "Initial Price (USD)", "type": "number", "default": 9.99,  "min": 0,   "max": 200,  "step": 0.01},
            {"name": "is_free",      "label": "Free to Play?",       "type": "toggle", "default": 0},
        ]
    },
    {
        "title": "🗓️ Release",
        "fields": [
            {"name": "release_month", "label": "Release Month", "type": "select",
             "options": ["1=January","2=February","3=March","4=April","5=May","6=June",
                         "7=July","8=August","9=September","10=October","11=November","12=December"],
             "default": 10},
        ]
    },
    {
        "title": "🎮 Genre",
        "fields": [
            {"name": "Action",     "label": "Action",     "type": "toggle", "default": 0},
            {"name": "Adventure",  "label": "Adventure",  "type": "toggle", "default": 0},
            {"name": "RPG",        "label": "RPG",        "type": "toggle", "default": 0},
            {"name": "Strategy",   "label": "Strategy",   "type": "toggle", "default": 0},
            {"name": "Simulation", "label": "Simulation", "type": "toggle", "default": 0},
            {"name": "Indie",      "label": "Indie",      "type": "toggle", "default": 1},
            {"name": "Sports",     "label": "Sports",     "type": "toggle", "default": 0},
            {"name": "Racing",     "label": "Racing",     "type": "toggle", "default": 0},
        ]
    },
    {
        "title": "🖥️ Platform",
        "fields": [
            {"name": "platform_windows", "label": "Windows", "type": "toggle", "default": 1},
            {"name": "platform_mac",     "label": "Mac",     "type": "toggle", "default": 0},
            {"name": "platform_linux",   "label": "Linux",   "type": "toggle", "default": 0},
            {"name": "platform_count",   "label": "Total Platforms (auto)", "type": "number",
             "default": 1, "min": 1, "max": 3, "step": 1, "hidden": True},
        ]
    },
    {
        "title": "🌍 Languages",
        "fields": [
            {"name": "supported_languages_count",  "label": "Text Languages Supported",       "type": "number", "default": 1,  "min": 0, "max": 50, "step": 1},
            {"name": "full_audio_languages_count", "label": "Full Audio Languages Supported", "type": "number", "default": 0,  "min": 0, "max": 20, "step": 1},
        ]
    },
    {
        "title": "🏪 Store Page",
        "fields": [
            {"name": "screenshot_count",  "label": "Number of Screenshots",  "type": "number", "default": 5,   "min": 0, "max": 20, "step": 1},
            {"name": "has_trailer",       "label": "Has Trailer?",            "type": "toggle", "default": 0},
            {"name": "trailer_count",     "label": "Number of Trailers",      "type": "number", "default": 0,   "min": 0, "max": 10, "step": 1},
            {"name": "about_length",      "label": "Description Length (chars)", "type": "number", "default": 500, "min": 0, "max": 5000, "step": 10},
            {"name": "has_detailed_desc", "label": "Detailed Description (>500 chars)?", "type": "toggle", "default": 0},
            {"name": "has_website",       "label": "Has Official Website?",   "type": "toggle", "default": 0},
            {"name": "has_support_email", "label": "Has Support Email?",      "type": "toggle", "default": 0},
        ]
    },
    {
        "title": "🏢 Developer / Publisher",
        "fields": [
            {"name": "developer_count",  "label": "Number of Developers",  "type": "number", "default": 1, "min": 1, "max": 20, "step": 1},
            {"name": "publisher_count",  "label": "Number of Publishers",  "type": "number", "default": 0, "min": 0, "max": 10, "step": 1},
            {"name": "has_publisher",    "label": "Has Publisher?",         "type": "toggle", "default": 0},
            {"name": "is_solo_dev",      "label": "Solo Developer?",        "type": "toggle", "default": 1},
            {"name": "required_age",     "label": "Required Age (0=none)",  "type": "number", "default": 0, "min": 0, "max": 18, "step": 1},
            {"name": "is_mature_content","label": "Mature Content (18+)?",  "type": "toggle", "default": 0},
        ]
    },
    {
        "title": "🏆 Steam Features",
        "fields": [
            {"name": "has_achievements",       "label": "Steam Achievements?",    "type": "toggle", "default": 0},
            {"name": "achievement_count",      "label": "Number of Achievements", "type": "number", "default": 0, "min": 0, "max": 500, "step": 1},
            {"name": "has_trading_cards",      "label": "Steam Trading Cards?",   "type": "toggle", "default": 0},
            {"name": "has_workshop",           "label": "Steam Workshop?",        "type": "toggle", "default": 0},
            {"name": "has_cloud_save",         "label": "Steam Cloud Save?",      "type": "toggle", "default": 0},
            {"name": "has_controller_support", "label": "Controller Support?",    "type": "toggle", "default": 0},
            {"name": "has_vr_support",         "label": "VR Support?",            "type": "toggle", "default": 0},
            {"name": "has_in_app_purchases",   "label": "In-App Purchases?",      "type": "toggle", "default": 0},
            {"name": "has_family_sharing",     "label": "Family Sharing?",        "type": "toggle", "default": 0},
            {"name": "category_count",         "label": "Total Steam Categories", "type": "number", "default": 3, "min": 0, "max": 15, "step": 1},
        ]
    },
    {
        "title": "🏷️ Tags & Community",
        "fields": [
            {"name": "tag_count",             "label": "Number of Tags",          "type": "number", "default": 5,  "min": 0, "max": 20,    "step": 1},
            {"name": "has_multiplayer_tag",   "label": "Has Multiplayer Tag?",    "type": "toggle", "default": 0},
            {"name": "top_tag_votes_total",   "label": "Top Tag Votes Total",     "type": "number", "default": 500,"min": 0, "max": 5000,  "step": 10},
            {"name": "top_tag_votes_mean",    "label": "Top Tag Votes Mean",      "type": "number", "default": 100,"min": 0, "max": 1000,  "step": 10},
            {"name": "is_multiplayer",        "label": "Multiplayer Game?",       "type": "toggle", "default": 0},
        ]
    },
    {
        "title": "📦 Packaging & DLC",
        "fields": [
            {"name": "dlc_count",     "label": "DLC Count",       "type": "number", "default": 0, "min": 0, "max": 50, "step": 1},
            {"name": "package_count", "label": "Package Count",   "type": "number", "default": 1, "min": 1, "max": 10, "step": 1},
            {"name": "sku_count",     "label": "SKU Count",       "type": "number", "default": 1, "min": 1, "max": 20, "step": 1},
        ]
    },
]

# ── Model performance metrics (for /model-info page) ─────────────────────────
MODEL_METRICS = {
    "ensemble": {"weighted_f1": 0.6211, "macro_f1": 0.2688, "accuracy": 0.7085},
    "random_forest": {"weighted_f1": 0.6329, "macro_f1": 0.3019, "accuracy": 0.663},
    "gradient_boosting": {"weighted_f1": 0.6244, "macro_f1": 0.2748, "accuracy": 0.702},
    "xgboost": {"weighted_f1": 0.5869, "macro_f1": 0.2902, "accuracy": 0.552},
}

DATASET_INFO = {
    "total_games": 10000,
    "train_test_split": "80/20",
    "n_features": 48,
    "n_classes": 6,
    "class_distribution": {
        "Class 0 (≤10K)": 4500,
        "Class 1 (35K)": 2200,
        "Class 2 (75K)": 1500,
        "Class 3 (150K)": 1000,
        "Class 4 (350K)": 504,
        "Class 5 (≥750K)": 296,
    }
}


def compute_derived_features(form_data: dict) -> dict:
    """
    Compute composite/derived features from raw form inputs.
    Mirrors the feature engineering in enrich_prelaunch.py.
    """
    d = form_data

    def f(key, default=0.0):
        try:
            return float(d.get(key, default))
        except (ValueError, TypeError):
            return float(default)

    # Platform count (auto-compute from toggles)
    platform_count = f("platform_windows") + f("platform_mac") + f("platform_linux")
    d["platform_count"] = platform_count

    # Derived composite scores — same formulas as enrich_prelaunch.py
    d["store_page_score"] = (
        min(f("screenshot_count"), 10) / 10 * 0.30 +
        f("has_trailer")                          * 0.25 +
        f("has_detailed_desc")                    * 0.25 +
        f("has_website")                          * 0.10 +
        f("has_support_email")                    * 0.10
    )

    d["steam_features_score"] = (
        f("has_achievements")       * 0.25 +
        f("has_trading_cards")      * 0.15 +
        f("has_cloud_save")         * 0.15 +
        f("has_workshop")           * 0.20 +
        f("has_controller_support") * 0.15 +
        f("has_family_sharing")     * 0.10
    )

    return d


# =============================================================================
# ERROR HANDLERS  — return JSON for /api/* and friendly HTML elsewhere
# =============================================================================
def _wants_json() -> bool:
    return (
        request.path.startswith("/api/")
        or request.is_json
        or "application/json" in (request.headers.get("Accept") or "")
    )

@app.errorhandler(400)
def _bad_request(e):
    msg = getattr(e, "description", "Bad request")
    if _wants_json():
        return jsonify({"error": "bad_request", "message": msg}), 400
    return render_template("dashboard.html",
                           form_sections=FORM_SECTIONS, result=None,
                           recommendations=None, form_data={},
                           class_ranges=CLASS_RANGES, show_results=False,
                           validation_errors=[msg]), 400

@app.errorhandler(413)
def _payload_too_large(e):
    if _wants_json():
        return jsonify({"error": "payload_too_large",
                        "message": "Request body exceeds 256 KB."}), 413
    return ("Request payload too large.", 413)

@app.errorhandler(404)
def _not_found(e):
    if _wants_json():
        return jsonify({"error": "not_found"}), 404
    return ("Page not found.", 404)

@app.errorhandler(500)
def _server_error(e):
    app.logger.exception("Unhandled server error")
    if _wants_json():
        return jsonify({"error": "internal_error",
                        "message": "An unexpected error occurred."}), 500
    return ("Internal server error.", 500)


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/", methods=["GET", "POST"])
def dashboard():
    """Main dashboard: form on left/top, results on right/bottom."""

    form_data = {}
    show_results = False
    result = None
    recs = None
    validation_errors = []

    if request.method == "POST":
        # 1. Collect form fields
        raw_form = request.form.to_dict()

        # 2. Default unchecked toggles to 0 BEFORE validation
        all_toggle_fields = [
            "is_free", "platform_windows", "platform_mac", "platform_linux",
            "has_trailer", "has_detailed_desc", "has_website", "has_support_email",
            "has_publisher", "is_solo_dev", "is_mature_content",
            "has_achievements", "has_trading_cards", "has_workshop", "has_cloud_save",
            "has_controller_support", "has_vr_support", "has_in_app_purchases",
            "has_family_sharing", "has_multiplayer_tag", "is_multiplayer",
            "Action", "Adventure", "RPG", "Strategy",
            "Simulation", "Indie", "Sports", "Racing",
        ]
        for field in all_toggle_fields:
            if field not in raw_form:
                raw_form[field] = 0

        # 2b. Enforce coupling: has_detailed_desc is derived from about_length (>500)
        try:
            _about_len = int(float(raw_form.get("about_length", 0) or 0))
        except (TypeError, ValueError):
            _about_len = 0
        raw_form["has_detailed_desc"] = 1 if _about_len > 500 else 0

        # 3. Validate inputs server-side
        cleaned, validation_errors = validate_form_data(raw_form, strict=False)

        if validation_errors:
            app.logger.info("Validation failed: %s", validation_errors)
            # Re-render the form with errors; keep user's submitted values
            return render_template(
                "dashboard.html",
                form_sections=FORM_SECTIONS,
                result=None,
                recommendations=None,
                form_data=raw_form,
                class_ranges=CLASS_RANGES,
                show_results=False,
                validation_errors=validation_errors,
            ), 400

        form_data = cleaned
        show_results = True

        # 4. Compute derived/composite features
        form_data = compute_derived_features(form_data)

        # 5. Run prediction (guarded)
        try:
            result = predict(form_data)
            recs   = get_recommendations(form_data, result["predicted_class"])
        except Exception:
            app.logger.exception("Prediction pipeline failed")
            return render_template(
                "dashboard.html",
                form_sections=FORM_SECTIONS,
                result=None,
                recommendations=None,
                form_data=raw_form,
                class_ranges=CLASS_RANGES,
                show_results=False,
                validation_errors=["Prediction failed. Please try again."],
            ), 500

    return render_template(
        "dashboard.html",
        form_sections=FORM_SECTIONS,
        result=result,
        recommendations=recs,
        form_data=form_data,
        class_ranges=CLASS_RANGES,
        show_results=show_results,
        validation_errors=validation_errors,
    )


@app.route("/model-info", methods=["GET"])
def model_info():
    """Model evaluation metrics and performance visualization."""
    return render_template(
        "model_info.html",
        metrics=MODEL_METRICS,
        dataset_info=DATASET_INFO,
        class_ranges=CLASS_RANGES,
    )


@app.route("/guide", methods=["GET"])
def guide():
    """User manual and documentation."""
    return render_template(
        "guide.html",
        form_sections=FORM_SECTIONS,
    )


@app.route("/about", methods=["GET"])
def about():
    """Thesis methodology and about page."""
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for prediction.
    Accepts JSON body with feature values.
    Returns prediction + recommendations as JSON.
    """
    if not request.is_json:
        return jsonify({
            "error": "invalid_content_type",
            "message": "Content-Type must be application/json",
        }), 400

    try:
        payload = request.get_json(silent=True)
    except Exception:
        payload = None

    if payload is None:
        return jsonify({
            "error": "invalid_json",
            "message": "Request body is not valid JSON.",
        }), 400

    if not isinstance(payload, dict):
        return jsonify({
            "error": "invalid_payload",
            "message": "JSON body must be an object of feature values.",
        }), 400

    # Validate
    cleaned, errors = validate_form_data(payload, strict=False)
    if errors:
        return jsonify({
            "error": "validation_failed",
            "messages": errors,
        }), 422

    cleaned = compute_derived_features(cleaned)

    try:
        result = predict(cleaned)
        recs   = get_recommendations(cleaned, result["predicted_class"])
    except Exception:
        app.logger.exception("API prediction failed")
        return jsonify({
            "error": "prediction_failed",
            "message": "Model pipeline error.",
        }), 500

    return jsonify({
        "prediction":      result,
        "recommendations": recs,
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status":   "ok",
        "models":   "loaded",
        "features": len(ALL_FEATURES),
        "classes":  N_CLASSES,
        "validated_fields": len(FIELD_SPECS),
    }), 200


# =============================================================================
# ENTRY POINT (local dev only — Gunicorn uses wsgi.py on Hostinger)
# =============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
