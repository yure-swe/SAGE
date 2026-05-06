# =============================================================================
# app.py — Main Flask Application  (v2)
# =============================================================================
# Thesis: Ensemble Learning for Predicting Game Success (Pre-Launch Focus)
#
# Changes from v1:
#   FORM_SECTIONS:
#     - Removed: has_trailer, trailer_count (all-zero data, removed from model)
#     - Removed: has_trading_cards, has_workshop (post-success indicators)
#     - Removed: dlc_count (post-success indicator)
#     - Removed: is_solo_dev, has_publisher, publisher_count, developer_count
#               (budget proxies, removed from model)
#     - Removed: has_multiplayer_tag (redundant with tag binary features)
#     - Removed: Indie genre flag (not a genre; captured by tag encoding)
#     - Added:   release_date date-picker → computes game_age_days
#     - Added:   language checklist → computes weighted_language_score
#     - Added:   top-tag checkboxes (read dynamically from feature_dict)
#     - "🏢 Developer / Publisher" section removed entirely
#
#   compute_derived_features:
#     - Delegates to predictor.compute_derived_features() — single source of truth
#     - Also handles game_age_days computation from release_date field
#     - Also handles weighted_language_score from selected_languages checklist
#
#   Routes:
#     - Result template now receives game_age_context and output_label from predict()
#
# Routes:
#   GET/POST /           → main dashboard (form + results)
#   GET      /model-info → model evaluation metrics
#   GET      /guide      → user manual/documentation
#   GET      /about      → methodology & thesis info
#   POST     /api/predict→ JSON endpoint
#   GET      /health     → server health check
# =============================================================================

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from flask import Flask, render_template, request, jsonify

from predictor import (
    predict,
    compute_derived_features,
    compute_game_age_days,
    ALL_FEATURES, TAG_FEATURES, CLASS_RANGES, N_CLASSES,
    LANGUAGE_WEIGHTS,
)
from recommender import get_recommendations
from validation import validate_form_data, FIELD_SPECS

app = Flask(__name__)

# ── Hardening: cap request size ───────────────────────────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024  # 256 KB

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "sage.log"), maxBytes=1_000_000, backupCount=3
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
app.logger.addHandler(_handler)
app.logger.setLevel(logging.INFO)

# =============================================================================
# LANGUAGE LIST — ordered by weight (most impactful first) for the checklist
# =============================================================================
LANGUAGE_LIST = sorted(LANGUAGE_WEIGHTS.items(), key=lambda x: x[1], reverse=True)

# =============================================================================
# TAG FEATURES LIST — from feature_dict, used to build tag checkbox section
# =============================================================================
# Convert tag column names back to display names for the UI.
# e.g. "tag_action_roguelike" → "Action Roguelike"
def col_to_tag_display(col_name: str) -> str:
    name = col_name.removeprefix("tag_").replace("_", " ").title()
    return name

TAG_DISPLAY = [(col, col_to_tag_display(col)) for col in TAG_FEATURES]

# =============================================================================
# FORM_SECTIONS (v2)
# =============================================================================
FORM_SECTIONS = [
    {
        "title": "💰 Pricing",
        "fields": [
            {"name": "price",         "label": "Price (USD)",         "type": "number",
             "default": 9.99, "min": 0, "max": 200, "step": 0.01},
            {"name": "initialprice",  "label": "Initial Price (USD)", "type": "number",
             "default": 9.99, "min": 0, "max": 200, "step": 0.01},
            {"name": "is_free",       "label": "Free to Play?",       "type": "toggle", "default": 0},
        ]
    },
    {
        # Release date as a date-picker; game_age_days is computed server-side.
        # Users planning a future/past launch can set this accordingly.
        "title": "🗓️ Release",
        "fields": [
            {"name": "release_date",  "label": "Release Date (or planned launch date)",
             "type": "date", "default": datetime.today().strftime("%Y-%m-%d"),
             "help": ("Used to compute how long the game has been (or will be) on Steam. "
                      "Set to your planned launch date for pre-launch predictions.")},
            {"name": "release_month", "label": "Release Month (auto)", "type": "select",
             "options": ["1=January","2=February","3=March","4=April","5=May","6=June",
                         "7=July","8=August","9=September","10=October","11=November","12=December"],
             "default": datetime.today().month, "hidden": True},
        ]
    },
    {
        "title": "🎮 Genre",
        "fields": [
            {"name": "Action",      "label": "Action",      "type": "toggle", "default": 0},
            {"name": "Adventure",   "label": "Adventure",   "type": "toggle", "default": 0},
            {"name": "RPG",         "label": "RPG",         "type": "toggle", "default": 0},
            {"name": "Strategy",    "label": "Strategy",    "type": "toggle", "default": 0},
            {"name": "Simulation",  "label": "Simulation",  "type": "toggle", "default": 0},
            {"name": "Sports",      "label": "Sports",      "type": "toggle", "default": 0},
            {"name": "Racing",      "label": "Racing",      "type": "toggle", "default": 0},
        ]
    },
    {
        "title": "🖥️ Platform",
        "fields": [
            {"name": "platform_windows", "label": "Windows", "type": "toggle", "default": 1},
            {"name": "platform_mac",     "label": "Mac",     "type": "toggle", "default": 0},
            {"name": "platform_linux",   "label": "Linux",   "type": "toggle", "default": 0},
            # platform_count is auto-computed in compute_derived_features
            {"name": "platform_count", "label": "Total Platforms (auto)", "type": "number",
             "default": 1, "min": 1, "max": 3, "step": 1, "hidden": True},
        ]
    },
    {
        # Language checklist — computes weighted_language_score server-side.
        # Replaces the old raw "supported_languages_count" number input.
        "title": "🌍 Languages",
        "fields": [
            {
                "name": "selected_languages",
                "label": "Supported Languages",
                "type": "language_checklist",
                "help": ("Check each language your game will support as text. "
                         "Higher-weight languages (English, Simplified Chinese, Russian) "
                         "reflect Steam's largest market segments."),
                "options": LANGUAGE_LIST,   # list of (lang_name, weight) tuples
            },
            {"name": "supported_languages_count",  "label": "Text Languages (auto-count)",
             "type": "number", "default": 1, "min": 0, "max": 50, "step": 1, "hidden": True},
            {"name": "full_audio_languages_count", "label": "Full Audio Languages",
             "type": "number", "default": 0, "min": 0, "max": 20, "step": 1},
        ]
    },
    {
        "title": "🏪 Store Page",
        "fields": [
            {"name": "screenshot_count",  "label": "Number of Screenshots",
             "type": "number", "default": 5,   "min": 0, "max": 20, "step": 1},
            {"name": "about_length",      "label": "Description Length (chars)",
             "type": "number", "default": 500, "min": 0, "max": 5000, "step": 10,
             "hint": ">500 chars automatically sets the Detailed Description flag."},
            {"name": "has_detailed_desc", "label": "Detailed Description (auto)",
             "type": "toggle", "default": 0, "hidden": True},
            {"name": "has_website",       "label": "Has Official Website?",
             "type": "toggle", "default": 0},
            {"name": "has_support_email", "label": "Has Support Email?",
             "type": "toggle", "default": 0},
        ]
    },
    {
        "title": "🏆 Steam Features",
        "fields": [
            {"name": "has_achievements",      "label": "Steam Achievements?",   "type": "toggle", "default": 0},
            {"name": "achievement_count",     "label": "Number of Achievements",
             "type": "number", "default": 0, "min": 0, "max": 500, "step": 1},
            {"name": "has_cloud_save",        "label": "Steam Cloud Save?",     "type": "toggle", "default": 0},
            {"name": "has_controller_support","label": "Controller Support?",   "type": "toggle", "default": 0},
            {"name": "has_vr_support",        "label": "VR Support?",           "type": "toggle", "default": 0},
            {"name": "has_in_app_purchases",  "label": "In-App Purchases?",     "type": "toggle", "default": 0},
            {"name": "has_family_sharing",    "label": "Family Sharing?",       "type": "toggle", "default": 0},
            # category_count is auto-derived from the toggles above + is_multiplayer
            {"name": "category_count",        "label": "Total Steam Categories (auto)",
             "type": "number", "default": 0, "min": 0, "max": 15, "step": 1, "hidden": True},
        ]
    },
    {
        "title": "🎯 Audience",
        "fields": [
            {"name": "required_age",    "label": "Required Age (0 = none)",
             "type": "number", "default": 0, "min": 0, "max": 18, "step": 1,
             "hint": "≥ 17 automatically sets the mature content flag."},
            {"name": "is_mature_content","label": "Mature Content (auto)",
             "type": "toggle", "default": 0, "hidden": True},
            {"name": "is_multiplayer",  "label": "Multiplayer Game?", "type": "toggle", "default": 0,
             "help": "Check if your game includes any online/local co-op, PvP, or MMO. "
                     "Also select the relevant multiplayer tags in the Tags section below."},
        ]
    },
    {
        # Tags section: top-N binary checkboxes + volume stats.
        # TAG_DISPLAY is built from feature_dict at startup — always in sync with models.
        "title": "🏷️ Tags",
        "fields": [
            {
                "name": "selected_tags",
                "label": "Steam Tags (select all that apply)",
                "type": "tag_checklist",
                "options": TAG_DISPLAY,  # list of (col_name, display_name) tuples
                "help": ("Select the tags that best describe your game's genre, "
                         "mechanics, and feel. These map directly to Steam's "
                         "community-driven tagging system."),
            },
            {"name": "tag_count",           "label": "Tag Count (auto)",
             "type": "number", "default": 5,   "min": 0, "max": 20, "step": 1, "hidden": True},

        ]
    },
    {
        "title": "📦 Packaging",
        "fields": [
            {"name": "package_count", "label": "Package Count",
             "type": "number", "default": 1, "min": 1, "max": 10, "step": 1},
            {"name": "sku_count",     "label": "SKU Count",
             "type": "number", "default": 1, "min": 1, "max": 20, "step": 1},
        ]
    },
]

# ── All toggle fields for the POST handler (unchecked → 0 default) ────────────
ALL_TOGGLE_FIELDS = [
    "is_free",
    "platform_windows", "platform_mac", "platform_linux",
    "has_website", "has_support_email",
    "has_achievements", "has_cloud_save", "has_controller_support",
    "has_vr_support", "has_in_app_purchases", "has_family_sharing",
    "is_multiplayer",
    "Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports", "Racing",
]

# ── Model performance metrics (for /model-info page) ─────────────────────────
# NOTE: Update these after retraining with v2 feature set.
MODEL_METRICS = {
    "ensemble":          {"weighted_f1": 0.6292, "macro_f1": 0.2954, "accuracy": 0.712},
    "random_forest":     {"weighted_f1": 0.6239, "macro_f1": 0.2782, "accuracy": 0.6925},
    "gradient_boosting": {"weighted_f1": 0.6327, "macro_f1": 0.2997, "accuracy": 0.7075},
    "xgboost":           {"weighted_f1": 0.6093, "macro_f1": 0.2954, "accuracy": 0.5695},
}
DATASET_INFO = {
    "total_games":       10000,
    "train_test_split":  "80/20",
    "n_features":        len(ALL_FEATURES),
    "n_classes":         6,
    "class_distribution": {
        0: 4500,
        1: 2200,
        2: 1500,
        3: 1000,
        4:  504,
        5:  296,
    }
}


# =============================================================================
# FORM PREPROCESSING
# =============================================================================
def preprocess_form(raw_form: dict) -> dict:
    """
    1. Default unchecked toggles to 0.
    2. Compute game_age_days from release_date.
    3. Collect selected_languages list from multi-select.
    4. Set supported_languages_count from selected_languages.
    5. Collect selected_tags and populate tag binary columns.
    6. Set tag_count from selected_tags.
    """
    d = dict(raw_form)

    # 1. Default toggles
    for field in ALL_TOGGLE_FIELDS:
        if field not in d:
            d[field] = 0

    # 2. game_age_days from release_date
    release_date_str = d.get("release_date", "")
    if release_date_str:
        d["game_age_days"] = compute_game_age_days(release_date_str)
        # Also set release_month if not already provided
        if "release_month" not in d or not str(d.get("release_month", "")).strip():
            try:
                dt = datetime.strptime(str(release_date_str).strip(), "%Y-%m-%d")
                d["release_month"] = dt.month
            except ValueError:
                d.setdefault("release_month", datetime.today().month)
    else:
        d.setdefault("game_age_days", 0)

    # 3. selected_languages: getlist handles multi-select in Flask
    # (When called from /api/predict with JSON, selected_languages arrives as a list.)
    langs = d.get("selected_languages", [])
    if isinstance(langs, str):
        langs = [langs] if langs else []
    d["selected_languages"]       = langs
    d["supported_languages_count"] = len(langs)

    # 4. selected_tags: populate individual tag binary columns
    tags_checked = d.get("selected_tags", [])
    if isinstance(tags_checked, str):
        tags_checked = [tags_checked] if tags_checked else []

    for col, _ in TAG_DISPLAY:
        d[col] = 1 if col in tags_checked else 0

    d["tag_count"] = len(tags_checked)

    return d


# =============================================================================
# ERROR HANDLERS
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
    form_data         = {}
    show_results      = False
    result            = None
    recs              = None
    validation_errors = []

    if request.method == "POST":
        # 1. Collect form fields
        raw_form = request.form.to_dict()

        # Multi-select fields need getlist, not to_dict()
        raw_form["selected_languages"] = request.form.getlist("selected_languages")
        raw_form["selected_tags"]      = request.form.getlist("selected_tags")

        # 2. Preprocess: compute game_age_days, tag columns, language score, etc.
        preprocessed = preprocess_form(raw_form)

        # 3. Validate
        cleaned, validation_errors = validate_form_data(preprocessed, strict=False)
        if validation_errors:
            app.logger.info("Validation failed: %s", validation_errors)
            return render_template(
                "dashboard.html",
                form_sections=FORM_SECTIONS,
                result=None, recommendations=None,
                form_data=raw_form, class_ranges=CLASS_RANGES,
                show_results=False, validation_errors=validation_errors,
            ), 400

        form_data    = cleaned
        show_results = True

        # Remember user-facing USD prices BEFORE compute_derived_features
        # converts them to cents (model space). Otherwise the cents value
        # would be rendered back into the form inputs (e.g. 9.99 → 999).
        display_price        = form_data.get("price")
        display_initialprice = form_data.get("initialprice")

        # 4. Compute derived/composite features (v2 formulas)
        form_data = compute_derived_features(form_data)

        # 5. Run prediction (predictor expects price in cents — already converted)
        try:
            result = predict(form_data)
            recs   = get_recommendations(form_data, result["predicted_class"])
        except Exception:
            app.logger.exception("Prediction pipeline failed")
            return render_template(
                "dashboard.html",
                form_sections=FORM_SECTIONS,
                result=None, recommendations=None,
                form_data=raw_form, class_ranges=CLASS_RANGES,
                show_results=False,
                validation_errors=["Prediction failed. Please try again."],
            ), 500

        # Restore USD values so the form re-renders with what the user typed
        # (compute_derived_features mutated these to cents for the model).
        form_data["price"]        = display_price
        form_data["initialprice"] = display_initialprice

    return render_template(
        "dashboard.html",
        form_sections=FORM_SECTIONS,
        result=result,
        recommendations=recs,
        form_data=form_data,
        class_ranges=CLASS_RANGES,
        show_results=show_results,
        validation_errors=validation_errors,
        tag_display=TAG_DISPLAY,
        language_list=LANGUAGE_LIST,
    )


@app.route("/model-info", methods=["GET"])
def model_info():
    return render_template(
        "model_info.html",
        metrics=MODEL_METRICS,
        dataset_info=DATASET_INFO,
        class_ranges=CLASS_RANGES,
    )


@app.route("/guide", methods=["GET"])
def guide():
    return render_template("guide.html", form_sections=FORM_SECTIONS)


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for prediction.

    Accepts JSON body with feature values.
    For languages: pass "selected_languages" as a list of language name strings.
    For tags:      pass "selected_tags" as a list of tag column name strings
                   (e.g. ["tag_action_roguelike", "tag_shooter"]).
    For game age:  pass "release_date" as "YYYY-MM-DD" string,
                   OR pass "game_age_days" directly as an integer.

    Returns prediction + recommendations as JSON.
    """
    if not request.is_json:
        return jsonify({"error": "invalid_content_type",
                        "message": "Content-Type must be application/json"}), 400

    try:
        payload = request.get_json(silent=True)
    except Exception:
        payload = None

    if payload is None:
        return jsonify({"error": "invalid_json",
                        "message": "Request body is not valid JSON."}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "invalid_payload",
                        "message": "JSON body must be an object of feature values."}), 400

    # Preprocess (handles release_date → game_age_days, tags, languages)
    preprocessed = preprocess_form(payload)

    # Validate
    cleaned, errors = validate_form_data(preprocessed, strict=False)
    if errors:
        return jsonify({"error": "validation_failed", "messages": errors}), 422

    cleaned = compute_derived_features(cleaned)

    try:
        result = predict(cleaned)
        recs   = get_recommendations(cleaned, result["predicted_class"])
    except Exception:
        app.logger.exception("API prediction failed")
        return jsonify({"error": "prediction_failed",
                        "message": "Model pipeline error."}), 500

    return jsonify({"prediction": result, "recommendations": recs})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "models":           "loaded",
        "features":         len(ALL_FEATURES),
        "tag_features":     len(TAG_FEATURES),
        "classes":          N_CLASSES,
        "validated_fields": len(FIELD_SPECS),
    }), 200


# =============================================================================
# ENTRY POINT (local dev only — Gunicorn uses wsgi.py on Hostinger)
# =============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
