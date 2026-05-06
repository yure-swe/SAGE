# =============================================================================
# predictor.py — Model Loading & Prediction Logic  (v2)
# =============================================================================
# Loads all .pkl files ONCE at startup (not per request).
# Exposes a single predict() function used by app.py.
#
# Changes from v1:
#   - Removed deprecated features from build_feature_vector:
#       has_trailer, trailer_count, has_trading_cards, has_workshop,
#       dlc_count, is_solo_dev, has_publisher, publisher_count,
#       developer_count, publisher_backing, has_multiplayer_tag, Indie
#   - Added game_age_days to CLASS_RANGES display context
#   - Tag binary features handled automatically via feature_dict["tag_features"]
#   - compute_derived_features mirrors enrich_prelaunch.py v2 formulas exactly
#   - Output dict now includes game_age_context string for UI display
# =============================================================================

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# ── Paths — relative to app/ directory ───────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# ── Load all artifacts once at module import ──────────────────────────────────
print("[predictor] Loading models...")

rf_model   = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
gb_model   = joblib.load(os.path.join(MODELS_DIR, "gb_classifier.pkl"))
xgb_model  = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.pkl"))
meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_classifier.pkl"))
scaler     = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feat_dict  = joblib.load(os.path.join(MODELS_DIR, "feature_dict.pkl"))

ALL_FEATURES     = feat_dict["all_features"]
NUMERIC_FEATURES = feat_dict["numeric_features"]
TAG_FEATURES     = feat_dict.get("tag_features", [])
CLASS_LABELS     = feat_dict["class_labels"]
N_CLASSES        = feat_dict["n_classes"]

print(f"[predictor] Models loaded. Features: {len(ALL_FEATURES)}, Classes: {N_CLASSES}")
print(f"[predictor] Tag binary features: {len(TAG_FEATURES)}")

# ── Steam language weights — must mirror enrich_prelaunch.py exactly ─────────
LANGUAGE_WEIGHTS = {
    "English":                 1.00,
    "Simplified Chinese":      0.85,
    "Russian":                 0.55,
    "German":                  0.45,
    "Spanish - Spain":         0.40,
    "French":                  0.40,
    "Portuguese - Brazil":     0.35,
    "Japanese":                0.30,
    "Korean":                  0.25,
    "Traditional Chinese":     0.20,
    "Polish":                  0.15,
    "Turkish":                 0.15,
    "Italian":                 0.12,
    "Dutch":                   0.10,
    "Czech":                   0.08,
    "Hungarian":               0.07,
    "Romanian":                0.07,
    "Spanish - Latin America": 0.12,
    "Portuguese - Portugal":   0.08,
    "Ukrainian":               0.08,
}
MAX_LANGUAGE_SCORE = sum(LANGUAGE_WEIGHTS.values())

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_RANGES = {
    0: {"label": "≤10K",   "range": "Up to 10,000 owners",         "tier": "Common Indie"},
    1: {"label": "≤35K",   "range": "10,000 – 35,000 owners",      "tier": "Niche"},
    2: {"label": "≤75K",   "range": "35,000 – 75,000 owners",      "tier": "Growing"},
    3: {"label": "≤150K",  "range": "75,000 – 150,000 owners",     "tier": "Established"},
    4: {"label": "≤350K",  "range": "150,000 – 350,000 owners",    "tier": "Popular"},
    5: {"label": "≥750K",  "range": "750,000+ owners",             "tier": "Breakout Hit"},
}
_UNKNOWN_CLASS = {"label": "Unknown", "range": "Unknown", "tier": "Unknown"}


# =============================================================================
# HELPER: tag name → column name
# Must mirror enrich_prelaunch.py's tag_to_col() exactly.
# =============================================================================
def tag_to_col(tag_name: str) -> str:
    col = tag_name.lower().strip()
    col = re.sub(r"[^a-z0-9\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    return f"tag_{col}"


# =============================================================================
# HELPER: compute weighted language score from a list of language names
# =============================================================================
def compute_weighted_language_score(language_list: list) -> float:
    raw = sum(LANGUAGE_WEIGHTS.get(lang.strip(), 0.0) for lang in language_list)
    return round(min(raw / MAX_LANGUAGE_SCORE, 1.0), 4)


# =============================================================================
# HELPER: compute game_age_days from a release date string or datetime
# =============================================================================
def compute_game_age_days(release_date_input) -> int:
    """
    Accepts:
      - A datetime object
      - An ISO date string: "2024-03-15"
      - A Steam-style string: "Jun 2, 2017" / "Jun 2017" / "2017"
    Returns integer days since release (0 if unparseable or future date).
    """
    if isinstance(release_date_input, datetime):
        dt = release_date_input
    else:
        date_str = str(release_date_input).strip()
        dt = None
        for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%b %Y", "%B %Y", "%Y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return 0
    age = (datetime.today() - dt).days
    return max(age, 0)


# =============================================================================
# HELPER: human-readable age context string for output label
# =============================================================================
def format_age_context(game_age_days: int) -> str:
    """
    Converts game_age_days to a readable string for the output label.
    e.g. 0 days → "at launch", 365 days → "~1 year on Steam", etc.
    """
    if game_age_days <= 30:
        return "at launch"
    elif game_age_days < 365:
        months = round(game_age_days / 30)
        return f"~{months} month{'s' if months != 1 else ''} on Steam"
    else:
        years  = game_age_days / 365
        if years < 1.5:
            return "~1 year on Steam"
        else:
            return f"~{years:.1f} years on Steam"


# =============================================================================
# CORE: compute_derived_features
# Mirrors enrich_prelaunch.py v2 formulas exactly.
# Called BEFORE build_feature_vector so the full feature set is available.
# =============================================================================
def compute_derived_features(form_data: dict) -> dict:
    """
    Compute all composite/derived features from raw form inputs.

    Inputs expected in form_data (all numeric after validation):
      screenshot_count, has_detailed_desc, weighted_language_score,
      has_website, has_support_email, platformindows, platform_mac,
      platform_linux, required_age, full_audio_languages_count,
      has_achievements, has_cloud_save, has_controller_support,
      has_family_sharing, has_vr_support, has_in_app_purchases,
      tag_count, supported_languages (list, optional) or 
      weighted_language_score (float)

    Sets in form_data:
      platform_count, is_mature_content, store_page_score,
      platform_reach, marketing_score, localization_score,
      steam_integration, weighted_language_score (if not already set)
    """
    d = form_data

    def f(key, default=0.0):
        try:
            return float(d.get(key, default))
        except (ValueError, TypeError):
            return float(default)

    # ── Price: form submits USD (e.g. 9.99); model was trained on cents (999) ─
    d["price"]        = round(f("price")        * 100)
    d["initialprice"] = round(f("initialprice") * 100)

    # ── Platform count ────────────────────────────────────────────────────────
    platform_count = f("platform_windows") + f("platform_mac") + f("platform_linux")
    d["platform_count"] = platform_count

    # ── Weighted language score ───────────────────────────────────────────────
    # If the form passed a list of selected language names, recompute.
    # Otherwise use whatever float value was submitted.
    if "selected_languages" in d and isinstance(d["selected_languages"], list):
        d["weighted_language_score"] = compute_weighted_language_score(d["selected_languages"])
    elif "weighted_language_score" not in d or d["weighted_language_score"] is None:
        # Fall back: derive approximate score from count (less accurate)
        count = f("supported_languages_count", 1)
        # Assume average weight per language ≈ 0.25 based on dataset mean
        raw = min(count * 0.25, MAX_LANGUAGE_SCORE)
        d["weighted_language_score"] = round(raw / MAX_LANGUAGE_SCORE, 4)

    wls = f("weighted_language_score")

    # ── Maturity flag ─────────────────────────────────────────────────────────
    d["is_mature_content"] = 1 if f("required_age") >= 17 else 0

    # ── has_detailed_desc: derived from about_length (mirrors enrich script) ─
    d["has_detailed_desc"] = 1 if f("about_length") > 500 else 0

    # ── category_count: sum of all Steam feature toggles + multiplayer ────────
    d["category_count"] = int(
        f("has_achievements") +
        f("has_cloud_save") +
        f("has_controller_support") +
        f("has_vr_support") +
        f("has_in_app_purchases") +
        f("has_family_sharing") +
        f("is_multiplayer")
    )

    # ── store_page_score (v2: no has_trailer term) ────────────────────────────
    d["store_page_score"] = (
        min(f("screenshot_count"), 10) / 10 * 0.40 +
        f("has_detailed_desc")            * 0.25 +
        wls                               * 0.20 +
        f("has_website")                  * 0.10 +
        f("has_support_email")            * 0.05
    )

    # ── platform_reach ────────────────────────────────────────────────────────
    d["platform_reach"] = platform_count / 3.0

    # ── marketing_score (v2: no has_trailer term) ─────────────────────────────
    d["marketing_score"] = (
        f("has_website")                              * 0.35 +
        min(f("screenshot_count"), 10) / 10 * 0.45 +
        f("has_support_email")                        * 0.20
    )

    # ── steam_integration (v2: no trading_cards / workshop) ───────────────────
    d["steam_integration"] = (
        f("has_achievements")        * 0.35 +
        f("has_cloud_save")          * 0.25 +
        f("has_controller_support")  * 0.25 +
        f("has_family_sharing")      * 0.15
    )

    # ── localization_score ────────────────────────────────────────────────────
    d["localization_score"] = (
        wls                                                   * 0.75 +
        min(f("full_audio_languages_count"), 10) / 10 * 0.25
    )

    return d


# =============================================================================
# CORE: build_feature_vector
# =============================================================================
def build_feature_vector(form_data: dict) -> pd.DataFrame:
    """
    Convert raw (already-validated + derived) form input into a properly
    ordered and scaled feature vector ready for the stacked ensemble.

    ALL_FEATURES order must match exactly what was used during training.
    Missing features default to 0.0 (safety net — real validation is upstream).
    """
    row = {}
    for feat in ALL_FEATURES:
        val = form_data.get(feat, 0)
        try:
            row[feat] = float(val)
        except (ValueError, TypeError):
            row[feat] = 0.0

    X = pd.DataFrame([row], columns=ALL_FEATURES)

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=ALL_FEATURES
    )
    return X_scaled


# =============================================================================
# PUBLIC API: predict()
# =============================================================================
def predict(form_data: dict) -> dict:
    """
    Run the full stacked ensemble prediction pipeline.

    Parameters
    ----------
    form_data : dict
        Cleaned & validated feature values from the Flask form or API.
        Must already have compute_derived_features() applied upstream.

    Returns
    -------
    dict with keys:
        predicted_class, predicted_label, predicted_range, predicted_tier,
        confidence, probabilities, class_labels, class_tiers,
        base_probs, game_age_context, output_label

    Raises
    ------
    RuntimeError
        If any stage of the pipeline fails.
    """
    try:
        # 1. Build and scale feature vector
        X_scaled = build_feature_vector(form_data)

        # 2. Base model probabilities
        rf_probs  = rf_model.predict_proba(X_scaled)
        gb_probs  = gb_model.predict_proba(X_scaled)
        xgb_probs = xgb_model.predict_proba(X_scaled)

        # 3. Stack into meta-features
        meta_features = np.hstack([rf_probs, gb_probs, xgb_probs])

        # 4. Meta-learner final prediction
        final_probs     = meta_model.predict_proba(meta_features)[0]
        predicted_class = int(np.argmax(final_probs))
        confidence      = float(final_probs[predicted_class]) * 100

    except Exception as exc:
        raise RuntimeError(f"Prediction pipeline failure: {exc}") from exc

    # 5. Build game_age context string for UI
    game_age_days = int(float(form_data.get("game_age_days", 0)))
    age_context   = format_age_context(game_age_days)

    # 6. Output label — "lifetime" framing with age context
    class_info   = CLASS_RANGES.get(predicted_class, _UNKNOWN_CLASS)
    output_label = (
        f"Predicted lifetime owner tier — estimated total owners your game is "
        f"likely to accumulate over its commercial lifespan on Steam "
        f"({age_context})."
    )

    return {
        "predicted_class":  predicted_class,
        "predicted_label":  class_info["label"],
        "predicted_range":  class_info["range"],
        "predicted_tier":   class_info["tier"],
        "confidence":       round(confidence, 1),
        "probabilities":    [round(float(p) * 100, 1) for p in final_probs],
        "class_labels":     [CLASS_RANGES.get(i, _UNKNOWN_CLASS)["label"] for i in range(N_CLASSES)],
        "class_tiers":      [CLASS_RANGES.get(i, _UNKNOWN_CLASS)["tier"]  for i in range(N_CLASSES)],
        "base_probs": {
            "Random Forest":     [round(float(p) * 100, 1) for p in rf_probs[0]],
            "Gradient Boosting": [round(float(p) * 100, 1) for p in gb_probs[0]],
            "XGBoost":           [round(float(p) * 100, 1) for p in xgb_probs[0]],
        },
        "game_age_days":    game_age_days,
        "game_age_context": age_context,
        "output_label":     output_label,
    }
