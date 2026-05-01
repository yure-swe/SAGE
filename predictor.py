# =============================================================================
# predictor.py — Model Loading & Prediction Logic
# =============================================================================
# Loads all .pkl files ONCE at startup (not per request).
# Exposes a single predict() function used by app.py.
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib

# ── Paths — relative to app/ directory ───────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# ── Load all artifacts once at module import ──────────────────────────────────
print("[predictor] Loading models...")

rf_model    = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
gb_model    = joblib.load(os.path.join(MODELS_DIR, "gb_classifier.pkl"))
xgb_model   = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.pkl"))
meta_model  = joblib.load(os.path.join(MODELS_DIR, "meta_classifier.pkl"))
scaler      = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feat_dict   = joblib.load(os.path.join(MODELS_DIR, "feature_dict.pkl"))

ALL_FEATURES     = feat_dict["all_features"]
NUMERIC_FEATURES = feat_dict["numeric_features"]
CLASS_LABELS     = feat_dict["class_labels"]
N_CLASSES        = feat_dict["n_classes"]

print(f"[predictor] Models loaded. Features: {len(ALL_FEATURES)}, Classes: {N_CLASSES}")

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_RANGES = {
    0: {"label": "≤10K",   "range": "Up to 10,000 owners",              "tier": "Common Indie"},
    1: {"label": "35K",    "range": "10,000 – 35,000 owners",           "tier": "Niche"},
    2: {"label": "75K",    "range": "35,000 – 75,000 owners",           "tier": "Growing"},
    3: {"label": "150K",   "range": "75,000 – 150,000 owners",          "tier": "Established"},
    4: {"label": "350K",   "range": "150,000 – 350,000 owners",         "tier": "Popular"},
    5: {"label": "≥750K",  "range": "750,000+ owners",                  "tier": "Breakout Hit"},
}

# Safe fallback metadata if predicted_class is somehow out of bounds.
_UNKNOWN_CLASS = {"label": "Unknown", "range": "Unknown", "tier": "Unknown"}


def build_feature_vector(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form input into a properly ordered and scaled feature vector
    ready for prediction.

    NOTE: This still defensively coerces to 0.0 as a *last-resort* safety net,
    but real validation now happens upstream in validation.py before this
    function is ever called.
    """
    row = {}
    for feat in ALL_FEATURES:
        val = form_data.get(feat, 0)
        try:
            row[feat] = float(val)
        except (ValueError, TypeError):
            row[feat] = 0.0

    X = pd.DataFrame([row], columns=ALL_FEATURES)

    # Scale using the fitted scaler from training
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=ALL_FEATURES
    )
    return X_scaled


def predict(form_data: dict) -> dict:
    """
    Run the full stacked ensemble prediction pipeline.

    Raises
    ------
    RuntimeError
        If any stage of the pipeline fails. Callers (app.py) should catch
        this and surface a friendly error to the user.
    """
    try:
        # 1. Build and scale feature vector
        X_scaled = build_feature_vector(form_data)

        # 2. Get base model probabilities
        rf_probs  = rf_model.predict_proba(X_scaled)
        gb_probs  = gb_model.predict_proba(X_scaled)
        xgb_probs = xgb_model.predict_proba(X_scaled)

        # 3. Stack into meta-features
        meta_features = np.hstack([rf_probs, gb_probs, xgb_probs])

        # 4. Meta-learner final prediction
        final_probs = meta_model.predict_proba(meta_features)[0]
        predicted_class = int(np.argmax(final_probs))
        confidence = float(final_probs[predicted_class]) * 100
    except Exception as exc:
        # Wrap any low-level numpy/sklearn error in a clean exception
        raise RuntimeError(f"Prediction pipeline failure: {exc}") from exc

    # 5. Build response — guard CLASS_RANGES lookup
    class_info = CLASS_RANGES.get(predicted_class, _UNKNOWN_CLASS)

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
            "Random Forest":      [round(float(p) * 100, 1) for p in rf_probs[0]],
            "Gradient Boosting":  [round(float(p) * 100, 1) for p in gb_probs[0]],
            "XGBoost":            [round(float(p) * 100, 1) for p in xgb_probs[0]],
        },
    }
