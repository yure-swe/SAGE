# =============================================================================
# recommender.py — Prescriptive Recommendations Logic
# =============================================================================
# Loads prescriptive_rules.csv once at startup.
# Generates actionable recommendations based on user input + SHAP rules.
# =============================================================================

import os
import pandas as pd

# ── Load prescriptive rules once ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "models", "shap_outputs", "prescriptive_rules.csv")

print("[recommender] Loading prescriptive rules...")
df_rules = pd.read_csv(RULES_PATH)
print(f"[recommender] Loaded {len(df_rules)} rules.")

# ── Human-readable feature labels ────────────────────────────────────────────
FEATURE_LABELS = {
    "has_trailer":                "Game Trailer",
    "screenshot_count":           "Screenshot Count",
    "supported_languages_count":  "Supported Languages",
    "has_publisher":              "Publisher Backing",
    "is_multiplayer":             "Multiplayer Support",
    "has_achievements":           "Steam Achievements",
    "has_workshop":               "Steam Workshop",
    "has_cloud_save":             "Steam Cloud Save",
    "has_controller_support":     "Controller Support",
    "has_trading_cards":          "Steam Trading Cards",
    "full_audio_languages_count": "Full Audio Languages",
    "platform_count":             "Platform Count (Win/Mac/Linux)",
    "store_page_score":           "Store Page Quality",
    "marketing_score":            "Marketing Investment",
    "localization_score":         "Localization Score",
    "steam_integration":          "Steam Ecosystem Integration",
    "publisher_backing":          "Publisher Backing Score",
    "tag_count":                  "Number of Tags",
    "achievement_count":          "Achievement Count",
    "has_website":                "Official Website",
    "has_support_email":          "Support Email",
    "dlc_count":                  "DLC Count",
    "trailer_count":              "Trailer Count",
    "about_length":               "Description Length",
    "has_detailed_desc":          "Detailed Description",
    "is_solo_dev":                "Solo Developer",
    "price":                      "Price (USD)",
    "initialprice":               "Initial Price (USD)",
    "is_free":                    "Free to Play",
    "release_month":              "Release Month",
}

# ── Recommendation templates ──────────────────────────────────────────────────
# Maps feature name → (condition_fn, recommendation_text)
# condition_fn(value) returns True when the recommendation applies
RECOMMENDATIONS = {
    "has_trailer": (
        lambda v: v == 0,
        "Add a game trailer — games with trailers significantly increase "
        "visibility and predicted ownership tier."
    ),
    "screenshot_count": (
        lambda v: v < 5,
        "Add more screenshots (aim for at least 5–8) — a rich store page "
        "improves first impressions and discoverability."
    ),
    "supported_languages_count": (
        lambda v: v < 5,
        "Support more languages — localization to at least 5 languages "
        "expands your addressable market considerably."
    ),
    "has_publisher": (
        lambda v: v == 0,
        "Consider partnering with a publisher — publisher-backed games "
        "reach higher ownership tiers more consistently."
    ),
    "is_multiplayer": (
        lambda v: v == 0,
        "Adding multiplayer support can significantly broaden your audience "
        "and improve long-term retention."
    ),
    "has_achievements": (
        lambda v: v == 0,
        "Implement Steam Achievements — they increase engagement and "
        "improve your Steam ecosystem integration score."
    ),
    "has_workshop": (
        lambda v: v == 0,
        "Steam Workshop support drives long-term engagement and community "
        "content creation, particularly for strategy and simulation games."
    ),
    "has_cloud_save": (
        lambda v: v == 0,
        "Enable Steam Cloud Save — a low-effort feature that improves "
        "player experience and Steam integration score."
    ),
    "has_controller_support": (
        lambda v: v == 0,
        "Add controller support — broadens your potential audience to "
        "console-style players and Steam Deck users."
    ),
    "has_trading_cards": (
        lambda v: v == 0,
        "Implement Steam Trading Cards — increases store page engagement "
        "and provides an additional monetization layer."
    ),
    "platform_count": (
        lambda v: v < 2,
        "Consider releasing on Mac or Linux — additional platform support "
        "expands your reachable audience."
    ),
    "has_website": (
        lambda v: v == 0,
        "Create an official game website — it signals developer "
        "professionalism and supports marketing efforts."
    ),
    "has_support_email": (
        lambda v: v == 0,
        "Add a support email — improves perceived developer credibility "
        "and player trust."
    ),
    "about_length": (
        lambda v: v < 500,
        "Write a more detailed game description — a thorough 'About This Game' "
        "section improves store page quality and search discoverability."
    ),
    "full_audio_languages_count": (
        lambda v: v == 0,
        "Consider adding full audio in at least one additional language — "
        "it signals higher production value to potential buyers."
    ),
    "achievement_count": (
        lambda v: v < 10,
        "Add more Steam Achievements (aim for 10+) — they increase "
        "replayability and player engagement."
    ),
}


def get_recommendations(form_data: dict, predicted_class: int,
                         max_recommendations: int = 5) -> dict:
    """
    Generate prescriptive recommendations based on user input and SHAP rules.

    Parameters
    ----------
    form_data        : dict — raw form input from user
    predicted_class  : int  — predicted owner tier class (0–5)
    max_recommendations : int — max items to return

    Returns
    -------
    dict with keys:
        positives       list — things working in the game's favor
        improvements    list — actionable recommendations
        impact_scores   dict — feature → SHAP impact score
    """
    positives    = []
    improvements = []

    # Get SHAP impact scores for context
    impact_scores = dict(zip(df_rules["feature"], df_rules["impact_score"]))

    # Sort recommendations by SHAP impact (highest impact first)
    sorted_rules = df_rules.sort_values("impact_score", ascending=False)

    for _, rule_row in sorted_rules.iterrows():
        feat      = rule_row["feature"]
        direction = rule_row["direction"]
        impact    = rule_row["impact_score"]

        if feat not in form_data:
            continue

        try:
            val = float(form_data[feat])
        except (ValueError, TypeError):
            val = 0.0

        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())

        # Check if this feature is a positive signal
        if direction == "higher is better" and val > 0.5:
            positives.append({
                "feature": feat,
                "label":   label,
                "value":   val,
                "impact":  round(impact, 4),
                "message": f"{label} is working in your favor.",
            })
        elif direction == "lower is better" and val < 0.5:
            positives.append({
                "feature": feat,
                "label":   label,
                "value":   val,
                "impact":  round(impact, 4),
                "message": f"{label} is appropriately set.",
            })

        # Check if this feature needs improvement
        if feat in RECOMMENDATIONS:
            condition_fn, rec_text = RECOMMENDATIONS[feat]
            if condition_fn(val):
                improvements.append({
                    "feature": feat,
                    "label":   label,
                    "value":   val,
                    "impact":  round(impact, 4),
                    "message": rec_text,
                })

    # Cap lists
    positives    = positives[:max_recommendations]
    improvements = improvements[:max_recommendations]

    # If no positives found, add a generic one
    if not positives:
        positives.append({
            "feature": "general",
            "label":   "Game Concept",
            "value":   1,
            "impact":  0,
            "message": "Your game has been submitted for analysis.",
        })

    return {
        "positives":    positives,
        "improvements": improvements,
        "impact_scores": impact_scores,
        "predicted_class": predicted_class,
    }
