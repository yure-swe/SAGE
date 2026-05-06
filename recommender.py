# =============================================================================
# recommender.py — Prescriptive Recommendations Logic  (v2)
# =============================================================================
# Loads prescriptive_rules.csv once at startup.
# Generates actionable recommendations based on user input + SHAP rules.
#
# Changes from v1:
#   - Removed stale FEATURE_LABELS and RECOMMENDATIONS for dropped features:
#       has_trailer, trailer_count, has_trading_cards, has_workshop,
#       dlc_count, has_publisher, is_solo_dev, publisher_backing
#   - Added labels/recommendations for:
#       weighted_language_score, game_age_days, tag binary features
#   - Tag binary features are handled generically (dynamic from TAG_FEATURES)
#   - weighted_language_score replaces supported_languages_count as the
#     primary language signal
#   - "positives" now also checks weighted_language_score and tag coverage
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
    # Pricing
    "price":                      "Price (USD)",
    "initialprice":               "Initial Price (USD)",
    "is_free":                    "Free to Play",
    # Timing
    "release_month":              "Release Month",
    "game_age_days":              "Time on Steam",
    # Store page
    "screenshot_count":           "Screenshot Count",
    "about_length":               "Description Length",
    "has_detailed_desc":          "Detailed Description",
    "has_website":                "Official Website",
    "has_support_email":          "Support Email",
    # Platform
    "platform_windows":           "Windows Support",
    "platform_mac":               "Mac Support",
    "platform_linux":             "Linux Support",
    "platform_count":             "Platform Count",
    # Languages
    "supported_languages_count":  "Languages Supported (count)",
    "full_audio_languages_count": "Full Audio Languages",
    "weighted_language_score":    "Language Coverage Score",
    # Audience
    "required_age":               "Required Age",
    "is_mature_content":          "Mature Content",
    # Steam features
    "has_achievements":           "Steam Achievements",
    "achievement_count":          "Achievement Count",
    "has_cloud_save":             "Steam Cloud Save",
    "has_controller_support":     "Controller Support",
    "has_vr_support":             "VR Support",
    "has_in_app_purchases":       "In-App Purchases",
    "has_family_sharing":         "Family Sharing",
    "category_count":             "Steam Categories",
    # Tags
    "tag_count":                  "Number of Tags",

    "is_multiplayer":             "Multiplayer",
    # Packaging
    "package_count":              "Package Count",
    "sku_count":                  "SKU Count",
    # Composite scores
    "store_page_score":           "Store Page Quality Score",
    "marketing_score":            "Marketing Score",
    "localization_score":         "Localization Score",
    "steam_integration":          "Steam Integration Score",
    "platform_reach":             "Platform Reach Score",
}

# ── Static recommendation rules ───────────────────────────────────────────────
# Maps feature name → (condition_fn, recommendation_text)
# condition_fn(value) returns True when the recommendation applies.
#
# NOTE: Tag binary features (tag_*) are handled generically below.
#       Only features actually in the v2 model appear here.
RECOMMENDATIONS = {
    "screenshot_count": (
        lambda v: v < 5,
        "Add more screenshots — aim for at least 5–8. A visually rich store page "
        "dramatically improves first impressions and Steam search discoverability."
    ),
    "weighted_language_score": (
        lambda v: v < 0.30,
        "Expand your language support — particularly English, Simplified Chinese, "
        "Russian, and German. These four languages cover Steam's largest market segments "
        "and meaningfully increase your addressable audience."
    ),
    "supported_languages_count": (
        lambda v: v < 3,
        "Support at least 3–5 languages. Even basic text localization to major languages "
        "expands your reach significantly."
    ),
    "is_multiplayer": (
        lambda v: v == 0,
        "Consider adding multiplayer support — multiplayer games broaden your audience "
        "and tend to have stronger long-term word-of-mouth."
    ),
    "has_achievements": (
        lambda v: v == 0,
        "Implement Steam Achievements — they increase engagement, show up in player "
        "profiles, and improve your Steam integration score."
    ),
    "has_cloud_save": (
        lambda v: v == 0,
        "Enable Steam Cloud Save — a low-effort feature that improves player experience "
        "across devices and signals a well-maintained game."
    ),
    "has_controller_support": (
        lambda v: v == 0,
        "Add controller support — this broadens your audience to Steam Deck users "
        "and console-style players, who represent a growing segment of the Steam market."
    ),
    "platform_count": (
        lambda v: v < 2,
        "Consider Mac or Linux support — Proton compatibility makes Windows-to-Linux "
        "a lower-effort expansion than it used to be, and increases platform reach score."
    ),
    "has_website": (
        lambda v: v == 0,
        "Create an official game website — it signals developer professionalism, "
        "supports SEO discoverability, and contributes to your marketing score."
    ),
    "has_support_email": (
        lambda v: v == 0,
        "Add a support contact email — improves perceived developer credibility "
        "and player trust signals on your store page."
    ),
    "about_length": (
        lambda v: v < 500,
        "Write a more detailed game description — a thorough 'About This Game' section "
        "(500+ characters) improves store page quality and helps Steam's search ranking."
    ),
    "full_audio_languages_count": (
        lambda v: v == 0,
        "Consider adding full audio in at least one additional language — "
        "it signals higher production value to potential buyers."
    ),
    "achievement_count": (
        lambda v: v < 10,
        "Add more Steam Achievements (aim for 10+) — they increase replayability, "
        "provide long-term engagement goals, and are entirely free to implement."
    ),
    "tag_count": (
        lambda v: v < 5,
        "Use more Steam tags — tags are how Steam's recommendation algorithm categorises "
        "and surfaces your game to the right audience. Aim for 10–15 accurate, relevant tags."
    ),
    "has_family_sharing": (
        lambda v: v == 0,
        "Enable Family Sharing — it costs nothing and increases the potential number "
        "of players who can access your game."
    ),
    "has_vr_support": (
        lambda v: v == 0,
        "If your game has any VR compatibility, flag it — VR users actively search "
        "Steam for VR-compatible titles and it increases discoverability."
    ),
}


def _label_for(feature: str) -> str:
    """Return human-readable label for any feature, including tag_* columns."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("tag_"):
        return feature.removeprefix("tag_").replace("_", " ").title() + " (tag)"
    return feature.replace("_", " ").title()


def get_recommendations(form_data: dict, predicted_class: int,
                        max_recommendations: int = 5) -> dict:
    """
    Generate prescriptive recommendations based on user input and SHAP rules.

    Parameters
    ----------
    form_data        : dict — cleaned + derived form input
    predicted_class  : int  — predicted owner tier class (0–5)
    max_recommendations : int — max items per list

    Returns
    -------
    dict with keys:
        positives       list  — things working in the game's favour
        improvements    list  — actionable recommendations
        impact_scores   dict  — feature → SHAP impact score
        predicted_class int
    """
    positives    = []
    improvements = []

    # SHAP impact scores for display context
    impact_scores = dict(zip(df_rules["feature"], df_rules["impact_score"]))

    # Work through rules ranked by SHAP impact (highest first)
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

        label = _label_for(feat)

        # ── Positives: feature is already contributing favourably ─────────────
        if direction == "higher is better" and val > 0.5:
            positives.append({
                "feature": feat,
                "label":   label,
                "value":   val,
                "impact":  round(impact, 4),
                "message": f"{label} is working in your favour.",
            })
        elif direction == "lower is better" and val < 0.5:
            positives.append({
                "feature": feat,
                "label":   label,
                "value":   val,
                "impact":  round(impact, 4),
                "message": f"{label} is appropriately set.",
            })

        # ── Improvements: static rules for specific features ──────────────────
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

    # ── Generic tag improvement: if user has few tags selected ────────────────
    # Check tag binary columns (tag_*) — if almost none selected, flag it
    tag_cols_selected = sum(
        1 for k, v in form_data.items()
        if k.startswith("tag_") and k != "tag_count"
        and _is_truthy(v)
    )
    if tag_cols_selected < 3 and len(improvements) < max_recommendations:
        improvements.append({
            "feature": "tag_count",
            "label":   "Steam Tags Selected",
            "value":   tag_cols_selected,
            "impact":  impact_scores.get("tag_count", 0.0),
            "message": (
                "Select more Steam tags from the tag checklist — tags tell Steam's "
                "recommendation engine exactly what your game is. Aim for 10–15 "
                "accurate, relevant tags that match your game's genre and mechanics."
            ),
        })

    # ── Language improvement: if weighted_language_score is very low ──────────
    wls = float(form_data.get("weighted_language_score", 0.0))
    if wls < 0.15 and "weighted_language_score" not in [i["feature"] for i in improvements]:
        improvements.append({
            "feature": "weighted_language_score",
            "label":   "Language Coverage",
            "value":   wls,
            "impact":  impact_scores.get("weighted_language_score", 0.0),
            "message": (
                "Your language coverage is very low. At a minimum, ensure English is "
                "supported. Adding Simplified Chinese and Russian dramatically expands "
                "your potential Steam audience."
            ),
        })

    # Cap lists
    positives    = positives[:max_recommendations]
    improvements = improvements[:max_recommendations]

    # Fallback positive if nothing found
    if not positives:
        positives.append({
            "feature": "general",
            "label":   "Game Profile",
            "value":   1,
            "impact":  0.0,
            "message": "Your game profile has been analysed successfully.",
        })

    return {
        "positives":       positives,
        "improvements":    improvements,
        "impact_scores":   impact_scores,
        "predicted_class": predicted_class,
    }


def _is_truthy(val) -> bool:
    """Return True if val represents a checked/active binary feature."""
    try:
        return float(val) > 0.5
    except (ValueError, TypeError):
        return False
