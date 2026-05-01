# Centralized validation rules for all user-supplied form / JSON inputs.
# Used by both the HTML dashboard route ("/") and the JSON API ("/api/predict").
#
# Design goals:
#    Fail loudly on bad input instead of silently coercing to 0.0
#    Enforce sane numeric ranges (e.g. month 1-12, non-negative counts)
#    Return human-readable error messages the UI can display
#    Never crash the server on malformed payloads
# =============================================================================

from typing import Any, Dict, List, Tuple

# ── Field specifications ─────────────────────────────────────────────────────
# Mirrors FORM_SECTIONS in app.py but in a flat, validation-friendly shape.
# Each entry: (min, max, kind)  where kind ∈ {"int", "float", "bool"}
FIELD_SPECS: Dict[str, Tuple[float, float, str]] = {
    # Pricing
    "price":                       (0,    200,  "float"),
    "initialprice":                (0,    200,  "float"),
    "is_free":                     (0,    1,    "bool"),

    # Release
    "release_month":               (1,    12,   "int"),

    # Genre toggles
    "Action":     (0, 1, "bool"), "Adventure":  (0, 1, "bool"),
    "RPG":        (0, 1, "bool"), "Strategy":   (0, 1, "bool"),
    "Simulation": (0, 1, "bool"), "Indie":      (0, 1, "bool"),
    "Sports":     (0, 1, "bool"), "Racing":     (0, 1, "bool"),

    # Platform
    "platform_windows": (0, 1, "bool"),
    "platform_mac":     (0, 1, "bool"),
    "platform_linux":   (0, 1, "bool"),
    "platform_count":   (0, 3, "int"),

    # Languages
    "supported_languages_count":  (0, 50, "int"),
    "full_audio_languages_count": (0, 20, "int"),

    # Store page
    "screenshot_count":  (0, 20,   "int"),
    "has_trailer":       (0, 1,    "bool"),
    "trailer_count":     (0, 10,   "int"),
    "about_length":      (0, 5000, "int"),
    "has_detailed_desc": (0, 1,    "bool"),
    "has_website":       (0, 1,    "bool"),
    "has_support_email": (0, 1,    "bool"),

    # Developer / Publisher
    "developer_count":   (1, 20, "int"),
    "publisher_count":   (0, 10, "int"),
    "has_publisher":     (0, 1,  "bool"),
    "is_solo_dev":       (0, 1,  "bool"),
    "required_age":      (0, 18, "int"),
    "is_mature_content": (0, 1,  "bool"),

    # Steam features
    "has_achievements":       (0, 1,   "bool"),
    "achievement_count":      (0, 500, "int"),
    "has_trading_cards":      (0, 1,   "bool"),
    "has_workshop":           (0, 1,   "bool"),
    "has_cloud_save":         (0, 1,   "bool"),
    "has_controller_support": (0, 1,   "bool"),
    "has_vr_support":         (0, 1,   "bool"),
    "has_in_app_purchases":   (0, 1,   "bool"),
    "has_family_sharing":     (0, 1,   "bool"),
    "category_count":         (0, 15,  "int"),

    # Tags & Community
    "tag_count":           (0, 20,   "int"),
    "has_multiplayer_tag": (0, 1,    "bool"),
    "top_tag_votes_total": (0, 5000, "int"),
    "top_tag_votes_mean":  (0, 1000, "int"),
    "is_multiplayer":      (0, 1,    "bool"),

    # Packaging & DLC
    "dlc_count":     (0, 50, "int"),
    "package_count": (1, 10, "int"),
    "sku_count":     (1, 20, "int"),
}

# Maximum keys we'll accept in a single payload (defense against bloat)
MAX_PAYLOAD_KEYS = 200


def _coerce(value: Any, kind: str) -> float:
    """Coerce a single value to numeric. Raises ValueError on failure."""
    if value is None or value == "":
        raise ValueError("empty value")
    # HTML toggles often arrive as "on"/"off"
    if kind == "bool":
        if isinstance(value, bool):
            return float(value)
        sval = str(value).strip().lower()
        if sval in ("1", "true", "on", "yes"):  return 1.0
        if sval in ("0", "false", "off", "no"): return 0.0
        # Fall through — let float() try
    num = float(value)
    if kind == "int":
        if num != int(num):
            raise ValueError("expected integer")
    return num


def validate_form_data(
    raw: Any,
    *,
    strict: bool = False,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Validate and clean an incoming feature payload.

    Parameters
    ----------
    raw : dict-like
        The submitted form/JSON data.
    strict : bool
        If True, unknown keys are rejected. If False, they're ignored.

    Returns
    -------
    (cleaned, errors)
        cleaned : dict of {feature: float}  — only contains keys with valid values
        errors  : list of human-readable error messages (empty = all good)
    """
    errors: List[str] = []
    cleaned: Dict[str, float] = {}

    if not isinstance(raw, dict):
        return {}, ["Payload must be a JSON object / form mapping."]

    if len(raw) > MAX_PAYLOAD_KEYS:
        return {}, [f"Payload too large ({len(raw)} keys, max {MAX_PAYLOAD_KEYS})."]

    for key, value in raw.items():
        spec = FIELD_SPECS.get(key)
        if spec is None:
            if strict:
                errors.append(f"Unknown field: '{key}'")
            continue

        lo, hi, kind = spec

        # Treat missing/blank as 0 for booleans (unchecked toggle behavior),
        # but require an explicit value for numeric fields.
        if (value is None or value == "") and kind == "bool":
            cleaned[key] = 0.0
            continue

        try:
            num = _coerce(value, kind)
        except (ValueError, TypeError):
            errors.append(f"'{key}': invalid value '{value}' (expected {kind})")
            continue

        if num < lo or num > hi:
            errors.append(f"'{key}': {num} is out of range [{lo}, {hi}]")
            continue

        cleaned[key] = num

    return cleaned, errors
