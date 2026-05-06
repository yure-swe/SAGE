# =============================================================================
# validation.py — Server-Side Input Validation  (v2)
# =============================================================================
# Validates and coerces all form / API inputs before they reach predictor.py.
#
# Changes from v1:
#   REMOVED from FIELD_SPECS (dropped from model):
#     - has_trailer, trailer_count
#     - has_trading_cards, has_workshop
#     - dlc_count
#     - is_solo_dev, has_publisher, publisher_count, developer_count
#     - has_multiplayer_tag
#     - json_price_raw, has_support_url, publisher_backing
#     - Indie (genre flag)
#
#   ADDED to FIELD_SPECS:
#     - game_age_days         (computed from release_date; 0 = launch day)
#     - weighted_language_score (0.0–1.0 normalized)
#     - short_desc_length     (if present in CSV)
#
#   NOTES:
#     - Tag binary columns (tag_*) are validated dynamically — any 0/1 column
#       that starts with tag_ passes through without being listed in FIELD_SPECS.
#     - weighted_language_score and game_age_days are computed by preprocess_form()
#       before validate_form_data() is called, so they are expected to be present.
# =============================================================================

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# FIELD_SPECS
# ---------------------------------------------------------------------------
# Each entry: (type_fn, min_val, max_val, default)
#   type_fn   — callable to coerce the raw string value
#   min_val   — inclusive lower bound (None = no check)
#   max_val   — inclusive upper bound (None = no check)
#   default   — value used if field is missing from the submission
# ---------------------------------------------------------------------------

FIELD_SPECS: dict[str, tuple] = {
    # ── Pricing ───────────────────────────────────────────────────────────────
    "price":                        (float,  0.0,    200.0,   9.99),
    "initialprice":                 (float,  0.0,    200.0,   9.99),
    "is_free":                      (int,    0,      1,       0),

    # ── Timing ────────────────────────────────────────────────────────────────
    "release_month":                (int,    1,      12,      10),
    "game_age_days":                (int,    0,      20000,   0),

    # ── Store page ────────────────────────────────────────────────────────────
    "screenshot_count":             (int,    0,      50,      5),
    "about_length":                 (int,    0,      10000,   500),
    "short_desc_length":            (int,    0,      1000,    0),
    "has_detailed_desc":            (int,    0,      1,       0),
    "has_website":                  (int,    0,      1,       0),
    "has_support_email":            (int,    0,      1,       0),

    # ── Platform ─────────────────────────────────────────────────────────────
    "platform_windows":             (int,    0,      1,       1),
    "platform_mac":                 (int,    0,      1,       0),
    "platform_linux":               (int,    0,      1,       0),
    "platform_count":               (int,    1,      3,       1),

    # ── Languages ─────────────────────────────────────────────────────────────
    "supported_languages_count":    (int,    0,      50,      1),
    "full_audio_languages_count":   (int,    0,      20,      0),
    "weighted_language_score":      (float,  0.0,    1.0,     0.0),

    # ── Audience ──────────────────────────────────────────────────────────────
    "required_age":                 (int,    0,      18,      0),
    "is_mature_content":            (int,    0,      1,       0),

    # ── Steam features ────────────────────────────────────────────────────────
    "has_achievements":             (int,    0,      1,       0),
    "achievement_count":            (int,    0,      1000,    0),
    "has_cloud_save":               (int,    0,      1,       0),
    "has_controller_support":       (int,    0,      1,       0),
    "has_vr_support":               (int,    0,      1,       0),
    "has_in_app_purchases":         (int,    0,      1,       0),
    "has_family_sharing":           (int,    0,      1,       0),
    "category_count":               (int,    0,      20,      3),

    # ── Genre flags ───────────────────────────────────────────────────────────
    "Action":                       (int,    0,      1,       0),
    "Adventure":                    (int,    0,      1,       0),
    "RPG":                          (int,    0,      1,       0),
    "Strategy":                     (int,    0,      1,       0),
    "Simulation":                   (int,    0,      1,       0),
    "Sports":                       (int,    0,      1,       0),
    "Racing":                       (int,    0,      1,       0),
    # Note: "Indie" intentionally removed — not a genre in v2

    # ── Tags (volume metrics) ─────────────────────────────────────────────────
    "tag_count":                    (int,    0,      30,      5),


    # ── Multiplayer ───────────────────────────────────────────────────────────
    "is_multiplayer":               (int,    0,      1,       0),

    # ── Packaging ─────────────────────────────────────────────────────────────
    "package_count":                (int,    1,      20,      1),
    "sku_count":                    (int,    1,      50,      1),

    # ── Derived composite scores (computed by predictor.compute_derived_features)
    # Listed here so validation accepts them if submitted; defaults prevent errors.
    "store_page_score":             (float,  0.0,    1.0,     0.0),
    "platform_reach":               (float,  0.0,    1.0,     0.0),
    "marketing_score":              (float,  0.0,    1.0,     0.0),
    "localization_score":           (float,  0.0,    1.0,     0.0),
    "steam_integration":            (float,  0.0,    1.0,     0.0),
}


def validate_form_data(
    raw_data: dict[str, Any],
    strict: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and coerce all form/API inputs.

    Parameters
    ----------
    raw_data : dict  — raw string values from request.form or JSON body
    strict   : bool  — if True, raise errors on unknown fields;
                       if False (default), silently pass them through

    Returns
    -------
    (cleaned_data, errors)
        cleaned_data : dict  — coerced values + defaults for missing fields
        errors       : list  — human-readable error strings (empty = ok)
    """
    cleaned = {}
    errors  = []

    for field, (type_fn, min_val, max_val, default) in FIELD_SPECS.items():
        raw = raw_data.get(field)

        # Missing or empty → apply default
        if raw is None or str(raw).strip() == "":
            cleaned[field] = default
            continue

        # Coerce type
        try:
            val = type_fn(raw)
        except (ValueError, TypeError):
            errors.append(
                f"'{field}': expected {type_fn.__name__}, got {repr(raw)!r}."
            )
            cleaned[field] = default
            continue

        # Range check
        if min_val is not None and val < min_val:
            errors.append(
                f"'{field}': value {val} is below minimum {min_val}."
            )
            val = min_val

        if max_val is not None and val > max_val:
            errors.append(
                f"'{field}': value {val} exceeds maximum {max_val}."
            )
            val = max_val

        cleaned[field] = val

    # Pass through tag binary columns (tag_*) — validated as 0/1 int
    for key, raw in raw_data.items():
        if key.startswith("tag_") and key not in cleaned:
            try:
                val = int(float(raw))
                cleaned[key] = max(0, min(1, val))   # clamp to 0/1
            except (ValueError, TypeError):
                cleaned[key] = 0

    # Pass through any fields not in FIELD_SPECS when not strict
    # (e.g. selected_languages list, release_date string — consumed upstream)
    if not strict:
        for key, val in raw_data.items():
            if key not in cleaned:
                cleaned[key] = val

    return cleaned, errors
