#!/usr/bin/env python3
# PRE-LAUNCH FEATURE ENRICHMENT SCRIPT
#
# Thesis: Ensemble Learning for Predicting Game Success
#
# Usage:
#   python3 enrich_prelaunch.py --json /path/to/games.json \
#                               --csv  /path/to/steam_10k_enriched.csv \
#                               --out  steam_10k_prelaunch.csv \
#                               --top-tags 50

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Enrich CSV with pre-launch Steam API features")
parser.add_argument("--json",     required=True, help="Path to the Steam API JSON file")
parser.add_argument("--csv",      required=True, help="Path to steam_10k_enriched.csv")
parser.add_argument("--out",      default="steam_10k_prelaunch.csv", help="Output CSV path")
parser.add_argument("--top-tags", type=int, default=50,
                    help="Number of top tags to one-hot encode (default: 50)")
args = parser.parse_args()

for path, label in [(args.json, "JSON"), (args.csv, "CSV")]:
    if not os.path.exists(path):
        print(f"[ERROR] {label} file not found: {path}")
        sys.exit(1)

TODAY = datetime.today()
TOP_N_TAGS = args.top_tags

# STEAM LANGUAGE WEIGHTS
# Based on Steam hardware survey + platform demographic data.
# Weights reflect relative market size of each language on Steam.
# Used to compute weighted_language_score instead of raw count.
LANGUAGE_WEIGHTS = {
    "English":               1.00,
    "Simplified Chinese":    0.85,
    "Russian":               0.55,
    "German":                0.45,
    "Spanish - Spain":       0.40,
    "French":                0.40,
    "Portuguese - Brazil":   0.35,
    "Japanese":              0.30,
    "Korean":                0.25,
    "Traditional Chinese":   0.20,
    "Polish":                0.15,
    "Turkish":               0.15,
    "Italian":               0.12,
    "Dutch":                 0.10,
    "Czech":                 0.08,
    "Hungarian":             0.07,
    "Romanian":              0.07,
    "Spanish - Latin America": 0.12,
    "Portuguese - Portugal": 0.08,
    "Ukrainian":             0.08,
}
MAX_LANGUAGE_SCORE = sum(LANGUAGE_WEIGHTS.values())  # for normalization

# MULTIPLAYER CATEGORIES
MULTIPLAYER_CATS = {
    "Multi-player", "Online PvP", "Online Co-op",
    "Local Co-op", "Local Multi-Player", "Co-op",
    "MMO", "Cross-Platform Multiplayer"
}

# STEP 1 — Load base CSV
print("  STEP 1: Loading base CSV")

df_base = pd.read_csv(args.csv)
print(f"  Loaded CSV: {df_base.shape[0]:,} rows × {df_base.shape[1]} columns")

# STEP 2 — Load JSON
print("\n  STEP 2: Loading Steam API JSON")

print(f"  Reading: {args.json}")
print("  (This may take a moment for large files...)")

with open(args.json, "r", encoding="utf-8") as f:
    raw = json.load(f)

print(f"  JSON entries loaded: {len(raw):,}")
sample_key = next(iter(raw))
print(f"  Sample key format: '{sample_key}' (type: {type(sample_key).__name__})")

# STEP 3 — First pass: collect all tag names to find top N
print("\n  STEP 3: Discovering top tags across dataset")

tag_frequency = {}  # tag_name -> number of games that have this tag

for appid_str, entry in raw.items():
    tags = entry.get("tags", {}) or {}
    if isinstance(tags, list):
        tags = {t: 1 for t in tags}
    for tag_name in tags.keys():
        tag_frequency[tag_name] = tag_frequency.get(tag_name, 0) + 1

# Sort by frequency and take top N
# Exclude 'Indie' — it describes dev context, not game content
EXCLUDE_TAGS = {"Indie"}
sorted_tags = sorted(
    [(tag, freq) for tag, freq in tag_frequency.items() if tag not in EXCLUDE_TAGS],
    key=lambda x: x[1],
    reverse=True
)

TOP_TAGS = [tag for tag, freq in sorted_tags[:TOP_N_TAGS]]

print(f"\n  Total unique tags found: {len(tag_frequency):,}")
print(f"  Top {TOP_N_TAGS} tags selected (excluding: {EXCLUDE_TAGS}):")
for i, (tag, freq) in enumerate(sorted_tags[:TOP_N_TAGS], 1):
    pct = freq / len(raw) * 100
    print(f"    {i:>3}. {tag:<40} {freq:>5,} games ({pct:.1f}%)")

# Build safe column names for top tags: lowercase, spaces → underscores, strip special chars
def tag_to_col(tag_name):
    """Convert tag name to a safe DataFrame column name."""
    import re
    col = tag_name.lower().strip()
    col = re.sub(r"[^a-z0-9\s]", "", col)   # remove special chars
    col = re.sub(r"\s+", "_", col)           # spaces to underscores
    return f"tag_{col}"

TAG_COLUMNS = {tag: tag_to_col(tag) for tag in TOP_TAGS}
print(f"\n  Sample column names:")
for tag in TOP_TAGS[:5]:
    print(f"    '{tag}' → '{TAG_COLUMNS[tag]}'")

# STEP 4 — Extract pre-launch features from each JSON entry
print("\n  STEP 4: Extracting pre-launch features")

def parse_release_date(date_str):
    """
    Parse Steam release date string to datetime.
    Handles formats: 'Jun 2, 2017', 'Jun 2017', '2017'
    Returns None if unparseable.
    """
    if not date_str:
        return None
    date_str = str(date_str).strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def compute_weighted_language_score(supported_languages):
    """
    Sum weights for each supported language, normalized to [0, 1].
    More meaningful than raw count: covers English + Chinese + Russian
    scores higher than covering 10 minor languages.
    """
    if not supported_languages:
        return 0.0
    total = sum(LANGUAGE_WEIGHTS.get(lang, 0.02) for lang in supported_languages)
    return min(total / MAX_LANGUAGE_SCORE, 1.0)  # cap at 1.0

def extract_prelaunch(appid_str, entry):
    """Extract all pre-launch features from a single JSON entry."""
    release_str = entry.get("release_date", {}).get("date") if isinstance(entry.get("release_date"), dict) else entry.get("release_date")
    release_dt = parse_release_date(release_str)
    row = {"appid": int(appid_str)}

    # ── Game age ──────────────────────────────────────────────────────────────
    if release_dt:
        game_age_days = (TODAY - release_dt).days
        row["game_age_days"] = game_age_days
        age_months = max(game_age_days / 30.0, 1.0)
        row["_age_months"] = age_months
    else:
        row["game_age_days"] = None
        row["_age_months"] = None
    # ── Basic metadata ────────────────────────────────────────────────────────
    row["required_age"]      = int(entry.get("required_age", 0) or 0)
    row["has_website"]       = 1 if entry.get("website") else 0
    row["has_support_url"]   = 1 if entry.get("support_url") else 0
    row["has_support_email"] = 1 if entry.get("support_email") else 0

    # ── Platform support ──────────────────────────────────────────────────────
    row["platform_windows"] = 1 if entry.get("windows") else 0
    row["platform_mac"]     = 1 if entry.get("mac") else 0
    row["platform_linux"]   = 1 if entry.get("linux") else 0
    row["platform_count"]   = row["platform_windows"] + row["platform_mac"] + row["platform_linux"]

    # ── Language support ──────────────────────────────────────────────────────
    supported = entry.get("supported_languages", []) or []
    full_audio = entry.get("full_audio_languages", []) or []
    row["supported_languages_count"]  = len(supported)
    row["full_audio_languages_count"] = len(full_audio)
    row["weighted_language_score"]    = compute_weighted_language_score(supported)

    # ── Store page content ────────────────────────────────────────────────────
    screenshots = entry.get("screenshots", []) or []
    row["screenshot_count"] = len(screenshots)

    about = entry.get("about_the_game", "") or ""
    short = entry.get("short_description", "") or ""
    row["about_length"]      = len(about)
    row["short_desc_length"] = len(short)
    row["has_detailed_desc"] = 1 if len(about) > 500 else 0

    # ── Categories ────────────────────────────────────────────────────────────
    cats = entry.get("categories", []) or []
    if cats and isinstance(cats[0], dict):
        cats = [c.get("description", "") for c in cats]
    cats_set = set(cats)

    row["is_multiplayer"]         = 1 if cats_set & MULTIPLAYER_CATS else 0
    row["has_achievements"]       = 1 if "Steam Achievements" in cats_set else 0
    row["has_cloud_save"]         = 1 if "Steam Cloud" in cats_set else 0
    row["has_controller_support"] = 1 if (
        "Full controller support" in cats_set or
        "Partial Controller Support" in cats_set
    ) else 0
    row["has_vr_support"]         = 1 if "VR Support" in cats_set else 0
    row["has_in_app_purchases"]   = 1 if "In-App Purchases" in cats_set else 0
    row["has_family_sharing"]     = 1 if "Family Sharing" in cats_set else 0
    row["category_count"]         = len(cats_set)

    # ── Tags ──────────────────────────────────────────────────────────────────
    tags = entry.get("tags", {}) or {}
    if isinstance(tags, list):
        tags = {t: 1 for t in tags}

    row["tag_count"] = len(tags)

    # Top tag vote signals (volume proxy for how well-curated the store page is)
    try:
        tag_votes = [float(v) for v in tags.values()]
        top5 = sorted(tag_votes, reverse=True)[:5]
        row["top_tag_votes_total"] = sum(top5)
        row["top_tag_votes_mean"]  = float(np.mean(top5)) if top5 else 0.0
    except Exception:
        row["top_tag_votes_total"] = 0
        row["top_tag_votes_mean"]  = 0.0

    # Top-N tag binary features (one-hot of most common tags in dataset)
    for tag_name, col_name in TAG_COLUMNS.items():
        row[col_name] = 1 if tag_name in tags else 0

    # ── Pricing ───────────────────────────────────────────────────────────────
    json_price = entry.get("price", None)
    if json_price is not None:
        try:
            row["json_price_raw"] = float(json_price)
        except Exception:
            row["json_price_raw"] = None
    else:
        row["json_price_raw"] = None

    # ── Packages ──────────────────────────────────────────────────────────────
    packages = entry.get("packages", []) or []
    row["package_count"] = len(packages)
    total_subs = sum(len(p.get("subs", [])) for p in packages if isinstance(p, dict))
    row["sku_count"] = total_subs

    # ── Achievements count ────────────────────────────────────────────────────
    row["achievement_count"] = int(entry.get("achievements", 0) or 0)

    return row


# Process all entries
records = []
errors  = 0

for appid_str, entry in raw.items():
    try:
        records.append(extract_prelaunch(appid_str, entry))
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  [WARN] appid {appid_str} failed: {e}")

df_json = pd.DataFrame(records)
df_json["appid"] = df_json["appid"].astype("int64")

print(f"\n  Extracted features from {len(df_json):,} JSON entries")
print(f"  Extraction errors: {errors}")
print(f"  game_age_days — non-null: {df_json['game_age_days'].notna().sum():,}  "
      f"null (no date): {df_json['game_age_days'].isna().sum():,}")

# STEP 5 — Merge with base CSV
print("\n  STEP 5: Merging with base CSV")

df_base["appid"] = df_base["appid"].astype("int64")
df_merged = df_base.merge(df_json, on="appid", how="left")

new_cols = [c for c in df_json.columns if c != "appid"]
matched   = df_merged["game_age_days"].notna().sum()
unmatched = df_merged["game_age_days"].isna().sum()

print(f"  Base CSV rows:        {len(df_base):,}")
print(f"  JSON entries:         {len(df_json):,}")
print(f"  Matched (appid join): {matched:,}")
print(f"  Unmatched (no JSON):  {unmatched:,}")

# Fill missing numerics with median/0
if unmatched > 0:
    print(f"\n  [INFO] Filling {unmatched} unmatched rows with median/0...")
    for col in new_cols:
        if df_merged[col].dtype in [float, "float64"] or "int" in str(df_merged[col].dtype):
            fill_val = df_merged[col].median() if df_merged[col].notna().sum() > 0 else 0
            df_merged[col] = df_merged[col].fillna(fill_val)

# Fill game_age_days nulls with dataset median (robust fallback)
if df_merged["game_age_days"].isna().sum() > 0:
    median_age = df_merged["game_age_days"].median()
    df_merged["game_age_days"] = df_merged["game_age_days"].fillna(median_age)
    print(f"  Filled null game_age_days with dataset median: {median_age:.0f} days")



# --------

df_merged["owners_per_month"] = df_merged["owners"] / df_merged["_age_months"].clip(lower=1)
df_merged.drop(columns=["_age_months"], inplace=True)

# STEP 6 — Derived / engineered features
print("\n  STEP 6: Engineering derived pre-launch features")

# Store page quality score
# Removed: has_trailer (all zeros), rebalanced remaining weights to sum to 1.0
df_merged["store_page_score"] = (
    df_merged["screenshot_count"].clip(0, 10) / 10 * 0.40 +   # visual richness
    df_merged["has_detailed_desc"]                * 0.25 +     # dev effort on description
    df_merged["weighted_language_score"]          * 0.20 +     # localization quality
    df_merged["has_website"]                      * 0.10 +     # external presence
    df_merged["has_support_email"]                * 0.05       # support signal
)

# Platform reach
df_merged["platform_reach"] = df_merged["platform_count"] / 3.0

# Maturity flag
df_merged["is_mature_content"] = (df_merged["required_age"] >= 17).astype(int)

# Marketing score (removed has_trailer, rebalanced)
df_merged["marketing_score"] = (
    df_merged["has_website"]                              * 0.35 +
    df_merged["screenshot_count"].clip(0, 10) / 10 * 0.45 +
    df_merged["has_support_email"]                        * 0.20
)

# Steam ecosystem integration
# Removed: has_trading_cards (post-success), has_workshop (post-traction)
# Rebalanced remaining weights to sum to 1.0
df_merged["steam_integration"] = (
    df_merged["has_achievements"]       * 0.35 +   # most common pre-launch feature
    df_merged["has_cloud_save"]         * 0.25 +
    df_merged["has_controller_support"] * 0.25 +
    df_merged["has_family_sharing"]     * 0.15
)

# Localization score (now uses weighted_language_score as primary signal)
df_merged["localization_score"] = (
    df_merged["weighted_language_score"]                             * 0.75 +
    df_merged["full_audio_languages_count"].clip(0, 10) / 10 * 0.25
)

derived_cols = [
    "store_page_score", "platform_reach", "is_mature_content",
    "marketing_score", "localization_score", "steam_integration",
]

print(f"  Derived features added: {len(derived_cols)}")
for col in derived_cols:
    print(f"    {col:<30}: mean={df_merged[col].mean():.3f}  std={df_merged[col].std():.3f}")

# STEP 7 — Drop columns 
print("\n  STEP 7: Removing deprecated features")

REMOVED_FEATURES = [
    "has_trailer",        # all zeros — no data in games.json
    "trailer_count",      # all zeros — no data in games.json
    "has_trading_cards",  # post-success indicator
    "has_workshop",       # post-traction indicator
    "dlc_count",          # post-success indicator
    "is_solo_dev",        # describes budget not quality
    "has_publisher",      # describes budget not quality
    "publisher_count",    # describes budget not quality
    "developer_count",    # describes budget not quality
    "has_multiplayer_tag",# redundant with top-tag binary features
    "Indie",              # not a genre; captured by tag encoding
    "json_price_raw",     # redundant with price column from base CSV
    "has_support_url",    # redundant with has_support_email
    "publisher_backing",  # derived from has_publisher (now removed)
]

actually_removed = []
for col in REMOVED_FEATURES:
    if col in df_merged.columns:
        df_merged.drop(columns=[col], inplace=True)
        actually_removed.append(col)
    else:
        print(f"  [INFO] '{col}' not found in merged df — skipping")

print(f"  Removed {len(actually_removed)} deprecated columns:")
for col in actually_removed:
    print(f"    ✗ {col}")

# STEP 8 — Final report & save
print("\n  STEP 8: Saving enriched dataset")

print(f"\n  Final shape: {df_merged.shape[0]:,} rows × {df_merged.shape[1]} columns")
df_merged.to_csv(args.out, index=False)
print(f"\n  ✅ Saved to: {args.out}")

# STEP 9 — Pre-launch feature list for modeling script
print("\n  STEP 9: Pre-launch feature list for train_prelaunch_model.py")

# Build the definitive feature list for the modeling script
tag_binary_cols = sorted(TAG_COLUMNS.values())

PRELAUNCH_FEATURES = [
    # Pricing
    "price", "initialprice", "is_free",
    # Timing
    "release_month", "owners_per_month",
    # Genre flags (coarse — enriched by tag binary features below)
    "Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports", "Racing",
    # Store page
    "screenshot_count", "has_detailed_desc", "about_length",
    "has_website", "has_support_email",
    # Platform
    "platform_windows", "platform_mac", "platform_linux", "platform_count",
    # Languages
    "supported_languages_count", "full_audio_languages_count",
    "weighted_language_score",
    # Age restriction
    "required_age", "is_mature_content",
    # Steam features
    "has_achievements", "achievement_count",
    "has_cloud_save", "has_controller_support",
    "has_vr_support", "has_in_app_purchases", "has_family_sharing",
    "category_count",
    # Tags
    "tag_count",
    # Packaging
    "package_count", "sku_count",
    # Multiplayer
    "is_multiplayer",
    # Derived composite scores
    "store_page_score", "platform_reach",
    "marketing_score", "localization_score", "steam_integration",
]

# Add all top-tag binary columns
PRELAUNCH_FEATURES += tag_binary_cols

# Filter to what's actually in the output
available = [f for f in PRELAUNCH_FEATURES if f in df_merged.columns]
missing   = [f for f in PRELAUNCH_FEATURES if f not in df_merged.columns]

print(f"\n  Available pre-launch features: {len(available)}")
print(f"  Top-tag binary features:       {len(tag_binary_cols)}")

if missing:
    print(f"\n  ⚠️  Not found in output: {missing}")

print(f"""
  Copy this into train_prelaunch_model.py → NUMERIC_FEATURES / TAG_FEATURES:

  TAG_FEATURES = {tag_binary_cols}

  (Total feature count with top-{TOP_N_TAGS} tags: {len(available)})
""")

# STEP 10 — Dataset stats summary
print("  STEP 10: Dataset summary")

print(f"\n  owners_per_month stats:")
print(f"    mean:   {df_merged['owners_per_month'].mean():.1f}")
print(f"    median: {df_merged['owners_per_month'].median():.1f}")
print(f"    min:    {df_merged['owners_per_month'].min():.1f}")
print(f"    max:    {df_merged['owners_per_month'].max():.1f}")

print(f"\n  game_age_days stats:")
print(f"    mean:   {df_merged['game_age_days'].mean():.0f} days")
print(f"    median: {df_merged['game_age_days'].median():.0f} days")
print(f"    min:    {df_merged['game_age_days'].min():.0f} days")
print(f"    max:    {df_merged['game_age_days'].max():.0f} days")

print(f"\n  weighted_language_score stats:")
print(f"    mean:   {df_merged['weighted_language_score'].mean():.3f}")
print(f"    median: {df_merged['weighted_language_score'].median():.3f}")

print(f"\n  Tag binary feature coverage (% of games with tag):")
tag_cols_in_df = [c for c in tag_binary_cols if c in df_merged.columns]
tag_coverage = df_merged[tag_cols_in_df].mean().sort_values(ascending=False)
for col, pct in tag_coverage.head(15).items():
    bar = "█" * max(1, int(pct * 30))
    print(f"    {col:<40}: {pct:.1%}  {bar}")

print(f"\n  Owner bucket distribution:")
owner_buckets = df_merged["owners"].value_counts().sort_index()
for val, count in owner_buckets.items():
    pct = count / len(df_merged) * 100
    print(f"    {val:>12,.0f} owners → {count:>5,} games ({pct:.1f}%)")

print("\n  ENRICHMENT COMPLETE")
