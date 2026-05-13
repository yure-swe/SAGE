#!/usr/bin/env python3
# Pre-Launch Stacked Ensemble
# Thesis: Ensemble Learning for Predicting Game Success (Pre-Launch Focus)
#
# Usage:
#   python train_prelaunch_model.py --csv steam_10k_prelaunch.csv
#   python train_prelaunch_model.py --csv steam_10k_prelaunch.csv --top-tags 50

# %% Imports & Config

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection    import train_test_split, StratifiedKFold
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost                    import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--csv",      default="steam_10k_prelaunch.csv")
parser.add_argument("--top-tags", type=int, default=50)
args = parser.parse_args()

INPUT_CSV    = args.csv
TOP_N_TAGS   = args.top_tags
OUTPUT_DIR   = "models"
MODELS_DIR   = f"{OUTPUT_DIR}/saved_models"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"blue": "#4C72B0", "green": "#55A868", "red": "#C44E52", "orange": "#DD8452"}


# %% Load Data

df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# CSV stores prices in cents (e.g. 999 = $9.99); convert to USD.
# predictor.py converts form input (USD) back to cents before building the
for _col in ("price", "initialprice"):
    if _col in df.columns:
        df[_col] = df[_col] / 100.0

# Detect tag binary columns (enrich_prelaunch.py names them tag_*)
all_tag_cols = sorted([
    c for c in df.columns
    if c.startswith("tag_")
    and c not in ("tag_count",)
    and df[c].dropna().isin([0, 1]).all()
])

# Limit to top N by prevalence (most common tags first)
if len(all_tag_cols) > TOP_N_TAGS:
    tag_prevalence = df[all_tag_cols].mean().sort_values(ascending=False)
    all_tag_cols   = tag_prevalence.head(TOP_N_TAGS).index.tolist()

print(f"Tag binary features detected: {len(all_tag_cols)}")


# %% Build Ordinal Class Target

# Owner bucket → ordinal class
# Top buckets (750K, 1.5M, 3.5M, 7.5M+) are merged into Class 5 (>=750K)
def assign_class(owners_val):
    if owners_val   <= 10_000:  return 0
    elif owners_val <= 35_000:  return 1
    elif owners_val <= 75_000:  return 2
    elif owners_val <= 150_000: return 3
    elif owners_val <= 350_000: return 4
    else:                       return 5

df["owner_class"] = df["owners"].apply(assign_class)

CLASS_LABELS = {
    0: "<=10K",
    1: "<=35K",
    2: "<=75K",
    3: "<=150K",
    4: "<=350K",
    5: ">=750K",
}
N_CLASSES = len(CLASS_LABELS)

print("\nClass distribution:")
class_counts = df["owner_class"].value_counts().sort_index()
for cls, label in CLASS_LABELS.items():
    n   = class_counts.get(cls, 0)
    pct = n / len(df) * 100
    print(f"  Class {cls} | {label:>6} | {n:>5,} games ({pct:.1f}%)")


# %% Feature Selection (Pre-Launch Only)

# Post-launch and identifier columns — always excluded to prevent data leakage
POST_LAUNCH_COLS = [
    "positive", "negative", "total_reviews", "positive_ratio",
    "average_forever", "average_2weeks", "median_forever", "median_2weeks",
    "ccu", "owners", "log_owners", "appid",
]

NUMERIC_FEATURES = [
    # Pricing
    "price", "initialprice", "is_free",
    # Release timing
    "release_month", "release_quarter", "release_dayofweek",
    "release_is_q4", "release_is_holiday", "release_is_summer", "release_is_tuesday",
    # Store page quality
    "screenshot_count", "about_length", "short_desc_length",
    "has_detailed_desc", "has_website", "has_support_email",
    # Platform support
    "platform_windows", "platform_mac", "platform_linux", "platform_count",
    # Language — quantity and market-weighted quality
    "supported_languages_count", "full_audio_languages_count", "weighted_language_score",
    # Audience
    "required_age", "is_mature_content",
    # Steam features (all configurable pre-launch)
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

# 'Indie' omitted — not a genre; captured by tag binary features instead
GENRE_FEATURES = [
    "Action", "Adventure", "RPG", "Strategy",
    "Simulation", "Sports", "Racing",
]

TAG_FEATURES = all_tag_cols

ALL_FEATURES = NUMERIC_FEATURES + GENRE_FEATURES + TAG_FEATURES

# Filter to only columns that exist in the CSV
ALL_FEATURES     = [f for f in ALL_FEATURES     if f in df.columns]
NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f in df.columns]
GENRE_FEATURES   = [f for f in GENRE_FEATURES   if f in df.columns]
TAG_FEATURES     = [f for f in TAG_FEATURES     if f in df.columns]

print(f"\nFeatures selected: {len(ALL_FEATURES)} total "
      f"({len(NUMERIC_FEATURES)} numeric, {len(GENRE_FEATURES)} genre, {len(TAG_FEATURES)} tag)")

# Warn if expected columns are missing
for col in ("weighted_language_score", "tag_count"):
    if col not in ALL_FEATURES:
        print(f"WARNING: '{col}' not found in CSV — was enrich_prelaunch.py used?")


# %% Missing Value Imputation

X = df[ALL_FEATURES].copy()
y = df["owner_class"].copy()

# Numeric: median imputation; binary genre/tag flags: zero-fill
for col in NUMERIC_FEATURES:
    if col in X.columns and X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

for col in GENRE_FEATURES + TAG_FEATURES:
    if col in X.columns:
        X[col] = X[col].fillna(0)

print(f"Missing values remaining after imputation: {X.isna().sum().sum()}")


# %% Train-Test Split (Stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")


# %% Feature Scaling (Z-Score)

# Scaler fitted on train only; applied to test to prevent leakage
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=ALL_FEATURES, index=X_train.index)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=ALL_FEATURES, index=X_test.index)


# %% Train Base Classifiers (Level 0)

print("\nTraining base classifiers...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train_sc, y_train)
rf_test_pred = rf_model.predict(X_test_sc)
rf_f1  = f1_score(y_test, rf_test_pred, average="weighted")
rf_acc = accuracy_score(y_test, rf_test_pred)
print(f"  Random Forest       — Weighted F1: {rf_f1:.4f}  Accuracy: {rf_acc:.4f}")

gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=RANDOM_STATE,
)
gb_model.fit(X_train_sc, y_train)
gb_test_pred = gb_model.predict(X_test_sc)
gb_f1  = f1_score(y_test, gb_test_pred, average="weighted")
gb_acc = accuracy_score(y_test, gb_test_pred)
print(f"  Gradient Boosting   — Weighted F1: {gb_f1:.4f}  Accuracy: {gb_acc:.4f}")

# XGBoost doesn't support class_weight directly for multiclass;
# use compute_sample_weight instead
sample_weights = compute_sample_weight("balanced", y_train)
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    num_class=N_CLASSES,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)
xgb_model.fit(X_train_sc, y_train, sample_weight=sample_weights)
xgb_test_pred = xgb_model.predict(X_test_sc)
xgb_f1  = f1_score(y_test, xgb_test_pred, average="weighted")
xgb_acc = accuracy_score(y_test, xgb_test_pred)
print(f"  XGBoost             — Weighted F1: {xgb_f1:.4f}  Accuracy: {xgb_acc:.4f}")


# %% Out-of-Fold Probability Predictions for Meta-Learner

# OOF probabilities are generated via 5-fold stratified CV so each training
# sample's probabilities come from a model that never saw that sample.
# This prevents leakage into the meta-learner.
skfold  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_rf  = np.zeros((len(X_train_sc), N_CLASSES))
oof_gb  = np.zeros((len(X_train_sc), N_CLASSES))
oof_xgb = np.zeros((len(X_train_sc), N_CLASSES))

print(f"\nGenerating {CV_FOLDS}-fold OOF predictions...")

for fold, (tr_idx, val_idx) in enumerate(skfold.split(X_train_sc, y_train)):
    X_f_tr  = X_train_sc.iloc[tr_idx]
    y_f_tr  = y_train.iloc[tr_idx]
    X_f_val = X_train_sc.iloc[val_idx]
    sw_fold = compute_sample_weight("balanced", y_f_tr)

    rf_f = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE + fold, n_jobs=-1,
    )
    gb_f = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        min_samples_split=5, min_samples_leaf=2,
        subsample=0.8, random_state=RANDOM_STATE + fold,
    )
    xgb_f = XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=5,
        min_child_weight=2, subsample=0.8, colsample_bytree=0.9,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        num_class=N_CLASSES, objective="multi:softprob",
        eval_metric="mlogloss", random_state=RANDOM_STATE + fold, verbosity=0,
    )

    rf_f.fit(X_f_tr, y_f_tr)
    gb_f.fit(X_f_tr, y_f_tr)
    xgb_f.fit(X_f_tr, y_f_tr, sample_weight=sw_fold)

    oof_rf[val_idx]  = rf_f.predict_proba(X_f_val)
    oof_gb[val_idx]  = gb_f.predict_proba(X_f_val)
    oof_xgb[val_idx] = xgb_f.predict_proba(X_f_val)

    fold_f1 = f1_score(y_train.iloc[val_idx], np.argmax(oof_rf[val_idx], axis=1),
                       average="weighted", zero_division=0)
    print(f"  Fold {fold+1}/{CV_FOLDS} — RF OOF weighted F1: {fold_f1:.4f}")

# 6 prob columns x 3 models = 18 meta-features
meta_train = np.hstack([oof_rf, oof_gb, oof_xgb])
meta_test  = np.hstack([
    rf_model.predict_proba(X_test_sc),
    gb_model.predict_proba(X_test_sc),
    xgb_model.predict_proba(X_test_sc),
])


# %% Train Meta-Learner (Level 1 — XGBoost)

# XGBoost meta-learner chosen over logistic regression because the 18
# meta-features are correlated (each model's 6 probs sum to 1.0); a
# non-linear learner handles this structure more effectively.
meta_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class=N_CLASSES,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)
meta_model.fit(meta_train, y_train)

meta_test_pred   = meta_model.predict(meta_test)
meta_f1_weighted = f1_score(y_test, meta_test_pred, average="weighted")
meta_f1_macro    = f1_score(y_test, meta_test_pred, average="macro")
meta_acc         = accuracy_score(y_test, meta_test_pred)

print(f"\nMeta-learner — Weighted F1: {meta_f1_weighted:.4f}  Macro F1: {meta_f1_macro:.4f}  Accuracy: {meta_acc:.4f}")


# %% Model Evaluation & Comparison

def eval_model(y_true, y_pred, name):
    return {
        "Model":       name,
        "Weighted F1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "Macro F1":    round(f1_score(y_true, y_pred, average="macro"),    4),
        "Accuracy":    round(accuracy_score(y_true, y_pred),               4),
    }

results = [
    eval_model(y_test, rf_test_pred,   "Random Forest"),
    eval_model(y_test, gb_test_pred,   "Gradient Boosting"),
    eval_model(y_test, xgb_test_pred,  "XGBoost"),
    eval_model(y_test, meta_test_pred, "Stacked Ensemble"),
]
df_results = pd.DataFrame(results)

print("\nTest Set Metrics:")
print(df_results.to_string(index=False))

ensemble_f1 = df_results[df_results["Model"] == "Stacked Ensemble"]["Weighted F1"].iloc[0]
best_indiv  = df_results[df_results["Model"] != "Stacked Ensemble"]["Weighted F1"].max()
improvement = (ensemble_f1 - best_indiv) / best_indiv * 100
print(f"\nEnsemble improvement over best individual: {improvement:+.2f}%")

target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
print("\nClassification Report (Stacked Ensemble):")
print(classification_report(y_test, meta_test_pred, target_names=target_names, zero_division=0))

df_results.to_csv(f"{OUTPUT_DIR}/model_evaluation_metrics.csv", index=False)


# %% Feature Importance Analysis

rf_imp  = pd.Series(rf_model.feature_importances_,  index=ALL_FEATURES).sort_values(ascending=False)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)

rf_imp.to_csv(f"{OUTPUT_DIR}/rf_feature_importance.csv",   header=["Importance"])
xgb_imp.to_csv(f"{OUTPUT_DIR}/xgb_feature_importance.csv", header=["Importance"])

print("Top 10 features (Random Forest):")
print(rf_imp.head(10).to_string())


# %% Save Models & Artifacts

joblib.dump(rf_model,   f"{MODELS_DIR}/rf_classifier.pkl")
joblib.dump(gb_model,   f"{MODELS_DIR}/gb_classifier.pkl")
joblib.dump(xgb_model,  f"{MODELS_DIR}/xgb_classifier.pkl")
joblib.dump(meta_model, f"{MODELS_DIR}/meta_classifier.pkl")
joblib.dump(scaler,     f"{MODELS_DIR}/scaler.pkl")

# feature_dict is consumed by predictor.py and app.py
feature_dict = {
    "all_features":     ALL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "genre_features":   GENRE_FEATURES,
    "tag_features":     TAG_FEATURES,
    "class_labels":     CLASS_LABELS,
    "n_classes":        N_CLASSES,
    "game_age_days_included": "game_age_days" in ALL_FEATURES,
    "output_label_note": (
        "Predicted lifetime owner tier — estimated total owners your game is "
        "likely to accumulate over its commercial lifespan on Steam."
    ),
}
joblib.dump(feature_dict, f"{MODELS_DIR}/feature_dict.pkl")

print(f"\nArtifacts saved to {MODELS_DIR}/")
print(f"  {len(ALL_FEATURES)} features | {len(TAG_FEATURES)} tag binary features")


# %% Visualizations

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Model comparison bar chart
models_list = df_results["Model"].tolist()
wf1_vals    = df_results["Weighted F1"].tolist()
bar_colors  = [COLORS["blue"]] * 3 + [COLORS["green"]]
axes[0, 0].bar(models_list, wf1_vals, color=bar_colors, edgecolor="white")
axes[0, 0].set_ylabel("Weighted F1 Score")
axes[0, 0].set_title("Model Comparison — Weighted F1 (Test Set)")
axes[0, 0].set_ylim([0, 1])
for i, v in enumerate(wf1_vals):
    axes[0, 0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
axes[0, 0].grid(axis="y", alpha=0.3)

# Confusion matrix (Stacked Ensemble, normalized)
cm      = confusion_matrix(y_test, meta_test_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names,
    ax=axes[0, 1], cbar=True, linewidths=0.5,
)
axes[0, 1].set_xlabel("Predicted Class")
axes[0, 1].set_ylabel("True Class")
axes[0, 1].set_title("Confusion Matrix — Stacked Ensemble (Normalized)")
axes[0, 1].tick_params(axis="x", rotation=45)

# RF feature importance (top 15)
top15 = rf_imp.head(15).sort_values(ascending=True)
axes[1, 0].barh(range(len(top15)), top15.values, color=COLORS["blue"])
axes[1, 0].set_yticks(range(len(top15)))
axes[1, 0].set_yticklabels(top15.index, fontsize=8)
axes[1, 0].set_xlabel("Feature Importance")
axes[1, 0].set_title("Random Forest — Top 15 Pre-Launch Features")
axes[1, 0].grid(axis="x", alpha=0.3)

# Per-class F1 (Stacked Ensemble)
per_class_f1 = f1_score(y_test, meta_test_pred, average=None, zero_division=0)
while len(per_class_f1) < N_CLASSES:
    per_class_f1 = np.append(per_class_f1, 0.0)
x_pos = np.arange(N_CLASSES)
axes[1, 1].bar(x_pos, per_class_f1[:N_CLASSES], color=COLORS["green"], edgecolor="white")
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(target_names, rotation=45, fontsize=9)
axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].set_title("Stacked Ensemble — Per-Class F1 Score")
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(axis="y", alpha=0.3)

plt.suptitle("Pre-Launch Stacked Ensemble — Training & Evaluation Results",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out_plot = f"{OUTPUT_DIR}/model_evaluation_plots.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {out_plot}")
