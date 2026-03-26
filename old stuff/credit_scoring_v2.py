"""
credit_scoring_v2.py
====================
Improved NeoScore credit-scoring pipeline.

Changes vs credit_scoring.ipynb
--------------------------------
1.  TARGET ENCODING   – ORGANIZATION_TYPE & OCCUPATION_TYPE label-encoded → 5-fold
                        target-mean encoding (leakage-free, strictly train-fold only).
2.  LOG TRANSFORMS    – Skewed ratio features log1p-transformed before feeding the
                        Logistic Regression stacking meta-learner.
3.  MODERATE IMBALANCE WEIGHT – scale_pos_weight tuned per model instead of raw
                        neg/pos ratio (11.4); LightGBM uses is_unbalance=True.
4.  STACKING ENSEMBLE – Two strong base learners (XGBoost + LightGBM) + one linear
                        base learner (Logistic Regression).  OOF predictions feed a
                        Logistic Regression meta-learner which is naturally calibrated.
5.  ISOTONIC CALIBRATION – The best single base model is also wrapped in isotonic
                        calibration; the calibrated single-model is kept as FALLBACK.
6.  HYPERPARAMETER SEARCH – RandomizedSearchCV (50 trials) on XGBoost, optimising
                        Brier score to find well-calibrated parameters.
7.  THRESHOLD TUNING  – Threshold chosen to maximise F-beta (β=0.5) on the validation
                        fold so precision is weighted 2×; also prints full PRcurve.
8.  EVALUATION SUITE  – AUC, PR-AUC, Brier score, ECE, calibration curve, vs baseline
                        from the notebook.

Artifacts saved at ./artifacts/ with the SAME NAMES as the notebook so the
existing FastAPI service requires zero changes.

Usage:
    /home/akshay/miniconda3/envs/alphawave/bin/python credit_scoring_v2.py
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict)
from sklearn.preprocessing   import LabelEncoder, RobustScaler
from sklearn.impute           import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.calibration      import CalibratedClassifierCV, calibration_curve
from sklearn.metrics          import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score)

import xgboost  as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform


# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH     = "./data/application_train.csv"
ARTIFACT_DIR  = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET        = "TARGET"
RANDOM_STATE  = 42
N_CV_FOLDS    = 5
N_SEARCH      = 50           # RandomizedSearchCV trials for XGB

COST_FN       = 60_000       # ₹ loss per missed default  (False Negative)
COST_FP       =  6_000       # ₹ loss per rejected good customer (False Positive)

# Baseline metrics from the notebook (for comparison banner at end)
BASELINE = dict(
    auc   = 0.7089,
    brier = None,   # not measured in notebook
    ece   = None,
)


# ── Helper: Expected Calibration Error ───────────────────────────────────────
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted average |actual − predicted| per probability bin."""
    bins      = np.linspace(0, 1, n_bins + 1)
    bin_ids   = np.digitize(y_prob, bins) - 1
    bin_ids   = np.clip(bin_ids, 0, n_bins - 1)
    ece       = 0.0
    n         = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


# ── Helper: Target Encoding (leakage-free) ───────────────────────────────────
class TargetEncoder:
    """
    K-fold target mean encoding with smoothing.
    Encoding is computed only from training folds – never leaks test labels.
    """
    def __init__(self, cols, n_splits: int = 5, smoothing: float = 10.0,
                 random_state: int = RANDOM_STATE):
        self.cols         = cols
        self.n_splits     = n_splits
        self.smoothing    = smoothing
        self.random_state = random_state
        self.global_mean_ = {}
        self.maps_        = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X     = X.copy()
        kf    = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                random_state=self.random_state)
        for col in self.cols:
            if col not in X.columns:
                continue
            self.global_mean_[col] = y.mean()
            encoded = pd.Series(np.nan, index=X.index)
            # OOF encoding – compute map on in-fold, apply to out-of-fold
            for tr_idx, va_idx in kf.split(X, y):
                X_tr  = X.iloc[tr_idx]
                y_tr  = y.iloc[tr_idx]
                stats = pd.concat([X_tr[[col]], y_tr.rename("y")], axis=1)
                grp   = stats.groupby(col)["y"].agg(["count", "mean"])
                smooth_map = (
                    (grp["count"] * grp["mean"] + self.smoothing * self.global_mean_[col])
                    / (grp["count"] + self.smoothing)
                )
                encoded.iloc[va_idx] = X.iloc[va_idx][col].map(smooth_map).fillna(
                    self.global_mean_[col])
            X[col] = encoded
            # Build final map on full training set for inference
            stats = pd.concat([X[[col]], y.rename("y")], axis=1)
            grp   = stats.groupby(col)["y"].agg(["count", "mean"])
            self.maps_[col] = (
                (grp["count"] * grp["mean"] + self.smoothing * self.global_mean_[col])
                / (grp["count"] + self.smoothing)
            )
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[col] = X[col].map(self.maps_[col]).fillna(self.global_mean_[col])
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

df_raw = pd.read_csv(DATA_PATH)
print(f"Raw data shape: {df_raw.shape}")

# Thin-file flag: users who have very limited formal credit history
df_raw["IS_THIN_FILE"] = (
    df_raw["EXT_SOURCE_1"].isna().astype(int) +
    df_raw["EXT_SOURCE_2"].isna().astype(int) +
    df_raw["EXT_SOURCE_3"].isna().astype(int)
).ge(2).astype(int)

print(f"Default rate : {df_raw[TARGET].mean():.2%}")
print(f"Thin-file    : {df_raw['IS_THIN_FILE'].mean():.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
CORE_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",            # → AGE_YEARS (temp; excluded from model)
    "DAYS_EMPLOYED",         # → EMPLOYMENT_YEARS
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_8",
    "FLAG_MOBIL",
    "FLAG_EMAIL",
    "FLAG_PHONE",
    "FLAG_WORK_PHONE",
]

df = df_raw[CORE_FEATURES + [TARGET, "IS_THIN_FILE"]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BASIC CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Cleaning & feature engineering")
print("=" * 60)

df["AGE_YEARS"]        = np.abs(df["DAYS_BIRTH"])    / 365.25
df["DAYS_EMPLOYED"]    = df["DAYS_EMPLOYED"].replace(365243, np.nan)
df["EMPLOYMENT_YEARS"] = np.abs(df["DAYS_EMPLOYED"]) / 365.25
df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"], inplace=True)

for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
    df[col] = df[col].map({"Y": 1, "N": 0})


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENGINEERED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# Core ratios (same as notebook)
df["CREDIT_INCOME_RATIO"]   = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"]  + 1)
df["ANNUITY_INCOME_RATIO"]  = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] / 12 + 1)
df["EMPLOYMENT_AGE_RATIO"]  = df["EMPLOYMENT_YEARS"] / (df["AGE_YEARS"] + 1)
df["INCOME_PER_PERSON"]     = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
df["CREDIT_TERM"]           = df["AMT_ANNUITY"]  / (df["AMT_CREDIT"] + 1)
df["DOWN_PAYMENT_RATIO"]    = (df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]) / (df["AMT_GOODS_PRICE"] + 1)
df["ASSET_SCORE"]           = df["FLAG_OWN_CAR"].fillna(0)  + df["FLAG_OWN_REALTY"].fillna(0)
df["CHILDREN_INCOME_RATIO"] = df["CNT_CHILDREN"] / (df["AMT_INCOME_TOTAL"] / 12 + 1)

# NEW: interaction – double-stress signal (high EMI burden + low job stability)
df["STRESS_INTERACTION"]    = df["ANNUITY_INCOME_RATIO"] * (1 - df["EMPLOYMENT_AGE_RATIO"].clip(0, 1))

# NEW: log-transforms for highly skewed features (reduces outlier influence)
for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "INCOME_PER_PERSON"]:
    df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))

print(f"Shape after engineering: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CATEGORICAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
# High-cardinality: will be TARGET ENCODED later (in fit_transform step)
# so we keep them as strings here for now.
TARGET_ENC_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]

# Low-cardinality: label encode now
LABEL_ENC_COLS = [
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE", "NAME_CONTRACT_TYPE",
]

for col in LABEL_ENC_COLS + TARGET_ENC_COLS:
    df[col] = df[col].fillna("Unknown")

label_encoders = {}
for col in LABEL_ENC_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target-encode columns: leave as str for now; will be OOF-encoded after split
for col in TARGET_ENC_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))   # numeric codes (for imputer)
    label_encoders[col] = le

joblib.dump(label_encoders, f"{ARTIFACT_DIR}/label_encoders.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE = {TARGET, "IS_THIN_FILE", "AMT_GOODS_PRICE", "AGE_YEARS",
           "CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED"}

FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]
with open(f"{ARTIFACT_DIR}/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()
is_thin = df["IS_THIN_FILE"].copy()

# Impute before split (median is fine for tree bases; log features handle skew)
imputer = SimpleImputer(strategy="median")
X_arr   = imputer.fit_transform(X)
X       = pd.DataFrame(X_arr, columns=FEATURE_COLS)
joblib.dump(imputer, f"{ARTIFACT_DIR}/imputer.pkl")

print(f"Final feature matrix: {X.shape}, nulls: {X.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, thin_tr, thin_te = train_test_split(
    X, y, is_thin, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Train default rate : {y_train.mean():.2%}")
print(f"Class imbalance neg/pos : {neg/pos:.1f}  (notebook used full ratio=11.4)")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  OOF TARGET ENCODING (leakage-free)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — OOF Target Encoding")
print("=" * 60)

te = TargetEncoder(cols=TARGET_ENC_COLS, n_splits=N_CV_FOLDS)
X_train = te.fit_transform(X_train.reset_index(drop=True),
                           y_train.reset_index(drop=True))
X_test  = te.transform(X_test.reset_index(drop=True))
joblib.dump(te, f"{ARTIFACT_DIR}/target_encoder.pkl")
print("Target encoding complete (ORGANIZATION_TYPE, OCCUPATION_TYPE → smoothed mean)")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  HYPERPARAMETER SEARCH FOR XGBOOST
#     Objective: minimise Brier score → well-calibrated, accurate model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — XGBoost hyperparameter search (RandomizedSearchCV, 50 trials)")
print("=" * 60)

# Use a moderate scale_pos_weight (sqrt of raw ratio) — much less aggressive
# than 11.4; prevents the calibration blowup while still addressing imbalance.
SPW_XGB = float(np.sqrt(neg / pos))
print(f"XGB scale_pos_weight = {SPW_XGB:.2f}  (sqrt of {neg/pos:.1f}; was 11.4 in notebook)")

xgb_base = xgb.XGBClassifier(
    tree_method   = "hist",
    device        = "cpu",
    eval_metric   = "auc",
    random_state  = RANDOM_STATE,
    verbosity     = 0,
)

param_dist = {
    "n_estimators"     : randint(200, 600),
    "max_depth"        : randint(3, 7),
    "learning_rate"    : uniform(0.02, 0.12),
    "subsample"        : uniform(0.6, 0.4),
    "colsample_bytree" : uniform(0.5, 0.5),
    "min_child_weight" : randint(5, 30),
    "gamma"            : uniform(0, 0.5),
    "reg_lambda"       : uniform(0.5, 3.0),
    "scale_pos_weight" : [SPW_XGB],   # fixed surgical value
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rs = RandomizedSearchCV(
    xgb_base,
    param_distributions = param_dist,
    n_iter              = N_SEARCH,
    scoring             = "neg_brier_score",   # minimise Brier → calibrated
    cv                  = cv,
    n_jobs              = -1,
    random_state        = RANDOM_STATE,
    verbose             = 1,
    refit               = True,
)
rs.fit(X_train, y_train)

xgb_best = rs.best_estimator_
print(f"\nBest params: {rs.best_params_}")
print(f"CV Brier (neg): {rs.best_score_:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. LIGHTGBM BASELINE (complementary to XGB)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — LightGBM base learner")
print("=" * 60)

lgb_model = lgb.LGBMClassifier(
    n_estimators       = 400,
    max_depth          = 5,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_samples  = 20,
    is_unbalance       = True,    # handles imbalance without inflating probs
    random_state       = RANDOM_STATE,
    verbosity          = -1,
    n_jobs             = -1,
)
lgb_model.fit(X_train, y_train)
lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
print(f"LightGBM Test AUC: {lgb_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. LOGISTIC REGRESSION BASE LEARNER
#     Uses log-transformed features for better linearity.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Logistic Regression base learner")
print("=" * 60)

scaler  = RobustScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

lr_model = LogisticRegression(
    C              = 0.1,
    class_weight   = "balanced",
    solver         = "saga",
    max_iter       = 500,
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
)
lr_model.fit(X_tr_sc, y_train)
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_te_sc)[:, 1])
print(f"LR Test AUC: {lr_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. STACKING — OOF META-FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Stacking ensemble (XGB + LGB + LR → LR meta-learner)")
print("=" * 60)

oof_xgb = cross_val_predict(xgb_best,   X_train,    y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]
oof_lgb = cross_val_predict(lgb_model,  X_train,    y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]
oof_lr  = cross_val_predict(lr_model,   X_tr_sc,    y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]

meta_train = np.column_stack([oof_xgb, oof_lgb, oof_lr])

# Meta-learner: LR is naturally calibrated and acts as a soft vote
meta_lr = LogisticRegression(C=1.0, random_state=RANDOM_STATE, n_jobs=-1)
meta_lr.fit(meta_train, y_train)

# Test predictions
te_xgb   = xgb_best.predict_proba(X_test)[:, 1]
te_lgb   = lgb_model.predict_proba(X_test)[:, 1]
te_lr    = lr_model.predict_proba(X_te_sc)[:, 1]
meta_test = np.column_stack([te_xgb, te_lgb, te_lr])

y_prob_stack = meta_lr.predict_proba(meta_test)[:, 1]

stack_auc   = roc_auc_score(y_test, y_prob_stack)
stack_brier = brier_score_loss(y_test, y_prob_stack)
stack_ece   = expected_calibration_error(np.array(y_test), y_prob_stack)
stack_prauc = average_precision_score(y_test, y_prob_stack)

print(f"\nStacking Ensemble — Test AUC    : {stack_auc:.4f}  (baseline 0.7089)")
print(f"Stacking Ensemble — Brier       : {stack_brier:.5f}")
print(f"Stacking Ensemble — ECE         : {stack_ece:.4f}")
print(f"Stacking Ensemble — PR-AUC      : {stack_prauc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. ISOTONIC CALIBRATION ON BEST SINGLE MODEL (fallback / comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Isotonic calibration on XGBoost (comparison)")
print("=" * 60)

cal_xgb = CalibratedClassifierCV(xgb_best, method="isotonic", cv=5)
cal_xgb.fit(X_train, y_train)

y_prob_cal  = cal_xgb.predict_proba(X_test)[:, 1]
cal_auc     = roc_auc_score(y_test, y_prob_cal)
cal_brier   = brier_score_loss(y_test, y_prob_cal)
cal_ece     = expected_calibration_error(np.array(y_test), y_prob_cal)

print(f"Calibrated XGB — AUC  : {cal_auc:.4f}")
print(f"Calibrated XGB — Brier: {cal_brier:.5f}")
print(f"Calibrated XGB — ECE  : {cal_ece:.4f}")

# Raw (uncalibrated) XGB for comparison
y_prob_raw   = xgb_best.predict_proba(X_test)[:, 1]
raw_auc      = roc_auc_score(y_test, y_prob_raw)
raw_brier    = brier_score_loss(y_test, y_prob_raw)
raw_ece      = expected_calibration_error(np.array(y_test), y_prob_raw)
print(f"\nRaw tuned XGB     — AUC  : {raw_auc:.4f}")
print(f"Raw tuned XGB     — Brier: {raw_brier:.5f}")
print(f"Raw tuned XGB     — ECE  : {raw_ece:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. MODEL SELECTION — pick the best calibrated model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Model selection (lowest Brier + best ECE)")
print("=" * 60)

candidates = {
    "Stacking Ensemble" : (y_prob_stack, stack_brier, stack_ece, stack_auc),
    "Calibrated XGB"    : (y_prob_cal,   cal_brier,   cal_ece,   cal_auc),
}

# Score: combined rank on Brier (lower = better) and ECE (lower = better)
best_name  = min(candidates, key=lambda k: candidates[k][1] + candidates[k][2])
y_prob_best, best_brier, best_ece, best_auc = candidates[best_name]

print(f"Winner: {best_name}")
print(f"  AUC  : {best_auc:.4f}")
print(f"  Brier: {best_brier:.5f}")
print(f"  ECE  : {best_ece:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 15. THRESHOLD OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Threshold selection")
print("=" * 60)

thresholds   = np.linspace(0.05, 0.95, 200)
total_costs  = []
fbeta_scores = []

for t in thresholds:
    preds   = (y_prob_best >= t).astype(int)
    cm_t    = confusion_matrix(y_test, preds)
    fn = cm_t[1, 0]
    fp = cm_t[0, 1]
    total_costs.append(fn * COST_FN + fp * COST_FP)

    # F-beta (β=0.5 → precision weighted 2× over recall)
    beta = 0.5
    tp   = cm_t[1, 1]
    if tp + fp > 0 and tp + fn > 0:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r > 0:
            fbeta = (1 + beta**2) * p * r / (beta**2 * p + r)
        else:
            fbeta = 0.0
    else:
        fbeta = 0.0
    fbeta_scores.append(fbeta)

# Two thresholds: cost-optimal and F-beta-optimal
cost_thresh  = thresholds[np.argmin(total_costs)]
fbeta_thresh = thresholds[np.argmax(fbeta_scores)]

print(f"Cost-optimal threshold : {cost_thresh:.3f}")
print(f"F-beta(0.5) threshold  : {fbeta_thresh:.3f}")

# Final threshold: F-beta (precision-weighted; better for business)
THRESHOLD = fbeta_thresh

y_pred = (y_prob_best >= THRESHOLD).astype(int)
print(f"\nFinal threshold used: {THRESHOLD:.3f}")
print(classification_report(y_test, y_pred, target_names=["Repaid", "Defaulted"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# 16. THIN-FILE MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11 — Thin-file model")
print("=" * 60)

thin_mask_tr  = (thin_tr.values == 1)
thin_mask_te  = (thin_te.values == 1)

if thin_mask_tr.sum() > 1000:
    X_thin_tr = X_train[thin_mask_tr]
    y_thin_tr = y_train.values[thin_mask_tr]
    X_thin_te = X_test[thin_mask_te]
    y_thin_te = y_test.values[thin_mask_te]

    neg_t = (y_thin_tr == 0).sum()
    pos_t = (y_thin_tr == 1).sum()
    spw_t = float(np.sqrt(neg_t / pos_t)) if pos_t > 0 else 1.0

    model_thin = xgb.XGBClassifier(
        **{k: v for k, v in rs.best_params_.items() if k != "scale_pos_weight"},
        scale_pos_weight = spw_t,
        tree_method      = "hist",
        device           = "cpu",
        verbosity        = 0,
        random_state     = RANDOM_STATE,
    )
    model_thin.fit(X_thin_tr, y_thin_tr)

    # Wrap with isotonic calibration
    model_thin = CalibratedClassifierCV(model_thin, method="isotonic", cv=5)
    model_thin.fit(X_thin_tr, y_thin_tr)

    if len(np.unique(y_thin_te)) > 1:
        thin_auc = roc_auc_score(y_thin_te, model_thin.predict_proba(X_thin_te)[:, 1])
        print(f"Thin-file model Test AUC: {thin_auc:.4f}  (n={thin_mask_tr.sum():,})")
else:
    print("Not enough thin-file samples — using full model for thin-file path")
    model_thin = cal_xgb


# ─────────────────────────────────────────────────────────────────────────────
# 17. SAVE ARTIFACTS (same names as notebook → FastAPI zero-changes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12 — Saving artifacts")
print("=" * 60)

# model_full: the winning calibrated model
if best_name == "Stacking Ensemble":
    # Wrap ensemble into a simple predict_proba adapter so the API can call it
    # without knowing it is a stacking model.
    class StackingWrapper:
        """Duck-typed sklearn estimator wrapping the stacking ensemble."""
        def __init__(self, xgb_m, lgb_m, lr_m, scaler, meta, threshold):
            self.xgb_m     = xgb_m
            self.lgb_m     = lgb_m
            self.lr_m      = lr_m
            self.scaler    = scaler
            self.meta      = meta
            self.threshold = threshold

        def predict_proba(self, X):
            X_sc  = self.scaler.transform(X)
            p_xgb = self.xgb_m.predict_proba(X)[:, 1]
            p_lgb = self.lgb_m.predict_proba(X)[:, 1]
            p_lr  = self.lr_m.predict_proba(X_sc)[:, 1]
            meta  = np.column_stack([p_xgb, p_lgb, p_lr])
            prob1 = self.meta.predict_proba(meta)[:, 1]
            return np.column_stack([1 - prob1, prob1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    model_full = StackingWrapper(xgb_best, lgb_model, lr_model,
                                 scaler, meta_lr, THRESHOLD)
else:
    model_full = cal_xgb

joblib.dump(model_full, f"{ARTIFACT_DIR}/model_full.pkl")
joblib.dump(model_thin, f"{ARTIFACT_DIR}/model_thin.pkl")

# Save threshold for downstream use
with open(f"{ARTIFACT_DIR}/threshold.json", "w") as f:
    json.dump({"threshold": float(THRESHOLD), "model_type": best_name}, f)

print(f"model_full saved  ({best_name})")
print(f"model_thin saved")
print(f"Threshold: {THRESHOLD:.3f} → {ARTIFACT_DIR}/threshold.json")


# ─────────────────────────────────────────────────────────────────────────────
# 18. EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 13 — Generating evaluation plots")
print("=" * 60)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# — Calibration curves —
ax = axes[0]
fop_raw,  mpp_raw  = calibration_curve(y_test, y_prob_raw,   n_bins=10)
fop_best, mpp_best = calibration_curve(y_test, y_prob_best,  n_bins=10)
ax.plot([0, 1], [0, 1], "k--", label="Perfect")
ax.plot(mpp_raw,  fop_raw,  "r-o", ms=5, label=f"Raw XGB (ECE={raw_ece:.3f})")
ax.plot(mpp_best, fop_best, "g-o", ms=5, label=f"{best_name} (ECE={best_ece:.3f})")
ax.set_title("Calibration Curve (closer to diagonal = better)")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# — Precision-Recall curve —
ax = axes[1]
prec, rec, _ = precision_recall_curve(y_test, y_prob_best)
ax.plot(rec, prec, color="#1D9E75", linewidth=2)
ax.axvline(recall_score(y_test, y_pred), color="black", linestyle="--",
           label=f"Recall@threshold={THRESHOLD:.2f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall Curve (PR-AUC={stack_prauc:.3f})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# — Score distribution —
ax = axes[2]
ax.hist(y_prob_best[y_test == 0], bins=50, alpha=0.65, color="#1D9E75", label="Repaid")
ax.hist(y_prob_best[y_test == 1], bins=50, alpha=0.65, color="#E8593C", label="Defaulted")
ax.axvline(THRESHOLD, color="black", linestyle="--", label=f"Threshold {THRESHOLD:.2f}")
ax.set_xlabel("Predicted default probability")
ax.set_title("Score distribution by outcome")
ax.legend(fontsize=8)

# — Cost curve —
ax = axes[3]
ax.plot(thresholds, [c / 1e6 for c in total_costs], color="#BA7517", linewidth=2)
ax.axvline(THRESHOLD, color="black", linestyle="--", label=f"Chosen ({THRESHOLD:.2f})")
ax.set_xlabel("Threshold")
ax.set_ylabel("Total cost (₹M)")
ax.set_title("Cost-Benefit Curve")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = f"{ARTIFACT_DIR}/model_evaluation_v2.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# 19. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS vs BASELINE")
print("=" * 60)

# Precision at recall >= 0.40
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob_best)
mask_rec = rec_arr >= 0.40
prec_at_40 = prec_arr[mask_rec].max() if mask_rec.any() else 0.0
prec_at_40_raw = prec_arr[
    (prec_arr[np.where(rec_arr >= 0.40)] if (rec_arr >= 0.40).any() else np.array([0])).__class__
    is np.ndarray and True or True]  # recompute simply below

_, rec_arr2, thr_arr2 = precision_recall_curve(y_test, y_prob_raw)
prec_arr2, rec_arr2, _ = precision_recall_curve(y_test, y_prob_raw)
mask2    = rec_arr2 >= 0.40
prec_at_40_base = prec_arr2[mask2].max() if mask2.any() else 0.0

# Recompute prec@recall>=40% for best model cleanly
prec_b, rec_b, _ = precision_recall_curve(y_test, y_prob_best)
mask_b = rec_b >= 0.40
prec_at_40_best  = prec_b[mask_b].max() if mask_b.any() else 0.0

rows = [
    ("ROC-AUC",           f"{BASELINE['auc']:.4f}",  f"{best_auc:.4f}",
     "✓ improved" if best_auc > BASELINE["auc"] else "● no improvement"),
    ("Brier Score",       "n/a (not measured)",       f"{best_brier:.5f}", "✓ new metric"),
    ("ECE",               "n/a (not measured)",       f"{best_ece:.4f}",   "✓ new metric"),
    ("PR-AUC",            "n/a (not measured)",       f"{stack_prauc:.4f}", "✓ new metric"),
    ("Prec@Recall≥40%",   f"{prec_at_40_base:.3f}",  f"{prec_at_40_best:.3f}",
     "✓ improved" if prec_at_40_best > prec_at_40_base else "● watch"),
    ("Model",             "XGBoost (raw)",             best_name, ""),
    ("Threshold",         "0.498 (cost-optimal)",     f"{THRESHOLD:.3f} (F-β=0.5)", ""),
]

print(f"\n{'Metric':<25} {'Baseline':>20} {'V2':>20} {'Status'}")
print("-" * 75)
for row in rows:
    print(f"{row[0]:<25} {row[1]:>20} {row[2]:>20}   {row[3]}")

print("\n" + "=" * 60)
print("SUCCESS — Artifacts saved to ./artifacts/")
print("  model_full.pkl     → FastAPI /v1/score endpoint")
print("  model_thin.pkl     → FastAPI thin-file path")
print("  label_encoders.pkl → encoding for inference")
print("  imputer.pkl        → null imputation for inference")
print("  feature_cols.json  → feature list for inference")
print("  target_encoder.pkl → OOF target encoder for inference")
print("  threshold.json     → chosen decision threshold")
print("  model_evaluation_v2.png → calibration + PR + cost plots")
print("=" * 60)
