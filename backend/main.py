"""
NeoScore Backend — Fixed Edition
Stack: Flask · MongoDB · Google OAuth · Groq AI Coach

FIXES APPLIED (see root cause analysis):
  1. Model path resolution via BASE_DIR — consistent across environments
  2. Both student_A and student_B loaded; USER_MODE routing restored
  3. prob_to_score now uses percentile method with population_prob_distribution.npy
  4. SHAP indexing fixed for TreeExplainer (interventional, single array not list)
  5. Categorical encoding fixed — raw strings encoded once, integer defaults bypass encoder
  6. USER_MODE + SIGNAL_STRENGTH computed per request; dual-model routing active
  7. FEATURE_DEFAULTS now stores raw string values for categoricals, not integer codes

FRONTEND CONTRACT: All routes, request/response shapes, and field names are
identical to the original main.py so zero frontend changes are required.

SETUP
─────
pip install flask flask-cors flask-session pymongo dnspython \
            numpy scikit-learn joblib python-dotenv requests pandas shap

.env vars:
    FLASK_SECRET_KEY=...
    FRONTEND_URL=http://localhost:3000
    MONGO_URI=mongodb://localhost:27017
    MONGO_DB_NAME=neoscore
    GOOGLE_CLIENT_ID=...
    GOOGLE_CLIENT_SECRET=...
    GOOGLE_REDIRECT_URI=http://localhost:5000/auth/callback/google
    GROQ_API_KEY=...

EXPECTED ARTIFACT LAYOUT (relative to this file):
    models/student_A.pkl
    models/student_B.pkl          ← pure-thin model (falls back to A if missing)
    models/explainer_A.pkl
    models/explainer_B.pkl        ← falls back to A if missing
    models/label_encoders.pkl
    data/population_prob_distribution.npy   ← used for percentile scoring
    data/population_score_distribution.npy  ← used for percentile display

ROUTES
──────
GET  /health
GET  /auth/login/google
GET  /auth/callback/google
GET  /auth/me
POST /auth/logout

POST /api/score
POST /api/counterfactual
POST /api/simulate
GET  /api/features

GET  /score/history
GET  /score/history/trend

POST /chat
"""

import os
import math
import secrets
import logging
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import pandas as pd
import requests as http
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, session
from flask_cors import CORS
from flask_session import Session
from pymongo import ASCENDING, DESCENDING, MongoClient
import joblib

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
SECRET_KEY       = os.getenv("FLASK_SECRET_KEY", "dev-secret")
FRONTEND_URL     = os.getenv("FRONTEND_URL", "http://localhost:3000")
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME    = os.getenv("MONGO_DB_NAME", "neoscore")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_SECRET    = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/auth/callback/google")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
IS_PROD          = os.getenv("FLASK_ENV") == "production"

SCORE_MIN, SCORE_MAX = 300, 900

# ─── FIX 1: Resolve paths relative to this file (not cwd) ────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ─── Feature list — must match student model training exactly ─────────────────
STUDENT_FEATURES = [
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT",
    "AGE_YEARS", "EMPLOYED_YEARS",
    "DEBT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_GOODS",
    "INCOME_PER_PERSON", "CHILDREN_RATIO", "EMI_BURDEN",
    "INCOME_STABILITY", "LOAN_TO_INCOME",
    "FINANCIAL_PRESSURE", "STABILITY_SCORE", "ASSET_SCORE", "INCOME_ADEQUACY",
]

# Categorical columns that go through LabelEncoder
# These are the ONLY columns that should be string-encoded at inference time.
CATEGORICAL_COLS = {
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_HOUSING_TYPE",
}

# ─── FIX 7: FEATURE_DEFAULTS use raw string values for categoricals ───────────
# Integer flags (FLAG_OWN_CAR etc.) stay as integers — they are NOT label-encoded.
# Categorical strings will be label-encoded by score_features() exactly once.
FEATURE_DEFAULTS = {
    # Counts / flags (numeric, never label-encoded)
    "CNT_CHILDREN":              0,
    "CNT_FAM_MEMBERS":           2,
    "FLAG_OWN_CAR":              0,
    "FLAG_OWN_REALTY":           1,
    "REGION_POPULATION_RELATIVE": 0.02,
    "REGION_RATING_CLIENT":      2,

    # Categoricals — raw strings matching LabelEncoder training classes
    "NAME_EDUCATION_TYPE":  "Secondary / secondary special",
    "NAME_FAMILY_STATUS":   "Married",
    "NAME_INCOME_TYPE":     "Working",
    "OCCUPATION_TYPE":      "Laborers",
    "ORGANIZATION_TYPE":    "Business Entity Type 3",
    "NAME_HOUSING_TYPE":    "House / apartment",

    # Amounts (numeric)
    "AMT_INCOME_TOTAL":  180000,
    "AMT_CREDIT":        500000,
    "AMT_ANNUITY":        25000,
    "AMT_GOODS_PRICE":   450000,
    "AGE_YEARS":              35,
    "EMPLOYED_YEARS":          5,

    # Derived ratios — computed by derive_computed_features() if not supplied
    "DEBT_TO_INCOME":       2.0,
    "ANNUITY_TO_INCOME":    0.1,
    "CREDIT_TO_GOODS":      1.1,
    "INCOME_PER_PERSON":  60000,
    "CHILDREN_RATIO":       0.0,
    "EMI_BURDEN":           0.1,
    "INCOME_STABILITY":    0.14,
    "LOAN_TO_INCOME":      0.23,
    "FINANCIAL_PRESSURE":   0.2,
    "STABILITY_SCORE":     0.25,
    "ASSET_SCORE":          0.6,
    "INCOME_ADEQUACY":      0.6,
}

# ─── Demo personas (categorical values as raw strings) ────────────────────────
# FIX 7 cont.: personas store raw strings for categoricals, not integer codes.
DEMO_PERSONAS = {
    "ravi": {
        "label": "Ravi — Gig worker, thin file",
        "features": {
            "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3,
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_INCOME_TYPE": "Working",
            "OCCUPATION_TYPE": "Laborers",
            "ORGANIZATION_TYPE": "Business Entity Type 3",
            "NAME_HOUSING_TYPE": "House / apartment",
            "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 0,
            "AMT_INCOME_TOTAL": 120000, "AMT_CREDIT": 100000,
            "AMT_ANNUITY": 7000, "AMT_GOODS_PRICE": 90000,
            "REGION_POPULATION_RELATIVE": 0.018, "REGION_RATING_CLIENT": 2,
            "AGE_YEARS": 28, "EMPLOYED_YEARS": 0.8,
            # Derived below by derive_computed_features — no need to hardcode
        }
    },
    "priya": {
        "label": "Priya — Salaried professional",
        "features": {
            "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2,
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Single / not married",
            "NAME_INCOME_TYPE": "Commercial associate",
            "OCCUPATION_TYPE": "Managers",
            "ORGANIZATION_TYPE": "Business Entity Type 2",
            "NAME_HOUSING_TYPE": "House / apartment",
            "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 1,
            "AMT_INCOME_TOTAL": 300000, "AMT_CREDIT": 250000,
            "AMT_ANNUITY": 15000, "AMT_GOODS_PRICE": 230000,
            "REGION_POPULATION_RELATIVE": 0.035, "REGION_RATING_CLIENT": 2,
            "AGE_YEARS": 32, "EMPLOYED_YEARS": 5,
        }
    },
    "deepa": {
        "label": "Deepa — Self-employed, moderate risk",
        "features": {
            "CNT_CHILDREN": 2, "CNT_FAM_MEMBERS": 4,
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_INCOME_TYPE": "Working",
            "OCCUPATION_TYPE": "Laborers",
            "ORGANIZATION_TYPE": "Self-employed",
            "NAME_HOUSING_TYPE": "House / apartment",
            "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 1,
            "AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 400000,
            "AMT_ANNUITY": 16000, "AMT_GOODS_PRICE": 380000,
            "REGION_POPULATION_RELATIVE": 0.025, "REGION_RATING_CLIENT": 2,
            "AGE_YEARS": 38, "EMPLOYED_YEARS": 2.5,
        }
    },
}

# ─── MongoDB ──────────────────────────────────────────────────────────────────
_mongo = None

def db():
    global _mongo
    if _mongo is None:
        _mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)[MONGO_DB_NAME]
        _mongo.users.create_index(
            [("provider_id", ASCENDING), ("provider", ASCENDING)], unique=True
        )
        _mongo.score_history.create_index(
            [("user_id", ASCENDING), ("created_at", DESCENDING)]
        )
        log.info("MongoDB ready")
    return _mongo

def now(): return datetime.now(timezone.utc)

def ser(doc):
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    return doc

# ─── FIX 1 + 2: Load both models + explainers from correct paths ──────────────
class MockModel:
    """Fallback when no real model is available."""
    def predict_proba(self, X):
        return np.array([[0.85, 0.15]] * len(X))

    def predict(self, X):
        # Mimics XGBRegressor returning a raw probability scalar per row
        return np.array([0.15] * len(X))

log.info("Loading ML artifacts from %s ...", MODELS_DIR)

# Student A: near-thin + thick-file users
try:
    _model_A = joblib.load(os.path.join(MODELS_DIR, "student_A.pkl"))
    log.info("Loaded student_A.pkl")
except FileNotFoundError:
    log.warning("student_A.pkl not found — using MockModel for model A")
    _model_A = MockModel()

# Student B: pure-thin users (zero credit history)
try:
    _model_B = joblib.load(os.path.join(MODELS_DIR, "student_B.pkl"))
    log.info("Loaded student_B.pkl")
except FileNotFoundError:
    log.warning("student_B.pkl not found — falling back to model A for pure-thin users")
    _model_B = _model_A

# SHAP explainers
try:
    _explainer_A = joblib.load(os.path.join(MODELS_DIR, "explainer_A.pkl"))
    log.info("Loaded explainer_A.pkl")
except FileNotFoundError:
    log.warning("explainer_A.pkl not found — SHAP explanations disabled")
    _explainer_A = None

try:
    _explainer_B = joblib.load(os.path.join(MODELS_DIR, "explainer_B.pkl"))
    log.info("Loaded explainer_B.pkl")
except FileNotFoundError:
    log.warning("explainer_B.pkl not found — falling back to explainer A")
    _explainer_B = _explainer_A

# Label encoders (fit on string class names)
try:
    _label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    log.info("Loaded label_encoders.pkl with keys: %s", list(_label_encoders.keys()))
except FileNotFoundError:
    log.warning("label_encoders.pkl not found — categoricals will be zero-encoded")
    _label_encoders = {}

# FIX 3: Load PROBABILITY distribution (not score distribution) for scoring
try:
    _pop_prob_dist = np.load(os.path.join(DATA_DIR, "population_prob_distribution.npy"))
    log.info("Loaded population_prob_distribution.npy (n=%d)", len(_pop_prob_dist))
except FileNotFoundError:
    log.warning("population_prob_distribution.npy not found — using linear score fallback")
    _pop_prob_dist = np.array([])

# Score distribution still used for percentile display
try:
    _score_dist = np.load(os.path.join(DATA_DIR, "population_score_distribution.npy"))
    log.info("Loaded population_score_distribution.npy (n=%d)", len(_score_dist))
except FileNotFoundError:
    log.warning("population_score_distribution.npy not found")
    _score_dist = np.array([])


# ─── FIX 3: Percentile-based scoring (matches notebook's prob_to_score) ───────
def prob_to_score(pd_: float) -> int:
    """
    Map default probability to 300–900 using population percentile rank.

    The notebook version:
        pct = (pop_probs > p).mean()   # fraction of population with HIGHER default prob
        score = int(300 + pct * 600)

    A higher default probability means more risk → lower score.
    When the population distribution is unavailable we fall back to the
    simple linear formula (same behaviour as original main.py).
    """
    clamped = float(np.clip(pd_, 0.001, 0.999))

    if len(_pop_prob_dist) > 0:
        # pct: fraction of the population that is RISKIER than this user
        pct   = float((_pop_prob_dist > clamped).mean())
        score = int(round(SCORE_MIN + pct * (SCORE_MAX - SCORE_MIN)))
    else:
        # Linear fallback (original behaviour)
        score = int(round(SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (1.0 - clamped)))

    return int(np.clip(score, SCORE_MIN, SCORE_MAX))


def risk_tier(score: int) -> str:
    if score >= 750: return "Excellent"
    if score >= 700: return "Good"
    if score >= 650: return "Fair"
    if score >= 600: return "Poor"
    return "Very Poor"


# ─── FIX 6: USER_MODE computation (ported from notebook) ─────────────────────
def compute_user_mode(has_bureau: bool, has_inst: bool, has_prev: bool,
                      signal_strength: float) -> str:
    """
    Replicates the notebook's assign_mode() logic.

    has_bureau / has_inst / has_prev: whether any record exists in that source.
    IS_THIN_FILE = True  ⟺  none of the three sources has data.
    """
    is_thin = not (has_bureau or has_inst or has_prev)
    if not is_thin:
        return "C_thick"
    elif signal_strength > 0:
        return "A_near_thin"
    else:
        return "B_pure_thin"


def infer_user_mode_from_features(features: dict) -> str:
    """
    At inference time the frontend sends only application-level features —
    no bureau/installment aggregates.  We therefore classify everyone as
    near-thin (A_near_thin) or pure-thin (B_pure_thin) based on employment
    signal, which is the strongest proxy available without history data.

    If EMPLOYED_YEARS > 0 we treat it as partial signal → A_near_thin → model A.
    If EMPLOYED_YEARS == 0 AND no other positive signal → B_pure_thin → model B.

    Callers that DO pass history features (bureau_count etc.) can supply
    'user_mode' directly in the features dict to override this inference.
    """
    # Allow explicit override from caller
    if "user_mode" in features:
        return str(features["user_mode"])

    emp   = float(features.get("EMPLOYED_YEARS", 0))
    own_r = float(features.get("FLAG_OWN_REALTY", 0))
    own_c = float(features.get("FLAG_OWN_CAR", 0))

    # Simple signal strength: any positive employment or asset ownership
    signal = emp > 0 or own_r > 0 or own_c > 0
    if signal:
        return "A_near_thin"   # → student_A
    else:
        return "B_pure_thin"   # → student_B


def get_model_and_explainer(user_mode: str):
    """Return (model, explainer) pair for the given user mode."""
    if user_mode == "B_pure_thin":
        return _model_B, _explainer_B
    # C_thick and A_near_thin both use model A
    return _model_A, _explainer_A


# ─── Derived feature computation ──────────────────────────────────────────────
def derive_computed_features(features: dict) -> dict:
    """
    Fill missing fields with FEATURE_DEFAULTS then auto-compute ratio /
    composite features that were not supplied by the caller.
    """
    # Start with defaults so every key exists
    f = {**FEATURE_DEFAULTS, **features}

    income   = float(f.get("AMT_INCOME_TOTAL",  180000))
    credit   = float(f.get("AMT_CREDIT",        500000))
    annuity  = float(f.get("AMT_ANNUITY",        25000))
    goods    = float(f.get("AMT_GOODS_PRICE",   450000))
    emp_yrs  = float(f.get("EMPLOYED_YEARS",         5))
    age_yrs  = float(f.get("AGE_YEARS",             35))
    children = float(f.get("CNT_CHILDREN",           0))
    fam      = float(f.get("CNT_FAM_MEMBERS",         2))
    own_car  = float(f.get("FLAG_OWN_CAR",            0))
    own_re   = float(f.get("FLAG_OWN_REALTY",         1))

    # Derived ratios — only compute if not already supplied
    if "DEBT_TO_INCOME"    not in features: f["DEBT_TO_INCOME"]    = credit  / (income  + 1)
    if "ANNUITY_TO_INCOME" not in features: f["ANNUITY_TO_INCOME"] = annuity / (income  + 1)
    if "CREDIT_TO_GOODS"   not in features: f["CREDIT_TO_GOODS"]   = credit  / (goods   + 1)
    if "INCOME_PER_PERSON" not in features: f["INCOME_PER_PERSON"] = income  / (fam     + 1)
    if "CHILDREN_RATIO"    not in features: f["CHILDREN_RATIO"]    = children / (fam    + 1)
    if "EMI_BURDEN"        not in features: f["EMI_BURDEN"]        = annuity / (income  + 1)
    if "INCOME_STABILITY"  not in features: f["INCOME_STABILITY"]  = emp_yrs / (age_yrs + 1)
    if "LOAN_TO_INCOME"    not in features: f["LOAN_TO_INCOME"]    = credit  / (income * 12 + 1)

    emi_burden  = f["EMI_BURDEN"]
    loan_income = f["LOAN_TO_INCOME"]
    child_ratio = f["CHILDREN_RATIO"]
    inc_stab    = f["INCOME_STABILITY"]

    if "FINANCIAL_PRESSURE" not in features:
        f["FINANCIAL_PRESSURE"] = (
            0.40 * min(emi_burden,  1) +
            0.35 * min(loan_income, 1) +
            0.25 * min(child_ratio, 1)
        )
    if "STABILITY_SCORE" not in features:
        f["STABILITY_SCORE"] = (
            0.50 * min(inc_stab,       1) +
            0.30 * min(emp_yrs / 40,   1) +
            0.20 * own_re
        )
    if "ASSET_SCORE" not in features:
        f["ASSET_SCORE"] = 0.60 * own_re + 0.40 * own_car

    if "INCOME_ADEQUACY" not in features:
        f["INCOME_ADEQUACY"] = min(income / (annuity * 12 + 1), 10)

    return f


# ─── Frontend → Training value translation ────────────────────────────────────
# The frontend dropdowns use simplified labels; the LabelEncoder was fit on the
# exact Home Credit dataset class names.  These maps translate before encoding.
FRONTEND_VALUE_MAP: dict[str, dict[str, str]] = {
    "NAME_EDUCATION_TYPE": {
        # Frontend label          → Home Credit class name
        "High School":            "Secondary / secondary special",
        "Secondary":              "Secondary / secondary special",
        "Bachelor":               "Higher education",
        "Bachelor's":             "Higher education",
        "Higher education":       "Higher education",
        "Master":                 "Higher education",          # no direct match; nearest
        "Master's":               "Higher education",
        "PhD":                    "Academic degree",
        "Doctorate":              "Academic degree",
        "Academic degree":        "Academic degree",
        "Incomplete higher":      "Incomplete higher",
        "Lower secondary":        "Lower secondary",
        "Secondary / secondary special": "Secondary / secondary special",
    },
    "NAME_FAMILY_STATUS": {
        "Single":                 "Single / not married",
        "Single / not married":   "Single / not married",
        "Married":                "Married",
        "Civil marriage":         "Civil marriage",
        "Separated":              "Separated",
        "Divorced":               "Separated",
        "Widow":                  "Widow",
        "Widowed":                "Widow",
    },
    "NAME_INCOME_TYPE": {
        "Working":                "Working",
        "Salaried":               "Working",
        "Commercial associate":   "Commercial associate",
        "Self-employed":          "Working",
        "Pensioner":              "Pensioner",
        "State servant":          "State servant",
        "Student":                "Student",
        "Unemployed":             "Unemployed",
        "Businessman":            "Businessman",
        "Maternity leave":        "Maternity leave",
    },
    "NAME_HOUSING_TYPE": {
        "House / apartment":      "House / apartment",
        "Own":                    "House / apartment",
        "Rented apartment":       "Rented apartment",
        "Rented":                 "Rented apartment",
        "With parents":           "With parents",
        "Municipal apartment":    "Municipal apartment",
        "Office apartment":       "Office apartment",
        "Co-op apartment":        "Co-op apartment",
    },
}

def translate_frontend_values(features: dict) -> dict:
    """
    Translate simplified frontend dropdown values to the exact class names
    the LabelEncoder was trained on.  Unknown values pass through unchanged
    so the 'Unknown class' fallback in encode_categoricals() still fires.
    """
    f = dict(features)
    for col, mapping in FRONTEND_VALUE_MAP.items():
        if col in f and isinstance(f[col], str):
            translated = mapping.get(f[col])
            if translated:
                f[col] = translated
            # else: leave as-is and let encode_categoricals warn + fallback
    return f


# ─── FIX 5: Categorical encoding — raw strings in, integer codes out ──────────
def encode_categoricals(features: dict) -> dict:
    """
    Apply LabelEncoder to each categorical column exactly once.

    Input:  features dict where categorical fields are raw strings
            (e.g. "Higher education", "Married").
    Output: same dict with those fields replaced by integer codes.

    Integer-valued inputs (FLAG_OWN_CAR = 0/1, etc.) are left untouched
    because they are not in CATEGORICAL_COLS.
    """
    f = dict(features)
    for col in CATEGORICAL_COLS:
        if col not in f:
            continue
        val = f[col]
        # If already an integer (e.g. passed by a caller that pre-encoded), skip
        if isinstance(val, (int, float, np.integer, np.floating)):
            continue
        encoder = _label_encoders.get(col)
        if encoder is None:
            log.warning("No encoder for %s — defaulting to 0", col)
            f[col] = 0
            continue
        try:
            f[col] = int(encoder.transform([str(val)])[0])
        except ValueError:
            # Unknown class — use the most common class (index 0 after LabelEncoder sort)
            log.warning("Unknown value '%s' for %s — defaulting to 0", val, col)
            f[col] = 0
    return f


# ─── Core scoring function ────────────────────────────────────────────────────
def score_features(features: dict) -> dict:
    """
    Full inference pipeline:
      1. Fill defaults + compute derived ratios
      2. Determine user mode → select model + explainer
      3. Encode categoricals (strings → ints)  [FIX 5]
      4. Build ordered DataFrame
      5. Predict with correct model             [FIX 2 + 6]
      6. Convert probability → score            [FIX 3]
      7. SHAP explanations with correct indexing [FIX 4]
    """
    # 1. Defaults + derived ratios
    features = derive_computed_features(features)

    # 2. FIX 6: Determine which model to use
    user_mode = infer_user_mode_from_features(features)
    model, explainer = get_model_and_explainer(user_mode)

    # 3. FIX 5: Translate frontend labels → training class names, then encode once
    features = translate_frontend_values(features)
    features = encode_categoricals(features)

    # 4. Build DataFrame in the exact column order the model was trained on
    df_input = pd.DataFrame([features], columns=STUDENT_FEATURES).fillna(0)

    # 5. Inference
    # XGBRegressor (trained with objective="binary:logistic") outputs raw
    # probabilities directly via .predict(); classifiers use .predict_proba().
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0]
        pd_   = float(proba[1])
    else:
        # Regressor path: output is already a probability in [0, 1]
        raw = model.predict(df_input)[0]
        pd_ = float(np.clip(raw, 0.0, 1.0))

    # 6. FIX 3: Percentile-based scoring
    score = prob_to_score(pd_)

    # Percentile display (uses score distribution, separate from scoring)
    percentile = 0.0
    if len(_score_dist) > 0:
        percentile = float((_score_dist < score).mean() * 100)

    # 7. FIX 4: SHAP — TreeExplainer returns interventional values as a
    #    single ndarray (not a list) when background data was provided.
    #    Shape is (n_samples, n_features) for binary classification.
    top_features = []
    reasoning    = "Score calculated."
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(df_input)
            # TreeExplainer with background:  shap_vals.shape == (n, n_features)
            # TreeExplainer without background (list): shap_vals[1].shape == (n, n_features)
            if isinstance(shap_vals, list):
                # Older SHAP / without background: index [1] = positive class
                local_shap = np.array(shap_vals[1][0])   # shape (n_features,)
            else:
                # With background (interventional): single array, already for
                # the positive class direction — shape (n_samples, n_features)
                local_shap = np.array(shap_vals[0])      # shape (n_features,)

            impacts = [
                {"feature": feat, "impact": float(val)}
                for feat, val in zip(STUDENT_FEATURES, local_shap)
            ]
            impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
            top_features = impacts[:5]

            # Build human-readable reasoning from top positive and negative drivers
            FEATURE_DISPLAY = {
                "CNT_CHILDREN": "number of dependants",
                "CNT_FAM_MEMBERS": "family size",
                "NAME_EDUCATION_TYPE": "education level",
                "NAME_FAMILY_STATUS": "marital status",
                "NAME_INCOME_TYPE": "income type",
                "OCCUPATION_TYPE": "occupation",
                "ORGANIZATION_TYPE": "employer type",
                "NAME_HOUSING_TYPE": "housing situation",
                "FLAG_OWN_CAR": "vehicle ownership",
                "FLAG_OWN_REALTY": "property ownership",
                "AMT_INCOME_TOTAL": "annual income",
                "AMT_CREDIT": "loan amount",
                "AMT_ANNUITY": "monthly EMI",
                "AMT_GOODS_PRICE": "goods price",
                "REGION_POPULATION_RELATIVE": "region density",
                "REGION_RATING_CLIENT": "regional credit rating",
                "AGE_YEARS": "age",
                "EMPLOYED_YEARS": "years of employment",
                "DEBT_TO_INCOME": "debt-to-income ratio",
                "ANNUITY_TO_INCOME": "annuity burden",
                "CREDIT_TO_GOODS": "credit-to-goods ratio",
                "INCOME_PER_PERSON": "income per family member",
                "CHILDREN_RATIO": "dependant ratio",
                "EMI_BURDEN": "EMI burden",
                "INCOME_STABILITY": "employment stability",
                "LOAN_TO_INCOME": "loan-to-income ratio",
                "FINANCIAL_PRESSURE": "overall financial pressure",
                "STABILITY_SCORE": "stability profile",
                "ASSET_SCORE": "asset ownership",
                "INCOME_ADEQUACY": "income adequacy",
            }

            helping  = [x for x in top_features if x["impact"] < 0]
            hurting  = [x for x in top_features if x["impact"] > 0]

            help_label = FEATURE_DISPLAY.get(
                helping[0]["feature"], helping[0]["feature"].replace("_", " ").lower()
            ) if helping else None
            hurt_label = FEATURE_DISPLAY.get(
                hurting[0]["feature"], hurting[0]["feature"].replace("_", " ").lower()
            ) if hurting else None

            if help_label and hurt_label:
                reasoning = (
                    f"Your {help_label} is working in your favour, "
                    f"but your {hurt_label} is the biggest drag on your score. "
                    f"Focus on improving it to move to the next tier."
                )
            elif help_label:
                reasoning = (
                    f"Your {help_label} is a strong positive for your score. "
                    f"Keep maintaining it to hold your current tier."
                )
            elif hurt_label:
                reasoning = (
                    f"Your {hurt_label} is the main factor holding your score back. "
                    f"Addressing it could unlock a significantly higher tier."
                )
            else:
                reasoning = "Your score reflects a balanced financial profile."
        except Exception as exc:
            log.warning("SHAP generation failed: %s", exc)

    return {
        "score":                score,
        "risk_tier":            risk_tier(score),
        "default_probability":  round(pd_, 4),
        "approval_probability": round(1.0 - pd_, 4),
        "percentile":           round(percentile, 1),
        "top_features":         top_features,
        "reasoning":            reasoning,
        "user_mode":            user_mode,          # extra debug field
        "features_used":        features,           # stripped before public responses
    }


# ─── Counterfactual Engine ────────────────────────────────────────────────────
# Only features a person can realistically change.
# AGE_YEARS, CNT_CHILDREN, REGION_* are excluded — immutable or ethically wrong.
CF_ACTIONS = [
    {"feature": "EMPLOYED_YEARS",   "label": "Job tenure",            "delta_pct":  0.50, "effort": "low",    "days":  90},
    {"feature": "AMT_INCOME_TOTAL", "label": "Annual income",         "delta_pct":  0.20, "effort": "medium", "days":  90},
    {"feature": "AMT_CREDIT",       "label": "Loan amount requested", "delta_pct": -0.20, "effort": "low",    "days":   0},
    {"feature": "FLAG_OWN_REALTY",  "label": "Property ownership",    "delta_val":   1,   "effort": "high",   "days": 365},
    {"feature": "FLAG_OWN_CAR",     "label": "Vehicle ownership",     "delta_val":   1,   "effort": "medium", "days": 180},
    {"feature": "AMT_ANNUITY",      "label": "Reduce monthly EMI",    "delta_pct": -0.15, "effort": "low",    "days":   0},
]

# Features that must NEVER appear in recommendations (immutable / unethical)
IMMUTABLE_FEATURES = {
    "AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE",
    "NAME_FAMILY_STATUS",
}

def counterfactual(features: dict, locked: list = None, max_steps: int = 3) -> dict:
    locked   = set(locked or []) | IMMUTABLE_FEATURES   # always block immutables
    features = derive_computed_features(features)
    original = score_features(features)
    moves    = []

    for action in CF_ACTIONS:
        feat = action["feature"]
        if feat in locked or feat not in features:
            continue

        new_f    = dict(features)
        orig_val = float(new_f.get(feat, 0))

        if "delta_pct" in action:
            new_val = orig_val * (1 + action["delta_pct"])
        else:
            new_val = orig_val + action["delta_val"]

        new_val    = max(0, new_val)
        new_f[feat] = new_val
        new_result  = score_features(new_f)
        delta       = new_result["score"] - original["score"]

        if delta > 0:
            moves.append({
                "feature":       feat,
                "label":         action["label"],
                "original_val":  orig_val,
                "new_val":       new_val,
                "score_delta":   delta,
                "new_score":     new_result["score"],
                "effort":        action["effort"],
                "timeline_days": action["days"],
            })

    moves.sort(key=lambda x: x["score_delta"], reverse=True)
    moves = moves[:max_steps]

    return {
        "original_score":        original["score"],
        "original_risk":         original["risk_tier"],
        "original_probability":  original["default_probability"],
        "moves":                 moves,
        "best_reachable_score":  moves[0]["new_score"] if moves else original["score"],
    }


# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SECRET_KEY              = SECRET_KEY,
    SESSION_TYPE            = "filesystem",
    SESSION_FILE_DIR        = os.path.join(os.getcwd(), "session_data"),
    SESSION_COOKIE_SAMESITE = "Lax",
    SESSION_COOKIE_SECURE   = IS_PROD,
)
Session(app)
CORS(
    app,
    origins            = ["http://localhost:3000", "http://127.0.0.1:3000"],
    supports_credentials = True,
    allow_headers      = ["Content-Type", "Authorization"],
    methods            = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)


# ─── Auth helpers ─────────────────────────────────────────────────────────────
def require_auth(fn):
    @wraps(fn)
    def inner(*a, **kw):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return fn(*a, **kw)
    return inner

def uid(): return session.get("user_id")


# ─── Auth routes — Google OAuth ───────────────────────────────────────────────
@app.route("/auth/login/google")
def google_login():
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + "&".join(
        f"{k}={v}" for k, v in params.items()
    )
    return redirect(url)


@app.route("/auth/callback/google")
def google_callback():
    if request.args.get("state") != session.pop("oauth_state", None):
        return jsonify({"error": "Invalid state"}), 400

    code      = request.args.get("code")
    token_res = http.post("https://oauth2.googleapis.com/token", data={
        "code": code, "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_SECRET, "redirect_uri": GOOGLE_REDIRECT,
        "grant_type": "authorization_code",
    }).json()

    access_token = token_res.get("access_token")
    if not access_token:
        return jsonify({"error": "Token exchange failed"}), 400

    info = http.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    ).json()

    user = db().users.find_one_and_update(
        {"provider_id": info["sub"], "provider": "google"},
        {"$set": {
            "name": info.get("name"), "email": info.get("email"),
            "picture": info.get("picture"), "updated_at": now(),
        }, "$setOnInsert": {"created_at": now()}},
        upsert=True, return_document=True,
    )
    session["user_id"]   = str(user["_id"])
    session["user_name"] = info.get("name")
    # Redirect to /home (matches frontend router — was /dashboard in old main.py)
    return redirect(f"{FRONTEND_URL}/home")


@app.route("/auth/me")
def auth_me():
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 200
    user = db().users.find_one({"_id": ObjectId(uid())})
    if not user:
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": {
        "id":      uid(),
        "name":    user.get("name"),
        "email":   user.get("email"),
        "picture": user.get("picture"),
    }})


@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


# ─── Public API — no auth required ───────────────────────────────────────────
@app.route("/api/score", methods=["POST"])
def api_score():
    """
    Public scoring endpoint.
    Body: { features: {...} }  OR  { persona: "ravi" }
    Response shape is identical to the original — frontend needs no changes.
    """
    body = request.get_json(silent=True) or {}

    if "persona" in body:
        persona = DEMO_PERSONAS.get(body["persona"])
        if not persona:
            return jsonify({"error": "Unknown persona. Use: ravi, priya, deepa"}), 400
        features = persona["features"]
    else:
        features = body.get("features", body)

    result = score_features(features)
    # Strip internal fields before returning to caller
    result.pop("features_used", None)
    result.pop("user_mode", None)
    return jsonify(result)


@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    """
    Returns actionable steps to improve score.
    Body: { features: {...}, locked_features: [...], max_steps: 3 }
    """
    body = request.get_json(silent=True) or {}
    if "persona" in body:
        persona  = DEMO_PERSONAS.get(body["persona"])
        features = persona["features"] if persona else {}
    else:
        features = body.get("features", {})

    locked    = body.get("locked_features", [])
    max_steps = int(body.get("max_steps", 3))
    return jsonify(counterfactual(features, locked, max_steps))


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    What-if simulation.
    Body: { features: {...}, changes: { EMPLOYED_YEARS: 5 } }
    """
    body     = request.get_json(silent=True) or {}
    features = body.get("features", {})
    changes  = body.get("changes", {})
    if not features and not changes:
        return jsonify({"error": "Provide features and changes"}), 400

    base   = score_features(features)
    merged = {**features, **changes}
    new    = score_features(merged)

    return jsonify({
        "original_score":       base["score"],
        "original_risk":        base["risk_tier"],
        "original_probability": base["default_probability"],
        "new_score":            new["score"],
        "new_risk":             new["risk_tier"],
        "new_probability":      new["default_probability"],
        "score_delta":          new["score"] - base["score"],
        "changes_applied":      changes,
    })


@app.route("/api/features", methods=["GET"])
def api_features():
    """Returns feature list, defaults, and demo persona labels."""
    return jsonify({
        "features":      STUDENT_FEATURES,
        "defaults":      FEATURE_DEFAULTS,
        "demo_personas": {k: v["label"] for k, v in DEMO_PERSONAS.items()},
    })


# ─── Authenticated score routes ───────────────────────────────────────────────
@app.route("/score/predict", methods=["POST"])
@require_auth
def score_predict():
    body = request.get_json(silent=True) or {}
    if "persona" in body:
        persona  = DEMO_PERSONAS.get(body["persona"])
        features = persona["features"] if persona else {}
    else:
        features = body.get("features", {})

    result = score_features(features)
    try:
        db().score_history.insert_one({
            "user_id":    uid(),
            "score":      result["score"],
            "risk_tier":  result["risk_tier"],
            "pd":         result["default_probability"],
            "user_mode":  result.get("user_mode"),
            "features":   result["features_used"],
            "created_at": now(),
        })
    except Exception as exc:
        log.warning("History insert failed: %s", exc)

    result.pop("features_used", None)
    result.pop("user_mode", None)
    return jsonify(result)


@app.route("/score/counterfactual", methods=["POST"])
@require_auth
def score_counterfactual():
    body     = request.get_json(silent=True) or {}
    features = body.get("features", {})
    locked   = body.get("locked_features", [])
    result   = counterfactual(features, locked, int(body.get("max_steps", 3)))
    return jsonify(result)


@app.route("/score/history", methods=["GET"])
@require_auth
def score_history():
    limit  = min(int(request.args.get("limit", 20)), 100)
    events = list(db().score_history.find(
        {"user_id": uid()},
        {"features": 0},
        sort=[("created_at", DESCENDING)],
        limit=limit,
    ))
    return jsonify({"events": [ser(e) for e in events]})


@app.route("/score/history/trend", methods=["GET"])
@require_auth
def score_trend():
    limit  = min(int(request.args.get("limit", 30)), 100)
    events = list(db().score_history.find(
        {"user_id": uid()},
        {"score": 1, "risk_tier": 1, "created_at": 1},
        sort=[("created_at", ASCENDING)],
        limit=limit,
    ))
    return jsonify({"trend": [
        {
            "score":     e["score"],
            "risk_tier": e["risk_tier"],
            "date":      e["created_at"].isoformat(),
        }
        for e in events
    ]})


# ─── AI Chat (Groq) ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NeoScore's financial AI coach.
You help users understand their credit score, explain what the features mean,
and give practical, India-specific advice on improving creditworthiness.
Be concise, empathetic, and actionable. Avoid jargon.
When mentioning scores, use the 300-900 scale.
"""

@app.route("/chat", methods=["POST"])
def chat():
    body    = request.get_json(silent=True) or {}
    message = body.get("message", "").strip()
    history = body.get("history", [])
    context = body.get("score_context", {})

    if not message:
        return jsonify({"error": "message required"}), 400
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 503

    system = SYSTEM_PROMPT
    if context:
        system += f"\n\nUser's current NeoScore data: {context}"

    messages = [{"role": "system", "content": system}]
    for h in history[-10:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        res = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model": "llama3-70b-8192",
                "messages":   messages,
                "max_tokens": 512,
            },
            timeout=15,
        )
        data  = res.json()
        reply = data["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except Exception as exc:
        log.error("Groq error: %s", exc)
        return jsonify({"error": "Chat service unavailable"}), 503


# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "service":     "neoscore",
        "model_a":     str(type(_model_A).__name__),
        "model_b":     str(type(_model_B).__name__),
        "pop_dist":    len(_pop_prob_dist) > 0,
        "shap_a":      _explainer_A is not None,
        "shap_b":      _explainer_B is not None,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    log.exception(e)
    return jsonify({"error": "Internal server error"}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=not IS_PROD)
