"""
NeoScore Backend — Merged Edition
Stack: Flask · MongoDB · Google OAuth · Groq AI Coach

Combines:
  - File 1: Frontend-compatible /api/score with strict REQUIRED_FIELDS validation,
             ThinFileCreditScorer class, thresholds.json, business rules, confidence
             scoring, counterfactual engine with fallbacks + target_score support.
  - File 2: Dual-model routing (student_A / student_B), percentile-based prob_to_score,
             SHAP indexing fix, translate_frontend_values(), encode_categoricals(),
             explain_with_llm() for ai_explanation, /chat using llama-3.3-70b-versatile.

OUTPUT CONTRACT (superset of both files — zero frontend breakage):
  /api/score returns ALL of:
    score, risk_tier, default_probability, approval_probability,
    confidence, confidence_label,
    ml_decision, final_decision, loan_eligible,
    population_percentile, population_summary,
    risk_drivers, protective_factors,
    rule_triggered, confidence_note, decision_reasoning,
    explanation  (narrative + drivers_positive + drivers_negative),
    top_features (SHAP list for SHAP chart),
    reasoning    (plain-English, overwritten by ai_explanation when available),
    ai_explanation,
    percentile   (alias of population_percentile for frontend compat)

ARTIFACT LAYOUT (relative to this file):
    student_A.pkl
    student_B.pkl
    explainer_A.pkl
    explainer_B.pkl
    label_encoders.pkl
    population_prob_distribution.npy
    population_score_distribution.npy
    config/thresholds.json

SETUP
─────
pip install flask flask-cors flask-session pymongo dnspython \\
            numpy pandas scikit-learn joblib python-dotenv requests shap

.env vars:
    FLASK_SECRET_KEY=...
    FRONTEND_URL=http://localhost:3000
    MONGO_URI=mongodb://localhost:27017
    MONGO_DB_NAME=neoscore
    GOOGLE_CLIENT_ID=...
    GOOGLE_CLIENT_SECRET=...
    GOOGLE_REDIRECT_URI=http://localhost:5000/auth/callback/google
    GROQ_API_KEY=...
"""

import os
import math
import json as _json
import pickle
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

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
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

# ─── Path resolution (relative to this file, not cwd) ────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS_PATH = os.path.join(BASE_DIR, "config", "thresholds.json")

# All pkl / npy artifacts live directly next to this file (File 1 layout).
STUDENT_A_PATH   = os.path.join(BASE_DIR, "student_A.pkl")
STUDENT_B_PATH   = os.path.join(BASE_DIR, "student_B.pkl")
EXPLAINER_A_PATH = os.path.join(BASE_DIR, "explainer_A.pkl")
EXPLAINER_B_PATH = os.path.join(BASE_DIR, "explainer_B.pkl")
ENCODERS_PATH    = os.path.join(BASE_DIR, "label_encoders.pkl")
POP_PROBS_PATH   = os.path.join(BASE_DIR, "population_prob_distribution.npy")
POP_SCORES_PATH  = os.path.join(BASE_DIR, "population_score_distribution.npy")

# ─── Feature list — must match training exactly ───────────────────────────────
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

# Categorical columns that go through LabelEncoder (strings → int codes)
CATEGORICAL_COLS = {
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_HOUSING_TYPE",
}

# Raw-string defaults — categoricals stored as strings so encode_categoricals()
# can process them. Derived ratios listed here only as last-resort fallbacks.
FEATURE_DEFAULTS = {
    "CNT_CHILDREN":              0,
    "CNT_FAM_MEMBERS":           2,
    "FLAG_OWN_CAR":              0,
    "FLAG_OWN_REALTY":           1,
    "REGION_POPULATION_RELATIVE": 0.02,
    "REGION_RATING_CLIENT":      2,
    "NAME_EDUCATION_TYPE":  "Secondary / secondary special",
    "NAME_FAMILY_STATUS":   "Married",
    "NAME_INCOME_TYPE":     "Working",
    "OCCUPATION_TYPE":      "Laborers",
    "ORGANIZATION_TYPE":    "Business Entity Type 3",
    "NAME_HOUSING_TYPE":    "House / apartment",
    "AMT_INCOME_TOTAL":  180000,
    "AMT_CREDIT":        500000,
    "AMT_ANNUITY":        25000,
    "AMT_GOODS_PRICE":   450000,
    "AGE_YEARS":              35,
    "EMPLOYED_YEARS":          5,
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

# ─── Demo personas ─────────────────────────────────────────────────────────────
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
        },
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
        },
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
        },
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

def now():
    return datetime.now(timezone.utc)

def ser(doc):
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    return doc

# ─── Artifact loading ─────────────────────────────────────────────────────────

class _MockModel:
    """Used when pkl artifacts are absent — produces a neutral mid-range output."""
    def predict(self, X):
        return np.array([0.15] * len(X))
    def predict_proba(self, X):
        return np.array([[0.85, 0.15]] * len(X))

log.info("Loading ML artifacts from %s ...", BASE_DIR)

try:
    with open(STUDENT_A_PATH, "rb") as f:
        _model_A = pickle.load(f)
    log.info("Loaded student_A.pkl")
except Exception as e:
    log.warning("student_A.pkl not found (%s) — using MockModel", e)
    _model_A = _MockModel()

try:
    with open(STUDENT_B_PATH, "rb") as f:
        _model_B = pickle.load(f)
    log.info("Loaded student_B.pkl")
except Exception as e:
    log.warning("student_B.pkl not found (%s) — falling back to model A", e)
    _model_B = _model_A

try:
    with open(EXPLAINER_A_PATH, "rb") as f:
        _explainer_A = pickle.load(f)
    log.info("Loaded explainer_A.pkl")
except Exception as e:
    log.warning("explainer_A.pkl not found (%s) — SHAP disabled", e)
    _explainer_A = None

try:
    with open(EXPLAINER_B_PATH, "rb") as f:
        _explainer_B = pickle.load(f)
    log.info("Loaded explainer_B.pkl")
except Exception as e:
    log.warning("explainer_B.pkl not found (%s) — falling back to explainer A", e)
    _explainer_B = _explainer_A

try:
    with open(ENCODERS_PATH, "rb") as f:
        _label_encoders = pickle.load(f)
    log.info("Loaded label_encoders.pkl — keys: %s", list(_label_encoders.keys()))
except Exception as e:
    log.warning("label_encoders.pkl not found (%s) — zero-encoding fallback", e)
    _label_encoders = {}

try:
    _pop_prob_dist = np.load(POP_PROBS_PATH)
    log.info("Loaded population_prob_distribution.npy (n=%d)", len(_pop_prob_dist))
except Exception as e:
    log.warning("population_prob_distribution.npy not found (%s) — linear fallback", e)
    _pop_prob_dist = np.array([])

try:
    _score_dist = np.load(POP_SCORES_PATH)
    log.info("Loaded population_score_distribution.npy (n=%d)", len(_score_dist))
except Exception as e:
    log.warning("population_score_distribution.npy not found (%s)", e)
    _score_dist = np.array([])

try:
    with open(THRESHOLDS_PATH) as f:
        _thresholds = _json.load(f)
    log.info("Loaded thresholds.json: %s", _thresholds)
except Exception as e:
    log.warning("thresholds.json not found (%s) — using defaults", e)
    _thresholds = {"approve": 0.083, "reject": 0.30}

_artifacts_ready = not isinstance(_model_A, _MockModel)

# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _prob_to_score(prob: float) -> int:
    """
    Percentile-based scoring (File 2 method):
      pct = fraction of population with HIGHER default probability than this applicant
      score = 300 + pct × 600
    Falls back to linear inversion when population distribution is unavailable.
    """
    clamped = float(np.clip(prob, 0.001, 0.999))
    if len(_pop_prob_dist) > 0:
        pct   = float((_pop_prob_dist > clamped).mean())
        score = int(round(SCORE_MIN + pct * (SCORE_MAX - SCORE_MIN)))
    else:
        # Linear fallback: lower prob → higher score
        score = int(round(SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (1.0 - clamped)))
    return int(np.clip(score, SCORE_MIN, SCORE_MAX))


def _get_tier(score: int) -> str:
    if score >= 800: return "Excellent"
    if score >= 740: return "Very Good"
    if score >= 670: return "Good"
    if score >= 580: return "Fair"
    return "Poor"


def _get_ml_decision(prob: float) -> str:
    t = _thresholds
    if prob < t.get("approve", 0.083):  return "APPROVE"
    if prob >= t.get("reject", 0.30):   return "REJECT"
    return "REVIEW"


def _get_confidence(prob: float) -> float:
    return round(abs(prob - 0.5) * 2, 3)


def _get_confidence_label(c: float) -> str:
    if c >= 0.70: return "High"
    if c >= 0.40: return "Medium"
    return "Low"


def _apply_business_rules(features: dict, ml_decision: str, prob: float):
    """Returns (final_decision, rule_triggered_or_None)."""
    emi = features.get("EMI_BURDEN", 0)
    lti = features.get("LOAN_TO_INCOME", 0)
    ia  = features.get("INCOME_ADEQUACY", 10)
    age = features.get("AGE_YEARS", 35)
    if emi > 0.60:
        return "REJECT", "EMI burden exceeds 60% of income"
    if lti > 10 and ml_decision == "APPROVE":
        return "REVIEW", "Loan amount exceeds 10× annual income"
    if ia < 0.5 and ml_decision == "APPROVE":
        return "REVIEW", "Income insufficient relative to loan obligations"
    if age < 21 and ml_decision == "APPROVE":
        return "REVIEW", "Applicant age below 21 — manual review required"
    return ml_decision, None


def _get_population_percentile(prob: float) -> float:
    if len(_pop_prob_dist) == 0:
        return 50.0
    return round(float((_pop_prob_dist >= prob).mean() * 100), 1)


def _narrative(prob: float) -> str:
    if prob < 0.083:  return "Strong credit profile — you are in the lower-risk tier of applicants."
    if prob < 0.149:  return "Moderate profile — a few improvements could move you to the approval tier."
    return "Elevated risk profile — focus on reducing financial pressure and building employment stability."


# ─── SHAP explanation templates (File 1 vocabulary) ──────────────────────────
EXPLANATION_TEMPLATES = {
    "FINANCIAL_PRESSURE":  ("High combined financial burden increased risk",
                            "Low financial pressure reduced risk"),
    "STABILITY_SCORE":     ("Low employment and asset stability increased risk",
                            "Strong employment history and asset ownership reduced risk"),
    "EMI_BURDEN":          ("EMI obligations are high relative to income",
                            "EMI obligations are manageable relative to income"),
    "LOAN_TO_INCOME":      ("Loan amount is large relative to annual income",
                            "Loan amount is proportionate to income"),
    "DEBT_TO_INCOME":      ("Total debt relative to income is elevated",
                            "Debt-to-income ratio is healthy"),
    "INCOME_STABILITY":    ("Income stability is low — short or irregular employment",
                            "Stable long-term employment reduces default risk"),
    "INCOME_ADEQUACY":     ("Income may be insufficient to service loan obligations",
                            "Income comfortably covers loan obligations"),
    "ASSET_SCORE":         ("No significant assets to act as collateral",
                            "Ownership of property or vehicle reduces risk"),
    "EMPLOYED_YEARS":      ("Short employment duration increases uncertainty",
                            "Long employment duration indicates stability"),
    "AGE_YEARS":           ("Younger applicant with less financial history",
                            "Age indicates established financial maturity"),
    "INCOME_PER_PERSON":   ("Low income per family member increases financial stress",
                            "Adequate income per family member reduces stress"),
    "CHILDREN_RATIO":      ("High dependency ratio increases financial obligations",
                            "Low dependency ratio reduces financial obligations"),
    "ANNUITY_TO_INCOME":   ("Annual loan repayment is a large share of income",
                            "Annual repayment is a small share of income"),
    "FLAG_OWN_REALTY":     ("No property ownership noted",
                            "Property ownership is a positive stability signal"),
    "FLAG_OWN_CAR":        ("No vehicle ownership noted",
                            "Vehicle ownership is a minor positive signal"),
    "REGION_RATING_CLIENT":("Region has higher credit risk profile",
                            "Region has lower credit risk profile"),
}

FEATURE_DISPLAY = {
    "CNT_CHILDREN": "number of dependants", "CNT_FAM_MEMBERS": "family size",
    "NAME_EDUCATION_TYPE": "education level", "NAME_FAMILY_STATUS": "marital status",
    "NAME_INCOME_TYPE": "income type", "OCCUPATION_TYPE": "occupation",
    "ORGANIZATION_TYPE": "employer type", "NAME_HOUSING_TYPE": "housing situation",
    "FLAG_OWN_CAR": "vehicle ownership", "FLAG_OWN_REALTY": "property ownership",
    "AMT_INCOME_TOTAL": "annual income", "AMT_CREDIT": "loan amount",
    "AMT_ANNUITY": "monthly EMI", "AMT_GOODS_PRICE": "goods price",
    "REGION_POPULATION_RELATIVE": "region density",
    "REGION_RATING_CLIENT": "regional credit rating",
    "AGE_YEARS": "age", "EMPLOYED_YEARS": "years of employment",
    "DEBT_TO_INCOME": "debt-to-income ratio", "ANNUITY_TO_INCOME": "annuity burden",
    "CREDIT_TO_GOODS": "credit-to-goods ratio",
    "INCOME_PER_PERSON": "income per family member",
    "CHILDREN_RATIO": "dependant ratio", "EMI_BURDEN": "EMI burden",
    "INCOME_STABILITY": "employment stability", "LOAN_TO_INCOME": "loan-to-income ratio",
    "FINANCIAL_PRESSURE": "overall financial pressure",
    "STABILITY_SCORE": "stability profile",
    "ASSET_SCORE": "asset ownership", "INCOME_ADEQUACY": "income adequacy",
}

# For explain_with_llm() prompt construction
FEATURE_MAP = {
    "AMT_CREDIT":            "loan amount",
    "AMT_INCOME_TOTAL":      "annual income",
    "EMPLOYED_YEARS":        "job stability",
    "DEBT_TO_INCOME":        "debt-to-income ratio",
    "ANNUITY_TO_INCOME":     "EMI-to-income burden",
    "LOAN_TO_INCOME":        "loan-to-income ratio",
    "FINANCIAL_PRESSURE":    "overall financial pressure",
    "STABILITY_SCORE":       "employment stability",
    "ASSET_SCORE":           "asset ownership",
    "INCOME_ADEQUACY":       "income adequacy",
    "FLAG_OWN_REALTY":       "property ownership",
    "FLAG_OWN_CAR":          "vehicle ownership",
    "AGE_YEARS":             "age profile",
    "CNT_CHILDREN":          "number of dependants",
    "INCOME_PER_PERSON":     "income per family member",
    "REGION_RATING_CLIENT":  "regional credit environment",
    "AMT_ANNUITY":           "monthly EMI",
    "AMT_GOODS_PRICE":       "goods price",
    "CREDIT_TO_GOODS":       "credit-to-goods ratio",
    "CHILDREN_RATIO":        "dependant ratio",
    "EMI_BURDEN":            "EMI burden",
    "INCOME_STABILITY":      "income stability",
    "NAME_EDUCATION_TYPE":   "education level",
    "NAME_FAMILY_STATUS":    "marital status",
    "NAME_INCOME_TYPE":      "income type",
    "OCCUPATION_TYPE":       "occupation",
    "ORGANIZATION_TYPE":     "employer type",
    "NAME_HOUSING_TYPE":     "housing situation",
    "CNT_FAM_MEMBERS":       "family size",
    "REGION_POPULATION_RELATIVE": "region density",
}

# ─── Frontend value translation (File 2) ─────────────────────────────────────
FRONTEND_VALUE_MAP: dict = {
    "NAME_EDUCATION_TYPE": {
        "High School":                    "Secondary / secondary special",
        "Secondary":                      "Secondary / secondary special",
        "Bachelor":                       "Higher education",
        "Bachelor's":                     "Higher education",
        "Higher education":               "Higher education",
        "Master":                         "Higher education",
        "Master's":                       "Higher education",
        "PhD":                            "Academic degree",
        "Doctorate":                      "Academic degree",
        "Academic degree":                "Academic degree",
        "Incomplete higher":              "Incomplete higher",
        "Lower secondary":                "Lower secondary",
        "Secondary / secondary special":  "Secondary / secondary special",
    },
    "NAME_FAMILY_STATUS": {
        "Single":               "Single / not married",
        "Single / not married": "Single / not married",
        "Married":              "Married",
        "Civil marriage":       "Civil marriage",
        "Separated":            "Separated",
        "Divorced":             "Separated",
        "Widow":                "Widow",
        "Widowed":              "Widow",
    },
    "NAME_INCOME_TYPE": {
        "Working":              "Working",
        "Salaried":             "Working",
        "Commercial associate": "Commercial associate",
        "Self-employed":        "Working",
        "Pensioner":            "Pensioner",
        "State servant":        "State servant",
        "Student":              "Student",
        "Unemployed":           "Unemployed",
        "Businessman":          "Businessman",
        "Maternity leave":      "Maternity leave",
    },
    "NAME_HOUSING_TYPE": {
        "House / apartment":   "House / apartment",
        "Own":                 "House / apartment",
        "Rented apartment":    "Rented apartment",
        "Rented":              "Rented apartment",
        "With parents":        "With parents",
        "Municipal apartment": "Municipal apartment",
        "Office apartment":    "Office apartment",
        "Co-op apartment":     "Co-op apartment",
    },
}

def translate_frontend_values(features: dict) -> dict:
    """Translate simplified frontend dropdown labels to training class names."""
    f = dict(features)
    for col, mapping in FRONTEND_VALUE_MAP.items():
        if col in f and isinstance(f[col], str):
            translated = mapping.get(f[col])
            if translated:
                f[col] = translated
    return f


# ─── Categorical encoding (File 2 method, robust) ────────────────────────────
def encode_categoricals(features: dict) -> dict:
    """
    Apply LabelEncoder to each categorical column exactly once.
    Integer-valued inputs are left untouched (they are not in CATEGORICAL_COLS).
    Unknown strings fall back to code 0 with a warning.
    """
    f = dict(features)
    for col in CATEGORICAL_COLS:
        if col not in f:
            continue
        val = f[col]
        if isinstance(val, (int, float, np.integer, np.floating)):
            continue   # already numeric — skip
        encoder = _label_encoders.get(col)
        if encoder is None:
            log.warning("No encoder for %s — defaulting to 0", col)
            f[col] = 0
            continue
        try:
            f[col] = int(encoder.transform([str(val)])[0])
        except ValueError:
            log.warning("Unknown value '%s' for %s — defaulting to 0", val, col)
            f[col] = 0
    return f


# ─── Derived feature computation ──────────────────────────────────────────────
def derive_computed_features(features: dict) -> dict:
    """Fill defaults then compute all ratio / composite features."""
    f = {**FEATURE_DEFAULTS, **features}

    income   = max(float(f.get("AMT_INCOME_TOTAL",  180000)), 1)
    credit   = max(float(f.get("AMT_CREDIT",        500000)), 0)
    annuity  = max(float(f.get("AMT_ANNUITY",        25000)), 0)
    goods    = max(float(f.get("AMT_GOODS_PRICE",   450000)), 1)
    emp_yrs  = max(float(f.get("EMPLOYED_YEARS",         5)), 0)
    age_yrs  = max(float(f.get("AGE_YEARS",             35)), 1)
    children = max(float(f.get("CNT_CHILDREN",           0)), 0)
    fam      = max(float(f.get("CNT_FAM_MEMBERS",         2)), 1)
    own_car  = float(f.get("FLAG_OWN_CAR",    0))
    own_re   = float(f.get("FLAG_OWN_REALTY", 1))

    if "DEBT_TO_INCOME"    not in features: f["DEBT_TO_INCOME"]    = min(credit / income, 20.0)
    if "ANNUITY_TO_INCOME" not in features: f["ANNUITY_TO_INCOME"] = min(annuity / income, 2.0)
    if "CREDIT_TO_GOODS"   not in features: f["CREDIT_TO_GOODS"]   = min(credit / goods, 5.0)
    if "INCOME_PER_PERSON" not in features: f["INCOME_PER_PERSON"] = min(income / fam, 1_000_000)
    if "CHILDREN_RATIO"    not in features: f["CHILDREN_RATIO"]    = min(children / fam, 1.0)
    if "EMI_BURDEN"        not in features: f["EMI_BURDEN"]        = min(annuity / income, 1.0)
    if "INCOME_STABILITY"  not in features: f["INCOME_STABILITY"]  = min(emp_yrs / age_yrs, 1.0)
    if "LOAN_TO_INCOME"    not in features: f["LOAN_TO_INCOME"]    = min(credit / (income * 12), 5.0)

    emi_burden  = min(f["EMI_BURDEN"], 1.0)
    loan_income = min(f["LOAN_TO_INCOME"], 1.0)
    child_ratio = min(f["CHILDREN_RATIO"], 1.0)
    inc_stab    = min(f["INCOME_STABILITY"], 1.0)

    if "FINANCIAL_PRESSURE" not in features:
        f["FINANCIAL_PRESSURE"] = min(0.40 * emi_burden + 0.35 * loan_income + 0.25 * child_ratio, 1.0)
    if "STABILITY_SCORE" not in features:
        f["STABILITY_SCORE"] = min(0.50 * inc_stab + 0.30 * min(emp_yrs / 40, 1) + 0.20 * own_re, 1.0)
    if "ASSET_SCORE"     not in features:
        f["ASSET_SCORE"]     = min(0.60 * own_re + 0.40 * own_car, 1.0)
    if "INCOME_ADEQUACY" not in features:
        f["INCOME_ADEQUACY"] = min(income / max(annuity * 12, 1), 10.0)

    return f


# ─── Dual-model routing (File 2 method) ──────────────────────────────────────
def _infer_user_mode(features: dict) -> str:
    """
    Route to student_A (near-thin) or student_B (pure-thin) based on
    employment / asset signal in the incoming features.
    Callers can supply 'user_mode' directly in features to override.
    """
    if "user_mode" in features:
        return str(features["user_mode"])
    emp   = float(features.get("EMPLOYED_YEARS", 0))
    own_r = float(features.get("FLAG_OWN_REALTY", 0))
    own_c = float(features.get("FLAG_OWN_CAR", 0))
    if emp > 0 or own_r > 0 or own_c > 0:
        return "A_near_thin"
    return "B_pure_thin"


def _get_model_and_explainer(user_mode: str):
    if user_mode == "B_pure_thin":
        return _model_B, _explainer_B
    return _model_A, _explainer_A


# ─── Heuristic fallback scorer (when models not loaded) ──────────────────────
def _heuristic_score(features: dict) -> dict:
    """Pure-math fallback used when pkl artifacts are absent."""
    fp   = min(features.get("FINANCIAL_PRESSURE", 0.2), 1.0)
    ss   = min(features.get("STABILITY_SCORE",    0.25), 1.0)
    ia   = min(features.get("INCOME_ADEQUACY",    0.6) / 10.0, 1.0)
    emi  = min(features.get("EMI_BURDEN",         0.1) / 0.5, 1.0)
    dti  = min(features.get("DEBT_TO_INCOME",     2.0) / 5.0, 1.0)
    lti  = min(features.get("LOAN_TO_INCOME",     0.23), 1.0)
    emp  = min(features.get("EMPLOYED_YEARS",     5) / 20.0, 1.0)
    own  = min(features.get("FLAG_OWN_REALTY",    0), 1.0)
    edu_raw = features.get("NAME_EDUCATION_TYPE", 1)
    edu  = (edu_raw if isinstance(edu_raw, (int, float)) else 1) / 4.0

    prob = max(0.03, min(0.95,
        0.30 * fp + 0.20 * emi + 0.15 * dti + 0.10 * lti
        - 0.10 * ss - 0.08 * ia - 0.06 * emp - 0.04 * own - 0.03 * edu + 0.08
    ))
    score = _prob_to_score(prob)
    tier  = _get_tier(score)
    conf  = _get_confidence(prob)
    ml_dec     = _get_ml_decision(prob)
    conf_note  = None
    if conf < 0.30 and ml_dec == "APPROVE":
        ml_dec    = "REVIEW"
        conf_note = "Low model confidence — flagged for manual review"
    final_dec, rule = _apply_business_rules(features, ml_dec, prob)
    pct = _get_population_percentile(prob)

    return {
        "score":                score,
        "risk_tier":            tier,
        "default_probability":  round(prob, 4),
        "approval_probability": round((score - 300) / 600, 4),
        "confidence":           conf,
        "confidence_label":     _get_confidence_label(conf),
        "ml_decision":          ml_dec,
        "final_decision":       final_dec,
        "loan_eligible":        final_dec in ("APPROVE", "REVIEW"),
        "user_mode":            "heuristic",
        "population_percentile": pct,
        "population_summary":   "Model artifacts not loaded — heuristic estimate only.",
        "percentile":           pct,
        "rule_triggered":       rule,
        "confidence_note":      conf_note,
        "risk_drivers":         [],
        "protective_factors":   [],
        "top_features":         [],
        "reasoning":            _narrative(prob),
        "ai_explanation":       None,
        "decision_reasoning":   f"Credit score of {score} ({tier}). {_narrative(prob)}",
        "explanation": {
            "narrative":        _narrative(prob),
            "drivers_positive": [],
            "drivers_negative": [],
        },
        "features_used":        features,
    }


# ─── SHAP explanation builder ─────────────────────────────────────────────────
def _build_shap_outputs(explainer, df_input: pd.DataFrame, prob: float):
    """
    Returns (top_features, risk_drivers, protective_factors, reasoning, explanation_dict).
    Handles both list-of-arrays (old SHAP) and single-array (new SHAP with background).
    """
    top_features      = []
    risk_drivers      = []
    protective_factors= []
    reasoning         = _narrative(prob)
    explanation       = {
        "narrative":        _narrative(prob),
        "drivers_positive": [],
        "drivers_negative": [],
    }

    if explainer is None:
        return top_features, risk_drivers, protective_factors, reasoning, explanation

    try:
        shap_vals = explainer.shap_values(df_input)

        # Unify indexing: File 2 fix — handles both output shapes
        if isinstance(shap_vals, list):
            # Older SHAP / without background: [class_0_array, class_1_array]
            local_shap = np.array(shap_vals[1][0])
        else:
            # With background (interventional): shape (n_samples, n_features)
            local_shap = np.array(shap_vals[0])

        impacts = [
            {"feature": feat, "impact": float(val)}
            for feat, val in zip(STUDENT_FEATURES, local_shap)
        ]
        impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_features = impacts[:5]

        # Plain-English risk/protective lists (File 1 vocabulary)
        def _text(feat: str, direction: str) -> str:
            tmpl = EXPLANATION_TEMPLATES.get(feat)
            if tmpl:
                return tmpl[0 if direction == "high" else 1]
            readable = feat.replace("_", " ").title()
            return f"{'High' if direction == 'high' else 'Low'} {readable}"

        drivers    = sorted([(f, v) for f, v in zip(STUDENT_FEATURES, local_shap) if v > 0],
                            key=lambda x: -x[1])[:3]
        protective = sorted([(f, v) for f, v in zip(STUDENT_FEATURES, local_shap) if v < 0],
                            key=lambda x:  x[1])[:3]

        risk_drivers       = [_text(f, "high") for f, _ in drivers]
        protective_factors = [_text(f, "low")  for f, _ in protective]

        explanation = {
            "narrative":        _narrative(prob),
            "drivers_positive": [{"text": t} for t in protective_factors],
            "drivers_negative": [{"text": t} for t in risk_drivers],
        }

        # Short plain-English reasoning sentence (File 2 style)
        helping = [x for x in top_features if x["impact"] < 0]
        hurting = [x for x in top_features if x["impact"] > 0]
        hl = FEATURE_DISPLAY.get(helping[0]["feature"], "") if helping else None
        hl2 = FEATURE_DISPLAY.get(hurting[0]["feature"], "") if hurting else None

        if hl and hl2:
            reasoning = (
                f"Your {hl} is working in your favour, "
                f"but your {hl2} is the biggest drag on your score. "
                f"Focus on improving it to move to the next tier."
            )
        elif hl:
            reasoning = f"Your {hl} is a strong positive. Keep maintaining it to hold your current tier."
        elif hl2:
            reasoning = f"Your {hl2} is the main factor holding your score back. Addressing it could unlock a higher tier."

    except Exception as exc:
        log.warning("SHAP generation failed: %s", exc)

    return top_features, risk_drivers, protective_factors, reasoning, explanation


# ─── LLM explanation (File 2) ─────────────────────────────────────────────────
BLOCKED_FROM_EXPLANATION = {
    "AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE", "NAME_FAMILY_STATUS",
}

def explain_with_llm(result: dict) -> str:
    """
    Call Groq with SHAP top_features to produce a plain-English ≤120-word explanation.
    Falls back to result["reasoning"] if unavailable.
    """
    if not GROQ_API_KEY:
        return result.get("reasoning", "Score calculated.")

    filtered = [
        f for f in result.get("top_features", [])
        if f["feature"] not in BLOCKED_FROM_EXPLANATION
    ]

    top_features_text = "\n".join([
        "- {}: {}".format(
            FEATURE_MAP.get(f["feature"], f["feature"].replace("_", " ").lower()),
            "hurting your score" if f["impact"] > 0 else "helping your score",
        )
        for f in filtered
    ]) or "No actionable factors available."

    prompt = (
        "Here's what's going on with your score:\n\n"
        "Credit Score: {} out of 900\n"
        "Risk Category: {}\n\n"
        "Key factors (ONLY use these — do not assume or invent anything else):\n"
        "{}\n\n"
        "Write a short, plain-English explanation (under 120 words). Rules:\n"
        "- Start with: 'Here\u2019s what\u2019s going on with your score:'\n"
        "- ONLY refer to the factors listed above — nothing else\n"
        "- DO NOT mention age, number of children, region, or any factor not in the list\n"
        "- DO NOT hallucinate factors like credit utilisation, credit card debt, or payment history\n"
        "- Mention 1-2 things working in the person\u2019s favour\n"
        "- Mention 1-2 things dragging the score down\n"
        "- Give 2 DIFFERENT actionable suggestions based on DIFFERENT factors\n"
        "- Each suggestion must be based on a different factor from the list\n"
        "- DO NOT repeat the same idea (e.g. job stability twice)\n"
        "- Prefer variety across income, loan amount, EMI, assets, stability\n"
        "- Use 'your' to address the person directly\n"
        "- No technical terms like SHAP, probability, or model"
    ).format(result["score"], result["risk_tier"], top_features_text)

    try:
        res = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a helpful financial advisor. Be concise and empathetic."},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 200,
            },
            timeout=10,
        )
        data = res.json()
        if "choices" not in data:
            log.warning("explain_with_llm: unexpected Groq response: %s", data)
            return result.get("reasoning", "Score calculated.")
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.warning("explain_with_llm failed: %s", exc)
        return result.get("reasoning", "Score calculated.")


# ─── Core scoring pipeline ────────────────────────────────────────────────────
def score_features(raw_features: dict) -> dict:
    """
    Full inference pipeline — returns a rich result dict.
    Steps:
      1. Derive defaults + computed ratios
      2. Infer user mode → select model + explainer
      3. Translate frontend labels → training class names
      4. Encode categoricals (strings → int codes)
      5. Build ordered DataFrame
      6. Predict default probability
      7. Percentile-based score
      8. Business rules + confidence override
      9. SHAP explanations
     10. LLM ai_explanation (Groq)
    """
    # 1. Derive
    features = derive_computed_features(raw_features)

    # 2. Route
    user_mode = _infer_user_mode(features)
    model, explainer = _get_model_and_explainer(user_mode)

    # 3–4. Translate + encode (must happen before _to_df)
    features = translate_frontend_values(features)
    features = encode_categoricals(features)

    # Heuristic path when models not loaded
    if not _artifacts_ready:
        result = _heuristic_score(features)
        result["ai_explanation"] = explain_with_llm(result)
        result["reasoning"]      = result["ai_explanation"] or result["reasoning"]
        return result

    # 5. DataFrame in exact training column order
    df_input = pd.DataFrame(
        [{col: float(features.get(col, FEATURE_DEFAULTS.get(col, 0))) for col in STUDENT_FEATURES}],
        columns=STUDENT_FEATURES,
    ).fillna(0)

    # 6. Predict
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df_input)[0][1])
    else:
        prob = float(np.clip(model.predict(df_input)[0], 0.0, 1.0))

    # 7. Percentile-based score + tier
    score = _prob_to_score(prob)
    tier  = _get_tier(score)

    # 8. Confidence + business rules
    confidence = _get_confidence(prob)
    conf_label = _get_confidence_label(confidence)
    ml_dec     = _get_ml_decision(prob)
    conf_note  = None
    if confidence < 0.30 and ml_dec == "APPROVE":
        ml_dec    = "REVIEW"
        conf_note = "Low model confidence — flagged for manual review"
    final_dec, rule = _apply_business_rules(features, ml_dec, prob)

    # 9. SHAP
    top_features, risk_drivers, protective_factors, reasoning, explanation = \
        _build_shap_outputs(explainer, df_input, prob)

    # Population percentile
    pct = _get_population_percentile(prob)

    # Build decision_reasoning (File 1 style)
    decision_reasoning_parts = [
        f"Credit score of {score} ({tier}) based on demographic and financial signals."
    ]
    if protective_factors:
        decision_reasoning_parts.append("Positive factors: " + "; ".join(protective_factors[:2]) + ".")
    if risk_drivers:
        decision_reasoning_parts.append("Risk factors: " + "; ".join(risk_drivers[:2]) + ".")
    if conf_note:
        decision_reasoning_parts.append(conf_note + ".")
    if rule:
        decision_reasoning_parts.append(f"Business rule applied: {rule}.")
    decision_reasoning_parts.append(f"Final decision: {final_dec}.")
    decision_reasoning = " ".join(decision_reasoning_parts)

    result = {
        # Core score
        "score":                 score,
        "risk_tier":             tier,
        "default_probability":   round(prob, 4),
        "approval_probability":  round((score - 300) / 600, 4),
        # Confidence (File 1)
        "confidence":            confidence,
        "confidence_label":      conf_label,
        # Decisions (File 1)
        "ml_decision":           ml_dec,
        "final_decision":        final_dec,
        "loan_eligible":         final_dec in ("APPROVE", "REVIEW"),
        # Population context (File 1 field names + File 2 alias)
        "population_percentile": pct,
        "population_summary":    f"Lower default risk than {pct}% of thin-file applicants",
        "percentile":            pct,
        # SHAP outputs — both vocabularies
        "risk_drivers":          risk_drivers,
        "protective_factors":    protective_factors,
        "top_features":          top_features,           # for SHAP chart (File 2)
        "explanation":           explanation,            # File 1 structure
        # Rules
        "rule_triggered":        rule,
        "confidence_note":       conf_note,
        # Reasoning (File 2 short form; overwritten by ai_explanation below)
        "reasoning":             reasoning,
        "decision_reasoning":    decision_reasoning,     # File 1 long form
        # Routing metadata (stripped before public responses)
        "user_mode":             user_mode,
        "features_used":         features,
    }

    # 10. LLM ai_explanation
    result["ai_explanation"] = explain_with_llm(result)
    # Overwrite reasoning so frontend (coachMessage: apiResponse.reasoning) gets AI text
    result["reasoning"]      = result["ai_explanation"] or result["reasoning"]

    return result


# ─── Counterfactual Engine (File 1 — richer, with fallbacks + target_score) ──
CF_ACTIONS = [
    {"feature": "EMPLOYED_YEARS",       "label": "Increase job tenure",
     "delta_pct":  0.60, "direction": 1,  "effort": "low",    "days": 180,
     "advice": "Stay with your current employer. Every additional year significantly reduces perceived risk."},
    {"feature": "AMT_INCOME_TOTAL",     "label": "Grow annual income",
     "delta_pct":  0.25, "direction": 1,  "effort": "medium", "days": 180,
     "advice": "A salary increase, side income, or bonus all count. Even a 25% rise meaningfully improves your profile."},
    {"feature": "AMT_CREDIT",           "label": "Reduce loan amount requested",
     "delta_pct": -0.25, "direction": -1, "effort": "low",    "days": 0,
     "advice": "Requesting a smaller loan lowers your debt-to-income ratio and signals discipline to lenders."},
    {"feature": "AMT_ANNUITY",          "label": "Reduce monthly EMI",
     "delta_pct": -0.20, "direction": -1, "effort": "low",    "days": 0,
     "advice": "Extending the loan tenure or choosing a smaller loan reduces EMI burden and improves affordability metrics."},
    {"feature": "FLAG_OWN_REALTY",      "label": "Acquire property ownership",
     "delta_val":  1,    "direction": 1,  "effort": "high",   "days": 365,
     "advice": "Owning property — even partially — substantially improves asset score and lender confidence."},
    {"feature": "REGION_RATING_CLIENT", "label": "Move to a better-rated region",
     "delta_val": -1,    "direction": -1, "effort": "high",   "days": 180,
     "advice": "Living in a higher-rated region reduces area-level risk. This is a long-term factor."},
    {"feature": "NAME_EDUCATION_TYPE",  "label": "Improve education level",
     "delta_val":  1,    "direction": 1,  "effort": "high",   "days": 540,
     "advice": "Higher qualifications correlate with lower default risk and better income prospects."},
]

FALLBACK_SUGGESTIONS = [
    {"feature": "EMPLOYED_YEARS",   "label": "Build employment tenure",
     "effort": "low",    "timeline_days": 365, "is_fallback": True,
     "advice": "Staying in stable employment for 2+ years is one of the most reliable ways to improve your creditworthiness."},
    {"feature": "AMT_CREDIT",       "label": "Request a smaller loan amount",
     "effort": "low",    "timeline_days": 0,   "is_fallback": True,
     "advice": "Reducing the requested loan amount lowers your debt-to-income ratio — a key signal for lenders."},
    {"feature": "AMT_INCOME_TOTAL", "label": "Increase your income",
     "effort": "medium", "timeline_days": 180, "is_fallback": True,
     "advice": "Any income growth — salary hike, freelance work, or rental income — improves multiple credit metrics simultaneously."},
]

IMMUTABLE_FEATURES = {
    "AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE", "NAME_FAMILY_STATUS",
}


def counterfactual(
    features: dict,
    locked: list = None,
    max_steps: int = 3,
    target_score: int = None,
) -> dict:
    locked   = set(locked or []) | IMMUTABLE_FEATURES
    features = derive_computed_features(features)
    original = score_features(features)
    moves    = []

    for action in CF_ACTIONS:
        feat = action["feature"]
        if feat in locked or feat not in features:
            continue

        orig_val   = float(features.get(feat, FEATURE_DEFAULTS.get(feat, 0)))
        best_delta = 0
        best_move  = None

        for multiplier in [1.0, 2.0, 3.0]:
            new_f = dict(features)
            if "delta_pct" in action:
                new_val = orig_val * (1 + action["delta_pct"] * multiplier)
            else:
                new_val = orig_val + action["delta_val"] * multiplier

            if action["direction"] == 1:
                new_val = max(orig_val, new_val)
            else:
                new_val = max(0, min(orig_val, new_val))

            new_f[feat] = new_val
            new_result  = score_features(new_f)
            delta       = new_result["score"] - original["score"]

            if delta >= best_delta:
                best_delta = delta
                best_move  = {
                    "feature":       feat,
                    "label":         action["label"],
                    "original_val":  round(orig_val, 4),
                    "new_val":       round(new_val, 4),
                    "score_delta":   delta,
                    "new_score":     new_result["score"],
                    "effort":        action["effort"],
                    "timeline_days": action["days"],
                    "advice":        action.get("advice", ""),
                    "is_fallback":   False,
                }

        if best_move is not None:
            moves.append(best_move)

    effort_rank = {"low": 0, "medium": 1, "high": 2, "n/a": 3}
    moves.sort(key=lambda x: (-x["score_delta"], effort_rank.get(x["effort"], 3)))

    positive_moves = [m for m in moves if m["score_delta"] > 0]
    if len(positive_moves) >= max_steps:
        moves = positive_moves[:max_steps]
    else:
        moves = positive_moves
        for m in [m for m in moves if m["score_delta"] == 0]:
            if len(moves) >= max_steps:
                break
            moves.append(m)

    if len(moves) < max_steps:
        used_features = {m["feature"] for m in moves}
        for fb in FALLBACK_SUGGESTIONS:
            if len(moves) >= max_steps:
                break
            if fb["feature"] not in used_features and fb["feature"] not in locked:
                fb_copy = dict(fb)
                fb_copy["original_val"] = round(float(features.get(fb["feature"], FEATURE_DEFAULTS.get(fb["feature"], 0))), 4)
                fb_copy["new_val"]      = None
                fb_copy["new_score"]    = None
                fb_copy["score_delta"]  = None
                moves.append(fb_copy)

    best_score = max(
        (m["new_score"] for m in moves if m.get("new_score")),
        default=original["score"],
    )

    result = {
        "original_score":       original["score"],
        "original_risk":        original["risk_tier"],
        "original_probability": original["default_probability"],
        "moves":                moves,
        "best_reachable_score": best_score,
    }

    if target_score:
        gap = max(0, target_score - original["score"])
        result["target_score"] = target_score
        result["score_gap"]    = gap
        result["on_track"]     = best_score >= target_score

    return result


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
    origins              = [FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:3000"],
    supports_credentials = True,
    allow_headers        = ["Content-Type", "Authorization"],
    methods              = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)

# ─── Auth helpers ─────────────────────────────────────────────────────────────
def require_auth(fn):
    @wraps(fn)
    def inner(*a, **kw):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return fn(*a, **kw)
    return inner

def uid():
    return session.get("user_id")

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


# ─── Public API ───────────────────────────────────────────────────────────────

@app.route("/api/score", methods=["POST"])
def api_score():
    """
    Public scoring endpoint.

    Accepts:
      { "features": { ... } }   — full feature dict (File 1 contract)
      { "persona": "ravi" }     — demo shortcut (File 2 contract)

    Returns the full superset response (all File 1 + File 2 fields).
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "No JSON body provided"}), 400

    # Persona shortcut (File 2)
    if "persona" in body:
        persona = DEMO_PERSONAS.get(body["persona"])
        if not persona:
            return jsonify({"error": "Unknown persona. Use: ravi, priya, deepa"}), 400
        features = persona["features"]
    else:
        # Strict validation (File 1)
        if "features" not in body:
            return jsonify({
                "error": "Missing 'features' key. Expected format: { features: {...} }"
            }), 400
        features = body["features"]
        if not isinstance(features, dict):
            return jsonify({"error": "'features' must be an object"}), 400

        REQUIRED_FIELDS = [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "EMPLOYED_YEARS",
            "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
            "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
            "FLAG_OWN_REALTY", "FLAG_OWN_CAR",
            "NAME_HOUSING_TYPE",
            "REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE",
        ]
        missing = [f for f in REQUIRED_FIELDS if f not in features]
        if missing:
            return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400

        unknown = [k for k in features if k not in STUDENT_FEATURES]
        if unknown:
            log.warning("Unknown fields received: %s", unknown)

    log.info("Scoring request — %d features", len(features))

    try:
        result = score_features(features)
        result.pop("features_used", None)
        result.pop("user_mode", None)
        return jsonify(result)
    except Exception as exc:
        log.exception("Scoring error")
        return jsonify({"error": "Scoring failed", "details": str(exc)}), 500


@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    """
    Returns actionable improvement steps.
    Body: { features: {...}, locked_features: [...], max_steps: 3, target_score: 750 }
    Always returns at least max_steps recommendations.
    """
    body = request.get_json(silent=True) or {}
    if "persona" in body:
        persona  = DEMO_PERSONAS.get(body["persona"])
        features = persona["features"] if persona else {}
    else:
        features = body.get("features", {})

    locked       = body.get("locked_features", [])
    max_steps    = int(body.get("max_steps", 3))
    target_score = body.get("target_score")
    if target_score:
        target_score = int(target_score)

    return jsonify(counterfactual(features, locked, max_steps, target_score))


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
        "original_explanation": base["explanation"],
        "new_score":            new["score"],
        "new_risk":             new["risk_tier"],
        "new_probability":      new["default_probability"],
        "new_explanation":      new["explanation"],
        "score_delta":          new["score"] - base["score"],
        "changes_applied":      changes,
    })


@app.route("/api/features", methods=["GET"])
def api_features():
    """Returns feature list, defaults, demo personas, and explainability metadata."""
    return jsonify({
        "features":             STUDENT_FEATURES,
        "defaults":             FEATURE_DEFAULTS,
        "demo_personas":        {k: v["label"] for k, v in DEMO_PERSONAS.items()},
        "feature_explanations": EXPLANATION_TEMPLATES,
        "model_ready":          _artifacts_ready,
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
            "features":   result.get("features_used", {}),
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
    body         = request.get_json(silent=True) or {}
    features     = body.get("features", {})
    locked       = body.get("locked_features", [])
    target_score = body.get("target_score")
    if target_score:
        target_score = int(target_score)
    result = counterfactual(features, locked, int(body.get("max_steps", 3)), target_score)
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
        {"score": e["score"], "risk_tier": e["risk_tier"], "date": e["created_at"].isoformat()}
        for e in events
    ]})


# ─── AI Chat (Groq) ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NeoScore's financial AI coach.
You help users understand their credit score, explain what the features mean,
and give practical, India-specific advice on improving creditworthiness.
Be concise, empathetic, and actionable. Avoid jargon.
When mentioning scores, use the 300-900 scale.
Reference specific features from the user's profile when relevant.
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
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model":      "llama-3.3-70b-versatile",
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
        "status":       "ok",
        "service":      "neoscore",
        "model_a":      type(_model_A).__name__,
        "model_b":      type(_model_B).__name__,
        "artifacts":    _artifacts_ready,
        "shap_a":       _explainer_A is not None,
        "shap_b":       _explainer_B is not None,
        "pop_dist":     len(_pop_prob_dist) > 0,
        "thresholds":   _thresholds,
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
