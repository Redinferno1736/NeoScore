"""
NeoScore Backend — Minimal Single-File Edition
Stack: Flask · MongoDB · Google OAuth · Groq AI Coach

SETUP
─────
pip install flask flask-cors flask-session pymongo dnspython \
            numpy scikit-learn joblib python-dotenv requests

.env vars:
    FLASK_SECRET_KEY=...
    FRONTEND_URL=http://localhost:3000
    MONGO_URI=mongodb://localhost:27017
    MONGO_DB_NAME=neoscore
    GOOGLE_CLIENT_ID=...
    GOOGLE_CLIENT_SECRET=...
    GOOGLE_REDIRECT_URI=http://localhost:5000/auth/callback/google
    GROQ_API_KEY=...
    STUDENT_MODEL_PATH=student_model.joblib   # optional

ROUTES
──────
GET  /health
GET  /auth/login/google
GET  /auth/callback/google
GET  /auth/me
POST /auth/logout

POST /api/score          — public API (no auth needed, returns JSON)
POST /api/counterfactual
POST /api/simulate
GET  /api/features

GET  /score/history
GET  /score/history/trend

POST /chat
"""

import os, math, secrets, logging
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import requests as http
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, session
from flask_cors import CORS
from flask_session import Session
from pymongo import ASCENDING, DESCENDING, MongoClient

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
SECRET_KEY        = os.getenv("FLASK_SECRET_KEY", "dev-secret")
FRONTEND_URL      = os.getenv("FRONTEND_URL", "http://localhost:3000")
MONGO_URI         = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME     = os.getenv("MONGO_DB_NAME", "neoscore")
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_SECRET     = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT   = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/auth/callback/google")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
IS_PROD           = os.getenv("FLASK_ENV") == "production"

SCORE_MIN, SCORE_MAX = 300, 900

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
CONFIG_DIR = os.path.join(BASE_DIR, "config")
MODELS_DIR = BASE_DIR   # since your .pkl files are directly in backend/

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS_PATH = os.path.join(BASE_DIR, "config", "thresholds.json")

STUDENT_A_PATH = os.path.join(MODELS_DIR, "student_A.pkl")
STUDENT_B_PATH = os.path.join(MODELS_DIR, "student_B.pkl")

EXPLAINER_A_PATH = os.path.join(MODELS_DIR, "explainer_A.pkl")
EXPLAINER_B_PATH = os.path.join(MODELS_DIR, "explainer_B.pkl")

ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")

POP_PROBS_PATH = os.path.join(MODELS_DIR, "population_prob_distribution.npy")
POP_SCORES_PATH = os.path.join(MODELS_DIR, "population_score_distribution.npy")


# Categorical label-encoding maps
EDUCATION_MAP    = {"Lower secondary": 0, "Secondary / secondary special": 1,
                    "Incomplete higher": 2, "Higher education": 3, "Academic degree": 4, "Unknown": 1}
FAMILY_MAP       = {"Civil marriage": 0, "Married": 1, "Separated": 2,
                    "Single / not married": 3, "Widow": 4, "Unknown": 3}
INCOME_TYPE_MAP  = {"Businessman": 0, "Commercial associate": 1, "Maternity leave": 2,
                    "Pensioner": 3, "State servant": 4, "Student": 5,
                    "Unemployed": 6, "Working": 7, "Unknown": 7}
HOUSING_MAP      = {"Co-op apartment": 0, "House / apartment": 1, "Municipal apartment": 2,
                    "Office apartment": 3, "Rented apartment": 4, "With parents": 5, "Unknown": 1}

# Numerical defaults for missing fields
FEATURE_DEFAULTS = {
    "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2,
    # Categoricals stored as raw strings — encode_categoricals() converts them before model inference
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS":  "Married",
    "NAME_INCOME_TYPE":    "Working",
    "OCCUPATION_TYPE":     "Laborers",
    "ORGANIZATION_TYPE":   "Business Entity Type 3",
    "NAME_HOUSING_TYPE":   "House / apartment",
    "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 1,
    "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000, "AMT_GOODS_PRICE": 450000,
    "REGION_POPULATION_RELATIVE": 0.02, "REGION_RATING_CLIENT": 2,
    "AGE_YEARS": 35, "EMPLOYED_YEARS": 5,
    # Derived features — computed dynamically, defaults only used as last resort
    "DEBT_TO_INCOME": 2.0, "ANNUITY_TO_INCOME": 0.1,
    "CREDIT_TO_GOODS": 1.1, "INCOME_PER_PERSON": 60000,
    "CHILDREN_RATIO": 0.0, "EMI_BURDEN": 0.1,
    "INCOME_STABILITY": 0.14, "LOAN_TO_INCOME": 0.23,
    "FINANCIAL_PRESSURE": 0.2, "STABILITY_SCORE": 0.25,
    "ASSET_SCORE": 0.6, "INCOME_ADEQUACY": 0.6,
}

# ─── Demo personas ─────────────────────────────────────────────────────────────
DEMO_PERSONAS = {
    # All categorical values use raw Home Credit class strings — same as frontend dropdowns.
    # Derived ratio features are NOT pre-computed here; derive_computed_features() handles them.
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
        _mongo.users.create_index([("provider_id", ASCENDING), ("provider", ASCENDING)], unique=True)
        _mongo.score_history.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        log.info("MongoDB ready")
    return _mongo

def now(): return datetime.now(timezone.utc)

def ser(doc):
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    return doc

# ─── Real model scorer ────────────────────────────────────────────────────────
import pickle, json as _json
import pandas as pd

class ThinFileCreditScorer:
    """
    Production inference wrapper backed by the trained dual-student XGBoost models,
    SHAP explainers, population distributions, and dynamic thresholds from Notebook 3.
    Load once at startup; call .score() per request.
    """
    def __init__(self):
        self._ready = False

        try:
            with open(STUDENT_A_PATH, "rb") as f:
                self.model_A = pickle.load(f)

            with open(STUDENT_B_PATH, "rb") as f:
                self.model_B = pickle.load(f)

            with open(EXPLAINER_A_PATH, "rb") as f:
                self.exp_A = pickle.load(f)

            with open(EXPLAINER_B_PATH, "rb") as f:
                self.exp_B = pickle.load(f)

            with open(ENCODERS_PATH, "rb") as f:
                self.enc = pickle.load(f)

            with open(THRESHOLDS_PATH) as f:
                self.thresh = _json.load(f)

            self.pop_probs = np.load(POP_PROBS_PATH)
            self.pop_scores = np.load(POP_SCORES_PATH)

            self._ready = True
            log.info("Scorer loaded successfully")

        except Exception as e:
            self._ready = False
            log.warning(f"Scorer fallback mode: {e}")
    # SHAP → plain-English explanation templates
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

    CATEGORICAL_COLS = {
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_HOUSING_TYPE",
    }

    def _encode_categoricals(self, features: dict) -> dict:
        """
        Convert raw string categorical values → integer codes using the
        label_encoders.pkl that was fit during training.
        Integer-valued inputs (FLAG_OWN_CAR, etc.) are left untouched.
        Unknown strings fall back to 0 with a warning.
        """
        if not self._ready:
            # No encoders loaded — use hardcoded fallback integer map
            FALLBACK = {
                "NAME_EDUCATION_TYPE": {"Lower secondary": 0, "Secondary / secondary special": 1,
                                        "Incomplete higher": 2, "Higher education": 3, "Academic degree": 4},
                "NAME_FAMILY_STATUS":  {"Civil marriage": 0, "Married": 1, "Separated": 2,
                                        "Single / not married": 3, "Widow": 4},
                "NAME_INCOME_TYPE":    {"Businessman": 0, "Commercial associate": 1, "Maternity leave": 2,
                                        "Pensioner": 3, "State servant": 4, "Student": 5,
                                        "Unemployed": 6, "Working": 7},
                "NAME_HOUSING_TYPE":   {"Co-op apartment": 0, "House / apartment": 1, "Municipal apartment": 2,
                                        "Office apartment": 3, "Rented apartment": 4, "With parents": 5},
            }
            f = dict(features)
            for col, mapping in FALLBACK.items():
                val = f.get(col)
                if isinstance(val, str):
                    f[col] = mapping.get(val, 0)
            # OCCUPATION_TYPE and ORGANIZATION_TYPE: unknown → 0
            for col in ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"):
                if isinstance(f.get(col), str):
                    f[col] = 0
            return f

        f = dict(features)
        for col in self.CATEGORICAL_COLS:
            val = f.get(col)
            if val is None or isinstance(val, (int, float)):
                continue   # already encoded or missing — leave as-is
            encoder = self.enc.get(col)
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
        try:
            with open(STUDENT_A_PATH,   "rb") as f: self.model_A   = pickle.load(f)
            with open(STUDENT_B_PATH,   "rb") as f: self.model_B   = pickle.load(f)
            with open(EXPLAINER_A_PATH, "rb") as f: self.exp_A     = pickle.load(f)
            with open(EXPLAINER_B_PATH, "rb") as f: self.exp_B     = pickle.load(f)
            with open(ENCODERS_PATH,    "rb") as f: self.enc       = pickle.load(f)
            with open(THRESHOLDS_PATH)        as f: self.thresh    = _json.load(f)
            self.pop_probs  = np.load(POP_PROBS_PATH)
            self.pop_scores = np.load(POP_SCORES_PATH)
            self._ready = True
            log.info("ThinFileCreditScorer: all artifacts loaded successfully")
        except Exception as e:
            self._ready = False
            log.warning("ThinFileCreditScorer: could not load artifacts (%s) — falling back to heuristic", e)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _derive(self, raw: dict) -> dict:
        """Compute all derived/composite features from a raw feature dict."""
        f = {**FEATURE_DEFAULTS, **raw}
        income   = max(float(f.get("AMT_INCOME_TOTAL", 180000)), 1)
        credit   = max(float(f.get("AMT_CREDIT", 500000)), 0)
        annuity  = max(float(f.get("AMT_ANNUITY", 25000)), 0)
        goods    = max(float(f.get("AMT_GOODS_PRICE", 450000)), 1)
        emp_yrs  = max(float(f.get("EMPLOYED_YEARS", 5)), 0)
        age_yrs  = max(float(f.get("AGE_YEARS", 35)), 1)
        children = max(float(f.get("CNT_CHILDREN", 0)), 0)
        fam      = max(float(f.get("CNT_FAM_MEMBERS", 2)), 1)
        own_car  = float(f.get("FLAG_OWN_CAR", 0))
        own_re   = float(f.get("FLAG_OWN_REALTY", 0))

        if "DEBT_TO_INCOME"    not in raw: f["DEBT_TO_INCOME"]    = min(credit / income, 20.0)
        if "ANNUITY_TO_INCOME" not in raw: f["ANNUITY_TO_INCOME"] = min(annuity / income, 2.0)
        if "CREDIT_TO_GOODS"   not in raw: f["CREDIT_TO_GOODS"]   = min(credit / goods, 5.0)
        if "INCOME_PER_PERSON" not in raw: f["INCOME_PER_PERSON"] = min(income / fam, 1_000_000)
        if "CHILDREN_RATIO"    not in raw: f["CHILDREN_RATIO"]    = min(children / fam, 1.0)
        if "EMI_BURDEN"        not in raw: f["EMI_BURDEN"]        = min(annuity / income, 1.0)
        if "INCOME_STABILITY"  not in raw: f["INCOME_STABILITY"]  = min(emp_yrs / age_yrs, 1.0)
        if "LOAN_TO_INCOME"    not in raw: f["LOAN_TO_INCOME"]    = min(credit / (income * 12), 5.0)

        emi_burden  = min(f["EMI_BURDEN"], 1.0)
        loan_income = min(f["LOAN_TO_INCOME"], 1.0)
        child_ratio = min(f["CHILDREN_RATIO"], 1.0)
        inc_stab    = min(f["INCOME_STABILITY"], 1.0)

        if "FINANCIAL_PRESSURE" not in raw:
            f["FINANCIAL_PRESSURE"] = min(0.40 * emi_burden + 0.35 * loan_income + 0.25 * child_ratio, 1.0)
        if "STABILITY_SCORE" not in raw:
            f["STABILITY_SCORE"] = min(0.50 * inc_stab + 0.30 * min(emp_yrs / 40, 1) + 0.20 * own_re, 1.0)
        if "ASSET_SCORE"     not in raw:
            f["ASSET_SCORE"]     = min(0.60 * own_re + 0.40 * own_car, 1.0)
        if "INCOME_ADEQUACY" not in raw:
            f["INCOME_ADEQUACY"] = min(income / max(annuity * 12, 1), 10.0)

        return f

    def _to_df(self, features: dict) -> pd.DataFrame:
        """Build a single-row DataFrame in STUDENT_FEATURES column order."""
        row = {f: float(features.get(f, FEATURE_DEFAULTS.get(f, 0))) for f in STUDENT_FEATURES}
        return pd.DataFrame([row])[STUDENT_FEATURES]

    def _signal_mode(self, raw: dict) -> str:
        """Determine routing mode from bureau/behavioural signal strength."""
        sig = (
            float(raw.get("bureau_count", 0) > 0) * 0.35 +
            float(raw.get("inst_count",   0) > 0) * 0.35 +
            float(raw.get("prev_app_count",0) > 0) * 0.20
        )
        return "B_pure_thin" if sig == 0 else "A_near_thin"

    def _prob_to_score(self, prob: float) -> int:
        """Percentile-based scoring: fraction of population with higher default prob."""
        pct   = float((self.pop_probs > prob).mean())
        score = int(np.clip(300 + pct * 600, 300, 900))
        return score

    def _get_tier(self, score: int) -> str:
        if score >= 800: return "Excellent"
        if score >= 740: return "Very Good"
        if score >= 670: return "Good"
        if score >= 580: return "Fair"
        return "Poor"

    def _get_ml_decision(self, prob: float) -> str:
        t = self.thresh
        if prob < t["approve"]:  return "APPROVE"
        if prob >= t["reject"]:  return "REJECT"
        return "REVIEW"

    def _get_confidence(self, prob: float) -> float:
        return round(abs(prob - 0.5) * 2, 3)

    def _get_confidence_label(self, c: float) -> str:
        if c >= 0.70: return "High"
        if c >= 0.40: return "Medium"
        return "Low"

    def _apply_business_rules(self, features: dict, ml_decision: str, prob: float):
        emi   = features.get("EMI_BURDEN", 0)
        lti   = features.get("LOAN_TO_INCOME", 0)
        ia    = features.get("INCOME_ADEQUACY", 10)
        age   = features.get("AGE_YEARS", 35)
        if emi > 0.60:
            return "REJECT", "EMI burden exceeds 60% of income"
        if lti > 10 and ml_decision == "APPROVE":
            return "REVIEW", "Loan amount exceeds 10x annual income"
        if ia < 0.5 and ml_decision == "APPROVE":
            return "REVIEW", "Income insufficient relative to loan obligations"
        if age < 21 and ml_decision == "APPROVE":
            return "REVIEW", "Applicant age below 21 — manual review required"
        return ml_decision, None

    def _shap_explanation(self, shap_vals, top_n=3):
        paired = list(zip(STUDENT_FEATURES, shap_vals))
        drivers    = sorted([(f, v) for f, v in paired if v > 0], key=lambda x: -x[1])[:top_n]
        protective = sorted([(f, v) for f, v in paired if v < 0], key=lambda x:  x[1])[:top_n]

        def text(feat, direction):
            tmpl = self.EXPLANATION_TEMPLATES.get(feat)
            if tmpl:
                return tmpl[0 if direction == "high" else 1]
            readable = feat.replace("_", " ").title()
            return f"{'High' if direction == 'high' else 'Low'} {readable}"

        return (
            [text(f, "high") for f, _ in drivers],
            [text(f, "low")  for f, _ in protective],
        )

    def _narrative(self, pd_: float) -> str:
        if pd_ < 0.083:  return "Strong credit profile — you are in the lower-risk tier of applicants."
        if pd_ < 0.149:  return "Moderate profile — a few improvements could move you to the approval tier."
        return "Elevated risk profile — focus on reducing financial pressure and building employment stability."

    def _build_reasoning(self, score, tier, conf, ml_dec, final_dec, rule, conf_note, drivers, protective):
        parts = [f"Credit score of {score} ({tier}) based on demographic and financial signals."]
        if protective: parts.append("Positive factors: " + "; ".join(protective[:2]) + ".")
        if drivers:    parts.append("Risk factors: "     + "; ".join(drivers[:2])    + ".")
        if conf_note:  parts.append(conf_note + ".")
        if rule:       parts.append(f"Business rule applied: {rule}.")
        parts.append(f"Final decision: {final_dec}.")
        return " ".join(parts)

    def _get_population_percentile(self, prob: float) -> float:
        return round(float((self.pop_probs >= prob).mean() * 100), 1)

    # ── Heuristic fallback (used when real models not available) ───────────────

    def _heuristic_score(self, features: dict) -> dict:
        fp      = min(features.get("FINANCIAL_PRESSURE", 0.2), 1.0)
        ss      = min(features.get("STABILITY_SCORE",    0.25), 1.0)
        ia      = min(features.get("INCOME_ADEQUACY",    0.6) / 10.0, 1.0)
        emi     = min(features.get("EMI_BURDEN",         0.1) / 0.5, 1.0)
        dti     = min(features.get("DEBT_TO_INCOME",     2.0) / 5.0, 1.0)
        lti     = min(features.get("LOAN_TO_INCOME",     0.23), 1.0)
        emp     = min(features.get("EMPLOYED_YEARS",     5) / 20.0, 1.0)
        own_re  = min(features.get("FLAG_OWN_REALTY",   0), 1.0)
        edu     = features.get("NAME_EDUCATION_TYPE", 1) / 4.0

        pd_ = max(0.03, min(0.95,
            0.30 * fp + 0.20 * emi + 0.15 * dti + 0.10 * lti
            - 0.10 * ss - 0.08 * ia - 0.06 * emp - 0.04 * own_re - 0.03 * edu + 0.08
        ))
        # log-odds stretch for score
        lo    = math.log(max(pd_, 0.01) / (1 - min(pd_, 0.99)))
        norm  = (lo + 4.6) / 9.2
        score = int(np.clip(SCORE_MAX - round((SCORE_MAX - SCORE_MIN) * norm), SCORE_MIN, SCORE_MAX))
        tier  = self._get_tier(score)
        conf  = self._get_confidence(pd_)
        ml_dec   = "APPROVE" if pd_ < 0.15 else "REJECT" if pd_ >= 0.30 else "REVIEW"
        final, rule = self._apply_business_rules(features, ml_dec, pd_)
        return {
            "score": score, "risk_tier": tier,
            "default_probability": round(pd_, 4),
            "approval_probability": round((score - 300) / 600, 4),
            "confidence": conf, "confidence_label": self._get_confidence_label(conf),
            "ml_decision": ml_dec, "final_decision": final,
            "loan_eligible": final in ("APPROVE", "REVIEW"),
            "user_mode": "heuristic",
            "population_percentile": None,
            "population_summary": "Model artifacts not loaded — heuristic estimate only.",
            "rule_triggered": rule, "confidence_note": None,
            "risk_drivers": [], "protective_factors": [],
            "decision_reasoning": self._build_reasoning(score, tier, conf, ml_dec, final, rule, None, [], []),
            "explanation": {"narrative": self._narrative(pd_), "drivers_positive": [], "drivers_negative": []},
        }

    # ── Public interface ───────────────────────────────────────────────────────

    def score(self, raw_features: dict) -> dict:
        """Full scoring pipeline. Returns rich result dict."""
        features = self._derive(raw_features)
        features = self._encode_categoricals(features)   # strings → ints before model

        if not self._ready:
            result = self._heuristic_score(features)
            result["features_used"] = features
            return result

        mode = self._signal_mode(raw_features)
        model    = self.model_A   if mode != "B_pure_thin" else self.model_B
        explainer = self.exp_A    if mode != "B_pure_thin" else self.exp_B

        X    = self._to_df(features)
        prob = float(np.clip(model.predict(X)[0], 0, 1))

        score      = self._prob_to_score(prob)
        tier       = self._get_tier(score)
        confidence = self._get_confidence(prob)
        conf_label = self._get_confidence_label(confidence)
        ml_dec     = self._get_ml_decision(prob)

        # Confidence override
        conf_note = None
        if confidence < 0.30 and ml_dec == "APPROVE":
            ml_dec    = "REVIEW"
            conf_note = "Low model confidence — flagged for manual review"

        final_dec, rule = self._apply_business_rules(features, ml_dec, prob)

        # SHAP
        try:
            sv = explainer.shap_values(X)[0]
            risk_drivers, protective = self._shap_explanation(sv)
        except Exception as e:
            log.warning("SHAP failed: %s", e)
            risk_drivers, protective = [], []

        pct = self._get_population_percentile(prob)

        explanation = {
            "narrative":        self._narrative(prob),
            "drivers_positive": [{"text": t} for t in protective],
            "drivers_negative": [{"text": t} for t in risk_drivers],
        }

        return {
            "score":                score,
            "risk_tier":            tier,
            "default_probability":  round(prob, 4),
            "approval_probability": round((score - 300) / 600, 4),
            "confidence":           confidence,
            "confidence_label":     conf_label,
            "ml_decision":          ml_dec,
            "final_decision":       final_dec,
            "loan_eligible":        final_dec in ("APPROVE", "REVIEW"),
            "user_mode":            mode,
            "population_percentile": pct,
            "population_summary":   f"Lower default risk than {pct}% of thin-file applicants",
            "risk_drivers":         risk_drivers,
            "protective_factors":   protective,
            "rule_triggered":       rule,
            "confidence_note":      conf_note,
            "decision_reasoning":   self._build_reasoning(
                score, tier, confidence, ml_dec, final_dec,
                rule, conf_note, risk_drivers, protective
            ),
            "explanation":          explanation,
            "features_used":        features,
        }


_scorer: ThinFileCreditScorer = None

def get_scorer() -> ThinFileCreditScorer:
    global _scorer
    if _scorer is None:
        _scorer = ThinFileCreditScorer()
    return _scorer


# ─── Thin wrappers kept for internal use ─────────────────────────────────────

def derive_computed_features(features: dict) -> dict:
    return get_scorer()._derive(features)


def score_features(features: dict) -> dict:
    """Convenience wrapper — calls the real scorer (encodes categoricals internally)."""
    return get_scorer().score(features)


# ─── Counterfactual Engine ────────────────────────────────────────────────────
# Extended action catalogue — more levers, better coverage
CF_ACTIONS = [
    # Employment / stability
    {"feature": "EMPLOYED_YEARS",       "label": "Increase job tenure",         "delta_pct":  0.60, "direction": 1,  "effort": "low",    "days": 180,
     "advice": "Stay with your current employer. Every additional year significantly reduces perceived risk."},
    # Income
    {"feature": "AMT_INCOME_TOTAL",     "label": "Grow annual income",           "delta_pct":  0.25, "direction": 1,  "effort": "medium", "days": 180,
     "advice": "A salary increase, side income, or bonus all count. Even a 25% rise meaningfully improves your profile."},
    # Loan size reduction
    {"feature": "AMT_CREDIT",           "label": "Reduce loan amount requested", "delta_pct": -0.25, "direction": -1, "effort": "low",    "days": 0,
     "advice": "Requesting a smaller loan lowers your debt-to-income ratio and signals discipline to lenders."},
    # Annuity / EMI reduction
    {"feature": "AMT_ANNUITY",          "label": "Reduce monthly EMI",           "delta_pct": -0.20, "direction": -1, "effort": "low",    "days": 0,
     "advice": "Extending the loan tenure or choosing a smaller loan reduces EMI burden and improves affordability metrics."},
    # Asset ownership
    {"feature": "FLAG_OWN_REALTY",      "label": "Acquire property ownership",   "delta_val":  1,    "direction": 1,  "effort": "high",   "days": 365,
     "advice": "Owning property — even partially — substantially improves asset score and lender confidence."},
    # Region / rating (advisory only)
    {"feature": "REGION_RATING_CLIENT", "label": "Move to a better-rated region","delta_val": -1,    "direction": -1, "effort": "high",   "days": 180,
     "advice": "Living in a higher-rated region reduces area-level risk. This is a long-term factor."},
    # Education
    {"feature": "NAME_EDUCATION_TYPE",  "label": "Improve education level",      "delta_val":  1,    "direction": 1,  "effort": "high",   "days": 540,
     "advice": "Higher qualifications correlate with lower default risk and better income prospects."},
]

# Static fallback suggestions shown when model is insensitive to feature changes.
# Ordered by general impact for a median-risk Indian applicant.
FALLBACK_SUGGESTIONS = [
    {
        "feature": "EMPLOYED_YEARS",
        "label": "Build employment tenure",
        "effort": "low",
        "timeline_days": 365,
        "advice": "Staying in stable employment for 2+ years is one of the most reliable ways to improve your creditworthiness.",
        "score_delta": None,
        "is_fallback": True,
    },
    {
        "feature": "AMT_CREDIT",
        "label": "Request a smaller loan amount",
        "effort": "low",
        "timeline_days": 0,
        "advice": "Reducing the requested loan amount lowers your debt-to-income ratio — a key signal for lenders.",
        "score_delta": None,
        "is_fallback": True,
    },
    {
        "feature": "AMT_INCOME_TOTAL",
        "label": "Increase your income",
        "effort": "medium",
        "timeline_days": 180,
        "advice": "Any income growth — salary hike, freelance work, or rental income — improves multiple credit metrics simultaneously.",
        "score_delta": None,
        "is_fallback": True,
    },
]


def counterfactual(features: dict, locked: list = None, max_steps: int = 3, target_score: int = None) -> dict:
    """
    Adaptive counterfactual engine.

    Improvements over v1:
    - Accepts any positive score_delta (threshold removed to avoid empty results)
    - Tries proportionally larger deltas if initial nudge produces no change
    - Falls back to curated static suggestions when model is insensitive
    - Supports target_score: shows how far each move takes you toward the goal
    - Always returns at least max_steps recommendations
    """
    locked   = set(locked or [])
    features = derive_computed_features(features)
    original = score_features(features)
    moves    = []

    for action in CF_ACTIONS:
        feat = action["feature"]
        if feat in locked or feat not in features:
            continue

        orig_val = float(features.get(feat, FEATURE_DEFAULTS.get(feat, 0)))
        best_delta = 0
        best_move  = None

        # Try progressively larger nudges (×1, ×2, ×3) to overcome model flatness
        for multiplier in [1.0, 2.0, 3.0]:
            new_f = dict(features)
            if "delta_pct" in action:
                new_val = orig_val * (1 + action["delta_pct"] * multiplier)
            else:
                new_val = orig_val + action["delta_val"] * multiplier

            # Clip to sensible bounds
            if action["direction"] == 1:
                new_val = max(orig_val, new_val)   # must increase
            else:
                new_val = max(0, min(orig_val, new_val))  # must decrease, non-negative

            new_f[feat]  = new_val
            new_result   = score_features(new_f)
            delta        = new_result["score"] - original["score"]

            # Accept any improvement (removed delta > 0 hard gate)
            if delta >= best_delta:
                best_delta = delta
                best_move  = {
                    "feature":        feat,
                    "label":          action["label"],
                    "original_val":   round(orig_val, 4),
                    "new_val":        round(new_val, 4),
                    "score_delta":    delta,
                    "new_score":      new_result["score"],
                    "effort":         action["effort"],
                    "timeline_days":  action["days"],
                    "advice":         action.get("advice", ""),
                    "is_fallback":    False,
                }

        # Include even zero-delta moves so we always have candidates to rank
        if best_move is not None:
            moves.append(best_move)

    # Sort by score_delta desc, then by effort (low > medium > high)
    effort_rank = {"low": 0, "medium": 1, "high": 2, "n/a": 3}
    moves.sort(key=lambda x: (-x["score_delta"], effort_rank.get(x["effort"], 3)))

    # Filter to genuinely positive moves first; if not enough, keep zero-delta ones too
    positive_moves = [m for m in moves if m["score_delta"] > 0]
    if len(positive_moves) >= max_steps:
        moves = positive_moves[:max_steps]
    else:
        # Pad with zero-delta moves, then fallbacks
        moves = positive_moves
        for m in [m for m in moves if m["score_delta"] == 0]:
            if len(moves) >= max_steps:
                break
            moves.append(m)

    # If still short, pad with curated fallback suggestions
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
                moves.append(fb_copy)

    best_score = max((m["new_score"] for m in moves if m.get("new_score")), default=original["score"])

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
    SESSION_COOKIE_SAMESITE = "Lax",
    SESSION_COOKIE_SECURE   = IS_PROD,
)
Session(app)
CORS(app, origins=[FRONTEND_URL], supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# ─── Auth helpers ─────────────────────────────────────────────────────────────
def require_auth(fn):
    @wraps(fn)
    def inner(*a, **kw):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return fn(*a, **kw)
    return inner

def uid(): return session.get("user_id")

# ─── Auth routes — Google only ────────────────────────────────────────────────
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
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + "&".join(f"{k}={v}" for k, v in params.items())
    return redirect(url)

@app.route("/auth/callback/google")
def google_callback():
    if request.args.get("state") != session.pop("oauth_state", None):
        return jsonify({"error": "Invalid state"}), 400

    code = request.args.get("code")
    token_res = http.post("https://oauth2.googleapis.com/token", data={
        "code": code, "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_SECRET, "redirect_uri": GOOGLE_REDIRECT,
        "grant_type": "authorization_code",
    }).json()

    access_token = token_res.get("access_token")
    if not access_token:
        return jsonify({"error": "Token exchange failed"}), 400

    info = http.get("https://www.googleapis.com/oauth2/v3/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"}).json()

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
    return redirect(f"{FRONTEND_URL}/dashboard")

@app.route("/auth/me")
def auth_me():
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 200
    user = db().users.find_one({"_id": ObjectId(uid())})
    if not user:
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": {
        "id": uid(), "name": user.get("name"),
        "email": user.get("email"), "picture": user.get("picture"),
    }})

@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})

@app.route("/api/score", methods=["POST"])
def api_score():
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "No JSON body provided"}), 400

    # 🚨 STRICT: must send { features: {...} }
    if "features" not in body:
        return jsonify({
            "error": "Missing 'features' key. Expected format: { features: {...} }"
        }), 400

    features = body["features"]

    if not isinstance(features, dict):
        return jsonify({"error": "'features' must be an object"}), 400

    # 🚨 REQUIRED FIELDS (match frontend exactly)
    REQUIRED_FIELDS = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "EMPLOYED_YEARS",
        "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "FLAG_OWN_REALTY", "FLAG_OWN_CAR",
        "NAME_HOUSING_TYPE",
        "REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE"
    ]

    missing = [f for f in REQUIRED_FIELDS if f not in features]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    # 🚨 DEBUG LOG (keep this for now)
    print("\n=== INCOMING FEATURES ===")
    for k, v in features.items():
        print(f"{k}: {v}")
    print("========================\n")

    # 🚨 OPTIONAL: detect unknown fields (helps catch typos)
    unknown = [k for k in features if k not in STUDENT_FEATURES]
    if unknown:
        print("⚠️ Unknown fields received:", unknown)

    try:
        result = score_features(features)
        result.pop("features_used", None)
        result.pop("user_mode", None)
        return jsonify(result)

    except Exception as e:
        print("❌ ERROR DURING SCORING:", str(e))
        return jsonify({
            "error": "Scoring failed",
            "details": str(e)
        }), 500


@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    """
    Returns actionable steps to improve score.
    Body: { features: {...}, locked_features: [...], max_steps: 3, target_score: 750 }
    Always returns at least max_steps recommendations — never an empty list.
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
    What-if simulation. Provide base features + a dict of changes.
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
    scorer = get_scorer()
    return jsonify({
        "features":             STUDENT_FEATURES,
        "defaults":             FEATURE_DEFAULTS,
        "demo_personas":        {k: v["label"] for k, v in DEMO_PERSONAS.items()},
        "feature_explanations": scorer.EXPLANATION_TEMPLATES,
        "model_ready":          scorer._ready,
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
            "features":   result["features_used"],
            "created_at": now(),
        })
    except Exception as e:
        log.warning("History insert failed: %s", e)

    del result["features_used"]
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
        {"score": e["score"], "risk_tier": e["risk_tier"],
         "date": e["created_at"].isoformat()} for e in events
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
        res  = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": messages, "max_tokens": 512},
            timeout=15,
        )
        data  = res.json()
        reply = data["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        log.error("Groq error: %s", e)
        return jsonify({"error": "Chat service unavailable"}), 503


# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "neoscore"})

@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    log.exception(e)
    return jsonify({"error": "Internal server error"}), 500

def debug_paths():
    paths = {
        "thresholds": THRESHOLDS_PATH,
        "student_A": STUDENT_A_PATH,
        "student_B": STUDENT_B_PATH,
        "explainer_A": EXPLAINER_A_PATH,
        "explainer_B": EXPLAINER_B_PATH,
        "encoders": ENCODERS_PATH,
    }

    print("\n=== FILE CHECK ===")
    for name, path in paths.items():
        print(f"{name}: {path} -> {os.path.exists(path)}")
    print("==================\n")

debug_paths()

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=not IS_PROD)