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
MODEL_PATH        = os.getenv("STUDENT_MODEL_PATH", "student_model.joblib")
IS_PROD           = os.getenv("FLASK_ENV") == "production"

SCORE_MIN, SCORE_MAX = 300, 900

# ─── Feature list — must match student model training exactly ─────────────────
# From backend/feature_sets.json → "student" key (Notebook 1, version 2)
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

# Categorical label-encoding maps (must match LabelEncoder fit order from notebook)
# These are ordinal integers — the model was trained with LabelEncoder so we keep same mapping.
# At inference time the frontend sends integer codes directly (same as stored in DB after encoding).
# If you want to accept string values, extend the maps below.
EDUCATION_MAP    = {"Lower secondary": 0, "Secondary / secondary special": 1,
                    "Incomplete higher": 2, "Higher education": 3, "Academic degree": 4, "Unknown": 1}
FAMILY_MAP       = {"Civil marriage": 0, "Married": 1, "Separated": 2,
                    "Single / not married": 3, "Widow": 4, "Unknown": 3}
INCOME_TYPE_MAP  = {"Businessman": 0, "Commercial associate": 1, "Maternity leave": 2,
                    "Pensioner": 3, "State servant": 4, "Student": 5,
                    "Unemployed": 6, "Working": 7, "Unknown": 7}
HOUSING_MAP      = {"Co-op apartment": 0, "House / apartment": 1, "Municipal apartment": 2,
                    "Office apartment": 3, "Rented apartment": 4, "With parents": 5, "Unknown": 1}

# Numerical defaults for missing fields (median-ish values from dataset)
FEATURE_DEFAULTS = {
    "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2,
    "NAME_EDUCATION_TYPE": 1, "NAME_FAMILY_STATUS": 1, "NAME_INCOME_TYPE": 7,
    "OCCUPATION_TYPE": 8, "ORGANIZATION_TYPE": 16, "NAME_HOUSING_TYPE": 1,
    "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 1,
    "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000, "AMT_GOODS_PRICE": 450000,
    "REGION_POPULATION_RELATIVE": 0.02, "REGION_RATING_CLIENT": 2,
    "AGE_YEARS": 35, "EMPLOYED_YEARS": 5,
    "DEBT_TO_INCOME": 2.0, "ANNUITY_TO_INCOME": 0.1,
    "CREDIT_TO_GOODS": 1.1, "INCOME_PER_PERSON": 60000,
    "CHILDREN_RATIO": 0.0, "EMI_BURDEN": 0.1,
    "INCOME_STABILITY": 0.14, "LOAN_TO_INCOME": 0.23,
    "FINANCIAL_PRESSURE": 0.2, "STABILITY_SCORE": 0.25,
    "ASSET_SCORE": 0.6, "INCOME_ADEQUACY": 0.6,
}

# ─── Demo personas (loaded from demo_personas.json or inline) ─────────────────
DEMO_PERSONAS = {
    "ravi": {
        "label": "Ravi — Gig worker, thin file",
        "features": {
            "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3,
            "NAME_EDUCATION_TYPE": 1, "NAME_FAMILY_STATUS": 1,
            "NAME_INCOME_TYPE": 7, "OCCUPATION_TYPE": 8,
            "ORGANIZATION_TYPE": 16, "NAME_HOUSING_TYPE": 1,
            "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 0,
            "AMT_INCOME_TOTAL": 120000, "AMT_CREDIT": 100000,
            "AMT_ANNUITY": 7000, "AMT_GOODS_PRICE": 90000,
            "REGION_POPULATION_RELATIVE": 0.018, "REGION_RATING_CLIENT": 2,
            "AGE_YEARS": 28, "EMPLOYED_YEARS": 0.8,
            "DEBT_TO_INCOME": 0.83, "ANNUITY_TO_INCOME": 0.06,
            "CREDIT_TO_GOODS": 1.11, "INCOME_PER_PERSON": 40000,
            "CHILDREN_RATIO": 0.33, "EMI_BURDEN": 0.058,
            "INCOME_STABILITY": 0.028, "LOAN_TO_INCOME": 0.069,
            "FINANCIAL_PRESSURE": 0.31, "STABILITY_SCORE": 0.11,
            "ASSET_SCORE": 0.0, "INCOME_ADEQUACY": 0.48,
        }
    },
    "priya": {
        "label": "Priya — Salaried professional",
        "features": {
            "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2,
            "NAME_EDUCATION_TYPE": 3, "NAME_FAMILY_STATUS": 3,
            "NAME_INCOME_TYPE": 1, "OCCUPATION_TYPE": 11,
            "ORGANIZATION_TYPE": 7, "NAME_HOUSING_TYPE": 1,
            "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 1,
            "AMT_INCOME_TOTAL": 300000, "AMT_CREDIT": 250000,
            "AMT_ANNUITY": 15000, "AMT_GOODS_PRICE": 230000,
            "REGION_POPULATION_RELATIVE": 0.035, "REGION_RATING_CLIENT": 2,
            "AGE_YEARS": 32, "EMPLOYED_YEARS": 5,
            "DEBT_TO_INCOME": 0.83, "ANNUITY_TO_INCOME": 0.05,
            "CREDIT_TO_GOODS": 1.09, "INCOME_PER_PERSON": 150000,
            "CHILDREN_RATIO": 0.0, "EMI_BURDEN": 0.05,
            "INCOME_STABILITY": 0.156, "LOAN_TO_INCOME": 0.069,
            "FINANCIAL_PRESSURE": 0.14, "STABILITY_SCORE": 0.45,
            "ASSET_SCORE": 0.6, "INCOME_ADEQUACY": 0.83,
        }
    },
    "deepa": {
        "label": "Deepa — Self-employed, moderate risk",
        "features": {
            "CNT_CHILDREN": 2, "CNT_FAM_MEMBERS": 4,
            "NAME_EDUCATION_TYPE": 3, "NAME_FAMILY_STATUS": 1,
            "NAME_INCOME_TYPE": 7, "OCCUPATION_TYPE": 8,
            "ORGANIZATION_TYPE": 22, "NAME_HOUSING_TYPE": 1,
            "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 1,
            "AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 400000,
            "AMT_ANNUITY": 16000, "AMT_GOODS_PRICE": 380000,
            "REGION_POPULATION_RELATIVE": 0.025, "REGION_RATING_CLIENT": 2,
            "AGE_YEARS": 38, "EMPLOYED_YEARS": 2.5,
            "DEBT_TO_INCOME": 2.0, "ANNUITY_TO_INCOME": 0.08,
            "CREDIT_TO_GOODS": 1.05, "INCOME_PER_PERSON": 50000,
            "CHILDREN_RATIO": 0.5, "EMI_BURDEN": 0.08,
            "INCOME_STABILITY": 0.065, "LOAN_TO_INCOME": 0.167,
            "FINANCIAL_PRESSURE": 0.27, "STABILITY_SCORE": 0.28,
            "ASSET_SCORE": 0.6, "INCOME_ADEQUACY": 0.52,
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

# ─── Scoring Model ────────────────────────────────────────────────────────────
class MockModel:
    """Fallback when no real model is available."""
    def predict_proba(self, X):
        return np.array([[1 - 0.15, 0.15]])

print("Loading ML artifacts...")
try:
    # This loads ALL your files into memory at startup
    _model = joblib.load("student_A.pkl")
    _explainer = joblib.load("explainer_A.pkl")
    _label_encoders = joblib.load("label_encoders.pkl") 
    _score_dist = np.load("population_score_distribution.npy")
    log.info("Successfully loaded real model, explainer, encoders, and dist!")
except Exception as e:
    log.warning(f"Could not load ML artifacts ({e}) — using mock fallbacks.")
    _model = MockModel()
    _explainer = None
    _label_encoders = {} # This prevents the NameError if loading fails!
    _score_dist = np.array([])

def get_model(): return _model

# ─── Scoring helpers ──────────────────────────────────────────────────────────
def prob_to_score(pd_: float) -> int:
    """Map default probability to 300-900 score (lower PD = higher score)."""
    clamped = max(0.001, min(0.999, pd_))
    score = SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (1 - clamped)
    return int(round(score))

def risk_tier(score: int) -> str:
    if score >= 750: return "Excellent"
    if score >= 700: return "Good"
    if score >= 650: return "Fair"
    if score >= 600: return "Poor"
    return "Very Poor"

def build_vector(features: dict) -> list:
    """Build feature vector in STUDENT_FEATURES order, filling defaults."""
    vec = []
    for f in STUDENT_FEATURES:
        val = features.get(f, FEATURE_DEFAULTS.get(f, 0))
        vec.append(float(val))
    return vec

def derive_computed_features(features: dict) -> dict:
    """Auto-compute composite/ratio features from raw inputs if not provided."""
    f = {**FEATURE_DEFAULTS, **features}
    income = float(f.get("AMT_INCOME_TOTAL", 180000))
    credit = float(f.get("AMT_CREDIT", 500000))
    annuity = float(f.get("AMT_ANNUITY", 25000))
    goods   = float(f.get("AMT_GOODS_PRICE", 450000))
    emp_yrs = float(f.get("EMPLOYED_YEARS", 5))
    age_yrs = float(f.get("AGE_YEARS", 35))
    children = float(f.get("CNT_CHILDREN", 0))
    fam     = float(f.get("CNT_FAM_MEMBERS", 2))
    own_car = float(f.get("FLAG_OWN_CAR", 0))
    own_re  = float(f.get("FLAG_OWN_REALTY", 0))

    # Derived ratios
    if "DEBT_TO_INCOME"    not in features: f["DEBT_TO_INCOME"]    = credit / (income + 1)
    if "ANNUITY_TO_INCOME" not in features: f["ANNUITY_TO_INCOME"] = annuity / (income + 1)
    if "CREDIT_TO_GOODS"   not in features: f["CREDIT_TO_GOODS"]   = credit / (goods + 1)
    if "INCOME_PER_PERSON" not in features: f["INCOME_PER_PERSON"] = income / (fam + 1)
    if "CHILDREN_RATIO"    not in features: f["CHILDREN_RATIO"]    = children / (fam + 1)
    if "EMI_BURDEN"        not in features: f["EMI_BURDEN"]        = annuity / (income + 1)
    if "INCOME_STABILITY"  not in features: f["INCOME_STABILITY"]  = emp_yrs / (age_yrs + 1)
    if "LOAN_TO_INCOME"    not in features: f["LOAN_TO_INCOME"]    = credit / (income * 12 + 1)

    # Composite scores
    emi_burden   = f.get("EMI_BURDEN", annuity / (income + 1))
    loan_income  = f.get("LOAN_TO_INCOME", credit / (income * 12 + 1))
    child_ratio  = f.get("CHILDREN_RATIO", children / (fam + 1))
    inc_stab     = f.get("INCOME_STABILITY", emp_yrs / (age_yrs + 1))

    if "FINANCIAL_PRESSURE" not in features:
        f["FINANCIAL_PRESSURE"] = (
            0.40 * min(emi_burden, 1) +
            0.35 * min(loan_income, 1) +
            0.25 * min(child_ratio, 1)
        )
    if "STABILITY_SCORE" not in features:
        f["STABILITY_SCORE"] = (
            0.50 * min(inc_stab, 1) +
            0.30 * min(emp_yrs / 40, 1) +
            0.20 * own_re
        )
    if "ASSET_SCORE" not in features:
        f["ASSET_SCORE"] = 0.60 * own_re + 0.40 * own_car
    if "INCOME_ADEQUACY" not in features:
        f["INCOME_ADEQUACY"] = min(income / (annuity * 12 + 1), 10)

    return f

import pandas as pd # Make sure this is at the very top of your file!

def score_features(features: dict) -> dict:
    # 1. Compute ratios and fill defaults
    features = derive_computed_features(features)
    
    # 2. CATEGORICAL ENCODING (This fixes the 'Bachelor' error!)
    for col, encoder in _label_encoders.items():
        if col in features:
            val = str(features[col])
            try:
                # Translates "Bachelor" into a number the model understands
                features[col] = encoder.transform([val])[0]
            except ValueError:
                features[col] = -1 # Fallback for unknown categories
                
    # 3. Build DataFrame strictly ordered by STUDENT_FEATURES
    # This completely replaces the old 'build_vector' function
    df_input = pd.DataFrame([features], columns=STUDENT_FEATURES).fillna(0)
    
    # 4. Inference
    model = get_model()
    proba = model.predict_proba(df_input)[0]
    pd_   = float(proba[1])
    score = prob_to_score(pd_)
    
    # 5. Calculate Percentile
    percentile = 0.0
    if len(_score_dist) > 0:
        percentile = float((_score_dist < score).mean() * 100)
        
    # 6. SHAP Explanations
    top_features = []
    reasoning = "Score calculated."
    if _explainer:
        try:
            shap_vals = _explainer.shap_values(df_input)
            local_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            
            impacts = [{"feature": f, "impact": float(val)} for f, val in zip(STUDENT_FEATURES, local_shap)]
            impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
            top_features = impacts[:5]
            
            top_driver = top_features[0]
            if top_driver["impact"] < 0:
                reasoning = f"Your score is boosted by your {top_driver['feature'].replace('_', ' ').lower()}."
            else:
                reasoning = f"Your score is held back slightly by your {top_driver['feature'].replace('_', ' ').lower()}."
        except Exception as e:
            log.warning(f"SHAP generation failed: {e}")

    return {
        "score": score,
        "risk_tier": risk_tier(score),
        "default_probability": round(pd_, 4),
        "approval_probability": round(1 - pd_, 4),
        "percentile": round(percentile, 1),
        "top_features": top_features,
        "reasoning": reasoning,
        "features_used": features, 
    }

# ─── Counterfactual Engine ────────────────────────────────────────────────────
# Which features can be improved and how
CF_ACTIONS = [
    {"feature": "EMPLOYED_YEARS",       "label": "Job tenure",            "delta_pct": 0.50, "direction": 1,  "effort": "low",    "days": 90},
    {"feature": "AMT_INCOME_TOTAL",     "label": "Annual income",         "delta_pct": 0.20, "direction": 1,  "effort": "medium", "days": 90},
    {"feature": "AMT_CREDIT",           "label": "Loan amount requested", "delta_pct": -0.20,"direction": -1, "effort": "low",    "days": 0},
    {"feature": "CNT_CHILDREN",         "label": "Dependants",            "delta_val": -1,   "direction": -1, "effort": "n/a",   "days": 0},
    {"feature": "FLAG_OWN_REALTY",      "label": "Property ownership",    "delta_val": 1,    "direction": 1,  "effort": "high",   "days": 365},
    {"feature": "REGION_RATING_CLIENT", "label": "Region credit rating",  "delta_val": -1,   "direction": -1, "effort": "high",   "days": 180},
]

def counterfactual(features: dict, locked: list = None, max_steps: int = 3) -> dict:
    locked = set(locked or [])
    features = derive_computed_features(features)
    original = score_features(features)
    moves = []

    for action in CF_ACTIONS:
        feat = action["feature"]
        if feat in locked or feat not in features:
            continue
        new_f = dict(features)
        orig_val = float(new_f.get(feat, 0))

        if "delta_pct" in action:
            new_val = orig_val * (1 + action["delta_pct"])
        else:
            new_val = orig_val + action["delta_val"]

        new_val = max(0, new_val)
        new_f[feat] = new_val
        new_result = score_features(new_f)
        delta = new_result["score"] - original["score"]

        if delta > 0:
            moves.append({
                "feature":      feat,
                "label":        action["label"],
                "original_val": orig_val,
                "new_val":      new_val,
                "score_delta":  delta,
                "new_score":    new_result["score"],
                "effort":       action["effort"],
                "timeline_days": action["days"],
            })

    moves.sort(key=lambda x: x["score_delta"], reverse=True)
    moves = moves[:max_steps]

    return {
        "original_score":       original["score"],
        "original_risk":        original["risk_tier"],
        "original_probability": original["default_probability"],
        "moves":                moves,
        "best_reachable_score": moves[0]["new_score"] if moves else original["score"],
    }

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SECRET_KEY              = SECRET_KEY,
    SESSION_TYPE            = "filesystem",
    SESSION_FILE_DIR        = os.path.join(os.getcwd(), "session_data"), # ADD THIS LINE!
    SESSION_COOKIE_SAMESITE = "Lax",
    SESSION_COOKIE_SECURE   = IS_PROD,
)
Session(app)
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000"], 
     supports_credentials=True,
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
    session["user_id"] = str(user["_id"])
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

# ─── Public API — no auth required ───────────────────────────────────────────
# This is the "API wrapper" that external apps can call with feature JSON

@app.route("/api/score", methods=["POST"])
def api_score():
    """
    Public scoring endpoint. Pass any subset of features; missing ones use defaults.
    Returns: score, risk_tier, default_probability, approval_probability
    Optionally pass "persona": "ravi"|"priya"|"deepa" to use demo data.
    """
    body = request.get_json(silent=True) or {}

    # Support demo persona shorthand
    if "persona" in body:
        persona = DEMO_PERSONAS.get(body["persona"])
        if not persona:
            return jsonify({"error": "Unknown persona. Use: ravi, priya, deepa"}), 400
        features = persona["features"]
    else:
        features = body.get("features", body)  # allow flat or nested

    result = score_features(features)
    # Strip internal features_used from public response to keep it clean
    del result["features_used"]
    return jsonify(result)

@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    """
    Returns actionable steps to improve score.
    Body: { features: {...}, locked_features: [...], max_steps: 3 }
    """
    body     = request.get_json(silent=True) or {}
    if "persona" in body:
        persona = DEMO_PERSONAS.get(body["persona"])
        features = persona["features"] if persona else {}
    else:
        features = body.get("features", {})

    locked    = body.get("locked_features", [])
    max_steps = int(body.get("max_steps", 3))
    return jsonify(counterfactual(features, locked, max_steps))

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
        "new_score":            new["score"],
        "new_risk":             new["risk_tier"],
        "new_probability":      new["default_probability"],
        "score_delta":          new["score"] - base["score"],
        "changes_applied":      changes,
    })

@app.route("/api/features", methods=["GET"])
def api_features():
    """Returns feature list, defaults, and demo personas."""
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
        persona = DEMO_PERSONAS.get(body["persona"])
        features = persona["features"] if persona else {}
    else:
        features = body.get("features", {})

    result = score_features(features)
    # Persist to history
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
    body     = request.get_json(silent=True) or {}
    features = body.get("features", {})
    locked   = body.get("locked_features", [])
    result   = counterfactual(features, locked, int(body.get("max_steps", 3)))
    return jsonify(result)

@app.route("/score/history", methods=["GET"])
@require_auth
def score_history():
    limit = min(int(request.args.get("limit", 20)), 100)
    events = list(db().score_history.find(
        {"user_id": uid()},
        {"features": 0},  # omit heavy features field
        sort=[("created_at", DESCENDING)],
        limit=limit,
    ))
    return jsonify({"events": [ser(e) for e in events]})

@app.route("/score/history/trend", methods=["GET"])
@require_auth
def score_trend():
    limit = min(int(request.args.get("limit", 30)), 100)
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
"""

@app.route("/chat", methods=["POST"])
def chat():
    body    = request.get_json(silent=True) or {}
    message = body.get("message", "").strip()
    history = body.get("history", [])  # [{role, content}]
    context = body.get("score_context", {})  # optional score data to include

    if not message:
        return jsonify({"error": "message required"}), 400
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 503

    system = SYSTEM_PROMPT
    if context:
        system += f"\n\nUser's current NeoScore data: {context}"

    messages = [{"role": "system", "content": system}]
    for h in history[-10:]:  # keep last 10 turns
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        res = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": messages, "max_tokens": 512},
            timeout=15,
        )
        data = res.json()
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

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=not IS_PROD)
