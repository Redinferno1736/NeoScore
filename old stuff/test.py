import pandas as pd
import numpy as np
import joblib
import json

from sklearn.metrics import precision_score, recall_score, roc_auc_score

# =========================
# LOAD ARTIFACTS
# =========================
model = joblib.load('./artifacts/model_full.pkl')
imputer = joblib.load('./artifacts/imputer.pkl')
label_encoders = joblib.load('./artifacts/label_encoders.pkl')

with open('./artifacts/feature_cols.json') as f:
    FEATURE_COLS = json.load(f)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('./data/application_train.csv')

# =========================
# FEATURE ENGINEERING (MATCH TRAINING)
# =========================
df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365.25
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['EMPLOYMENT_YEARS'] = np.abs(df['DAYS_EMPLOYED']) / 365.25

# Binary flags
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})

# Ratios
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / ((df['AMT_INCOME_TOTAL'] / 12) + 1)
df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
df['CREDIT_TERM'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
df['DOWN_PAYMENT_RATIO'] = (df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']) / (df['AMT_GOODS_PRICE'] + 1)

# Custom features
df['ASSET_SCORE'] = df['FLAG_OWN_CAR'].fillna(0) + df['FLAG_OWN_REALTY'].fillna(0)
df['CHILDREN_INCOME_RATIO'] = df['CNT_CHILDREN'] / ((df['AMT_INCOME_TOTAL'] / 12) + 1)
df['EMI_PER_EMPLOYMENT_YEAR'] = df['AMT_ANNUITY'] / (df['EMPLOYMENT_YEARS'] + 1)
df['INCOME_STABILITY'] = df['AMT_INCOME_TOTAL'] * df['EMPLOYMENT_YEARS']

# =========================
# ENCODE CATEGORICALS
# =========================
CAT_COLS = [
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE',
    'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
    'NAME_HOUSING_TYPE', 'NAME_CONTRACT_TYPE'
]

for col in CAT_COLS:
    df[col] = df[col].fillna('Unknown')
    le = label_encoders[col]
    
    df[col] = df[col].astype(str).map(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0
    )

# =========================
# BUILD FEATURE MATRIX
# =========================
X = df[FEATURE_COLS].copy()
X = pd.DataFrame(imputer.transform(X), columns=FEATURE_COLS)
y = df['TARGET']

# =========================
# PREDICT
# =========================
y_prob = model.predict_proba(X)[:, 1]

# =========================
# METRICS
# =========================
threshold = 0.15
y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
auc = roc_auc_score(y, y_prob)

# =========================
# CALIBRATION METRICS
# =========================
brier = np.mean((y_prob - y) ** 2)

# =========================
# PRECISION@K
# =========================
def precision_at_k(y_true, y_prob, k=0.1):
    cutoff = int(len(y_prob) * k)
    idx = np.argsort(y_prob)[::-1][:cutoff]
    return y_true.iloc[idx].mean()

p10 = precision_at_k(y, y_prob, 0.10)
p20 = precision_at_k(y, y_prob, 0.20)

# =========================
# DECISION BANDS
# =========================
def get_band(p):
    if p < 0.10:
        return "APPROVE"
    elif p < 0.25:
        return "REVIEW"
    else:
        return "REJECT"

df['band'] = [get_band(p) for p in y_prob]

band_stats = df.groupby('band')['TARGET'].agg(['count', 'mean'])

# =========================
# OUTPUT
# =========================
print("\n=== MODEL PERFORMANCE ===")
print(f"ROC-AUC: {auc:.4f}")
print(f"Brier Score: {brier:.4f}")

print(f"\nThreshold: {threshold}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

print(f"\nPrecision@10%: {p10:.3f}")
print(f"Precision@20%: {p20:.3f}")

print("\n=== DECISION BANDS ===")
for band in ['APPROVE', 'REVIEW', 'REJECT']:
    count = band_stats.loc[band, 'count']
    rate = band_stats.loc[band, 'mean']
    print(f"{band:8} → {count:6} users | default rate = {rate:.2%}")