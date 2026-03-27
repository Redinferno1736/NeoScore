# [cite_start]NeoScore [cite: 1]
### [cite_start]Explainable AI Credit Scoring — System Report [cite: 2]

[cite_start]**NeoScore** is a full-stack credit scoring platform for the Indian lending market, focused on thin-file users — applicants with little or no credit bureau history. [cite: 5] [cite_start]It combines a knowledge-distillation ML pipeline with an interactive dashboard, REST API, and Groq-powered AI financial coach. [cite: 6]

---

## 🛠️ Tech Stack at a Glance

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | [cite_start]Next.js 14 · Zustand · Tailwind · framer-motion [cite: 7] |
| **Backend** | [cite_start]Flask · MongoDB · Google OAuth · Groq (llama3-8b) · CORS [cite: 7] |
| **ML Pipeline** | [cite_start]XGBoost Teacher → Student (knowledge distillation) · SHAP explainability [cite: 7] |
| **Score Range** | [cite_start]300–900 (population-percentile mapped, India-aligned) [cite: 7] |
| **Key Innovation** | [cite_start]Thin-file scoring via soft-label distillation — no bureau history needed [cite: 7] |

---

## 🏗️ Architecture & Data Flow

[cite_start]The pipeline moves from raw data → trained artifacts → live API → user interface: [cite: 9]

1.  [cite_start]**Home Credit Dataset**: Utilizes 5 CSV files (`application_train`, `bureau`, `previous_application`, `installments`, `credit_card_balance`). [cite: 10]
2.  [cite_start]**Notebook 1 (Data Pipeline)**: Performs feature engineering, creates composite scores, and outputs `thick_file_dataset.parquet`. [cite: 10]
3.  [cite_start]**Notebook 2 (Teacher Model)**: Trains on thick-file users only and generates calibrated soft labels (`SOFT_LABEL`) for all users. [cite: 10]
4.  [cite_start]**Notebook 3 (Student Models A & B)**: Trains on 30 static features guided by soft labels via knowledge distillation. [cite: 10]
5.  [cite_start]**Artifacts Saved**: Saves `.pkl` files for models, explainers, and encoders, along with `thresholds.json`. [cite: 10]
6.  [cite_start]**Flask Backend (Inference)**: Loads artifacts at startup and handles `POST /api/score` to serve JSON responses. [cite: 10]
7.  [cite_start]**Next.js Frontend**: Consumes the API to display the persona selector, results page, score gauge, and AI coach. [cite: 10]

---

## [cite_start]🧠 ML Pipeline — Key Design Decisions [cite: 11]

### [cite_start]Data Engineering (Notebook 1) [cite: 12]
* [cite_start]Merges 5 Home Credit files. [cite: 13]
* [cite_start]Engineers 30 student features including four composite scores (`FINANCIAL_PRESSURE`, `STABILITY_SCORE`, `ASSET_SCORE`, `INCOME_ADEQUACY`). [cite: 13]
* [cite_start]Classifies users as thin-file (`IS_THIN_FILE`) based on the absence of bureau/installment/previous-application records. [cite: 13]

### [cite_start]Teacher Model (Notebook 2) [cite: 14]
* [cite_start]An XGBoost classifier trained only on thick-file users with a three-way split (train / calibration / validation). [cite: 15]
* [cite_start]Isotonic regression calibration prevents probability overconfidence. [cite: 16]
* [cite_start]The Teacher scores all users — including thin-file — producing `SOFT_LABEL` targets for distillation. [cite: 16]
* [cite_start]Monotonic constraints enforce economic logic (e.g. higher DTI must increase risk). [cite: 17]

### [cite_start]Student Models + Full System (Notebook 3) [cite: 18]
* [cite_start]**Dual-mode**: Uses two students routed by `USER_MODE`: Model A (near-thin + thick) and Model B (pure-thin). [cite: 19]
* [cite_start]**Distillation**: An XGBoost Regressor trained on `SOFT_LABEL` with the same monotonic constraints as the Teacher. [cite: 20]
* [cite_start]**Thresholds**: Dynamic APPROVE / REVIEW / REJECT thresholds at the 60th / 90th percentile of the training distribution. [cite: 21]
* [cite_start]**Rules**: A post-ML rule overlay automatically rejects if EMI > 60% income, and downgrades to REVIEW if confidence < 0.30. [cite: 22]
* [cite_start]**SHAP**: A SHAP TreeExplainer per model provides the top-5 features returned with each score using human-readable templates. [cite: 23]
* [cite_start]**Scoring Math**: Score = population percentile × 600 + 300 (ranges 300–900). [cite: 24]

---

## [cite_start]🔌 Backend API (Flask) [cite: 25]

[cite_start]The backend is a single-file server (`main.py`). [cite: 26] [cite_start]Artifacts are loaded once at startup. [cite: 26] [cite_start]Public endpoints require no auth, while history endpoints require a Google OAuth session. [cite: 26] [cite_start]At inference, `derive_computed_features()` recomputes all 12 ratio/composite features from raw inputs before model prediction. [cite: 28]

| Endpoint | Auth | Purpose |
| :--- | :--- | :--- |
| `POST /api/score` | No | [cite_start]Score any feature dict; supports persona shorthand [cite: 27] |
| `POST /api/counterfactual` | No | [cite_start]Top-N greedy actions to improve score [cite: 27] |
| `POST /api/simulate` | No | [cite_start]What-if: base features + changes → score delta [cite: 27] |
| `GET /api/features` | No | [cite_start]Feature list, defaults, demo personas [cite: 27] |
| `POST /score/predict` | Yes | [cite_start]Score + persist to MongoDB history [cite: 27] |
| `GET /score/history` | Yes | [cite_start]Past score events for authenticated user [cite: 27] |
| `POST /chat` | No | [cite_start]Groq AI coach (llama3-8b, 10-turn context) [cite: 27] |
| `GET /auth/login/google` | No | [cite_start]OAuth 2.0 flow with CSRF state token [cite: 27] |

---

## [cite_start]💻 Frontend (Next.js 14) [cite: 29]

| Route | Purpose |
| :--- | :--- |
| `/home` | [cite_start]Persona selector (5 presets) + manual input form → calls `POST /api/score` [cite: 30] |
| `/results` | [cite_start]Score gauge · SHAP score drivers · what-if sliders (live re-score) · AI coach · recommendations [cite: 30] |
| `/history` | [cite_start]Chronological score history with delete/clear [cite: 30] |
| `/loans` | [cite_start]Eligibility-matched loan product cards [cite: 30] |
| `/api-docs` | [cite_start]Developer reference with JS/Python code snippets [cite: 30] |
