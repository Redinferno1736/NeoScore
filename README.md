#  NeoScore

### Explainable AI Credit Scoring

NeoScore is a full-stack credit scoring platform designed for the Indian lending ecosystem, especially targeting **thin-file users** — individuals with little or no credit history.

It combines a **knowledge-distillation ML pipeline**, a **real-time scoring API**, and a **Groq-powered AI financial coach** to deliver transparent, explainable, and actionable credit insights.

---

## ✨ Key Features

* 📊 **Credit Score (300–900)**
  Industry-style scoring mapped to population percentiles

* 🧠 **Explainable AI (SHAP + LLM)**
  Understand exactly *why* a score was assigned

* 🤖 **AI Financial Coach**
  Converts model output into human-friendly advice

* 🔄 **What-If Simulator**
  Adjust inputs (income, DTI, savings) and see live score changes

* 📈 **Counterfactual Recommendations**
  “Do this → gain +X points” actionable insights

* 👤 **Persona-Based Demo**
  Prebuilt profiles for quick testing

* 🔐 **Google OAuth + History**
  Save and track past credit evaluations

---

## 📦 Pretrained Models

All required trained models and artifacts (teacher model, student model, calibration files, etc.) are available here:

🔗 https://drive.google.com/drive/folders/1ouQRWfIVffsxAdwKYrOwVtDnvFfOJS5q?usp=sharing

### How to Use

1. Download all files from the Drive folder  
2. Place them inside the backend artifacts directory:

```bash
backend/artifacts/
```

3. Ensure filenames match those expected in your code (e.g., `student_model.pkl`, `calibrator.pkl`, etc.)

---

## 🛠️ Tech Stack

| Layer               | Technologies                                        |
| ------------------- | --------------------------------------------------- |
| **Frontend**        | Next.js 14 · Zustand · Tailwind CSS · Framer Motion |
| **Backend**         | Flask · MongoDB · Google OAuth · Groq API           |
| **ML Pipeline**     | XGBoost (Teacher → Student) · SHAP                  |
| **Score Range**     | 300–900                                             |
| **Core Innovation** | Thin-file scoring using knowledge distillation      |

---

## 🏗️ Architecture

```text
User Input
   ↓
Frontend (Next.js + Zustand)
   ↓
Flask API (/api/score)
   ↓
Feature Engineering
   ↓
Student Model (XGBoost)
   ↓
SHAP (Top Features)
   ↓
Groq LLM (AI Explanation)
   ↓
Frontend (Score + Insights)
```

---

## 🧠 ML Pipeline

### 1. Data Pipeline

* Combines 5 Home Credit datasets
* Engineers 30+ features
* Creates composite scores:

  * Financial Pressure
  * Stability Score
  * Asset Score
  * Income Adequacy

---

### 2. Teacher Model

* XGBoost classifier trained on **thick-file users**
* Uses **isotonic calibration** for probability accuracy
* Generates **soft labels** for all users

---

### 3. Student Model (Distillation)

* Learns from soft labels (not raw targets)
* Works for **thin-file users**
* Uses monotonic constraints (financial logic)

---

### 4. Scoring Formula

```text
Score = Percentile × 600 + 300
```

---

### 5. Explainability

* SHAP → identifies top contributing features
* LLM → converts into human-readable insights
* Filters out:

  * Age
  * Region
  * Sensitive attributes

---

## 🔌 Backend API

| Endpoint              | Method | Description             |
| --------------------- | ------ | ----------------------- |
| `/api/score`          | POST   | Get score + explanation |
| `/api/counterfactual` | POST   | Suggest improvements    |
| `/api/simulate`       | POST   | What-if score changes   |
| `/api/features`       | GET    | Feature metadata        |
| `/chat`               | POST   | AI assistant            |
| `/auth/login/google`  | GET    | OAuth login             |

---

## 💻 Frontend Routes

| Route       | Purpose                          |
| ----------- | -------------------------------- |
| `/home`     | Persona selection + manual input |
| `/results`  | Score + insights + simulator     |
| `/history`  | Score history                    |
| `/loans`    | Loan recommendations             |
| `/api-docs` | API reference                    |

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/neoscore.git
cd neoscore
```

---

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_key_here
MONGO_URI=your_mongo_uri
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_secret
```

Run server:

```bash
python main.py
```

---

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Example API Call

```bash
curl -X POST http://localhost:5000/api/score \
-H "Content-Type: application/json" \
-d '{
  "features": {
    "AMT_INCOME_TOTAL": 500000,
    "AMT_CREDIT": 200000,
    "EMPLOYED_YEARS": 5
  }
}'
```

---

## 📊 Sample Response

```json
{
  "score": 398,
  "risk_tier": "Very Poor",
  "ai_explanation": "Here’s what’s going on with your score...",
  "top_features": [...]
}
```

---

## 🎯 Problem Solved

Traditional credit scoring:

* Requires credit history
* Not transparent
* Not actionable

NeoScore:

* Works for first-time borrowers
* Fully explainable
* Provides clear improvement steps

---

## 🚀 Future Improvements

* 📱 Mobile app
* 🏦 Banking API integration
* 📊 Personalized financial roadmap
* 🤝 Loan matching system

---

⭐ If you like this project, consider giving it a star!
