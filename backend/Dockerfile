FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend entry point
COPY server.py .

# Config folder (thresholds.json)
COPY config/ ./config/

# ML artifacts — place these in the same folder before building
# or mount them at runtime with -v
COPY student_A.pkl ./student_A.pkl
COPY student_B.pkl ./student_B.pkl
COPY explainer_A.pkl ./explainer_A.pkl
COPY explainer_B.pkl ./explainer_B.pkl
COPY label_encoders.pkl ./label_encoders.pkl
COPY population_prob_distribution.npy ./population_prob_distribution.npy
COPY population_score_distribution.npy ./population_score_distribution.npy

RUN mkdir -p /app/session_data

EXPOSE 7860

ENV FLASK_ENV=production

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "server:app"]