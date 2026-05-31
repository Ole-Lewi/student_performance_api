# 📊 Student Performance Predictor API

**A machine learning model that predicts a student's exam score based on hours studied, served as a deployed REST API.**

This was an early ML-to-production project — the goal wasn't to build a complex model, but to practice the full pipeline: train a model, evaluate it, serialize it, and expose it as a real API endpoint deployed to the cloud.

---

## The Full Pipeline

```
Training Data (Hours → Scores)
        │
        ▼
  Linear Regression (scikit-learn)
  - Train/test split (80/20)
  - Evaluate: R² score + MSE
  - Print coefficients and intercept
        │
        ▼
  joblib serialization
  → Linear Regression.pkl
        │
        ▼
  FastAPI inference server (api.py)
  GET  /         → health check
  POST /predict  → { predicted_score: float }
        │
        ▼
  Deployed on Render (render.yaml)
```

---

## API Endpoints

**Health check**
```
GET /
→ { "message": "Welcome to the Student Performance Predictor API" }
```

**Predict score**
```
POST /predict
Body: { "hours": 7.5 }
→ { "studied_hours": 7.5, "predicted_score": 82.5 }
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Feature | Hours studied (float) |
| Target | Exam score (0–100) |
| Training samples | 8 (80% of 10) |
| Test samples | 2 (20% of 10) |
| Evaluation metrics | R² score, Mean Squared Error |

The dataset is small and intentionally simple — the focus here is the end-to-end workflow, not model complexity.

---

## Project Structure

```
student_performance_api/
├── Linear Regression.py    # Training script — fit, evaluate, serialize
├── api.py                  # FastAPI server — load model, serve predictions
├── Linear Regression.pkl   # Serialized trained model
├── render.yaml             # Render deployment config
└── requirements.txt        # Dependencies
```

---

## Running Locally

**1. Clone and install**
```bash
git clone https://github.com/Ole-Lewi/student_performance_api.git
cd student_performance_api
pip install -r requirements.txt
```

**2. Train and save the model**
```bash
python "Linear Regression.py"
```

**3. Start the API**
```bash
uvicorn api:app --reload
```

**4. Test the endpoint**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"hours": 6}'
```

---

## What This Project Demonstrates

Even as an early project, the patterns here matter:

- **Train/serve separation** — the model is trained once and serialized; the API loads the artifact, it doesn't retrain on every request
- **Pydantic input validation** — the `StudyHours` model ensures the API rejects malformed requests before they reach the model
- **Root endpoint** — a health check route is standard practice for any deployed service
- **Render deployment** — the model is live, not just running locally

These same patterns appear in the more complex projects in this portfolio — NLP clustering API, RAG chatbot — just at greater scale.

---

## Author

**Lewis Miano (Lincoln)**
ALX Backend Web Dev · ML/NLP · Agentic AI Systems

[GitHub](https://github.com/Ole-Lewi) · [Portfolio Bot](https://professional-portfolio-5.onrender.com) · [NLP Review Analyzer](https://github.com/Ole-Lewi/NLP-Review-Analyzer--Sentiment-Cluster)