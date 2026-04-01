# 🔮 Customer Churn Prediction — End-to-End MLOps Pipeline

> Predict which customers are likely to churn, with a fully automated ML pipeline from data ingestion to cloud deployment.

---

## 🏗️ Architecture

```
Raw Data → EDA → Preprocessing → Feature Engineering
       → Model Training (MLflow) → Evaluation (SHAP)
       → FastAPI → Docker → GitHub Actions CI/CD → AWS EC2
```

---

## 🛠️ Tech Stack

| Layer               | Tools                                           |
|---------------------|-------------------------------------------------|
| Data & ML           | Python, Pandas, Scikit-learn, Imbalanced-learn  |
| Experiment Tracking | MLflow, DagsHub                                 |
| Data Versioning     | DVC                                             |
| Explainability      | SHAP                                            |
| API                 | FastAPI, Uvicorn, Pydantic                      |
| Containerization    | Docker, Docker Compose                          |
| CI/CD               | GitHub Actions                                  |
| Cloud               | AWS EC2, DockerHub                              |

---

## 📁 Folder Structure

```
customer-churn-mlops/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training_MLflow.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_FastAPI_Deployment.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   └── main.py
├── tests/
│   └── test_api.py
├── models/                 (DVC tracked)
├── data/raw/               (DVC tracked)
├── data/processed/
├── reports/
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/customer-churn-mlops.git
cd customer-churn-mlops
pip install -r requirements.txt

# 2. Run notebooks 01 → 05 in order

# 3. Launch API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs → http://localhost:8000/docs

# 4. Run tests
pytest tests/ -v

# 5. Docker
docker-compose up
# API    → http://localhost:8000/docs
# MLflow → http://localhost:5000
```

---

## 📊 Model Performance

| Model               | F1    | ROC-AUC |
|---------------------|-------|---------|
| Logistic Regression | ~0.60 | ~0.78   |
| Random Forest       | ~0.65 | ~0.84   |
| Gradient Boosting   | ~0.67 | ~0.86   |

---

## 🔌 API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":1,"SeniorCitizen":0,"Partner":0,"Dependents":0,
       "tenure":2,"PhoneService":1,"PaperlessBilling":1,
       "MonthlyCharges":99.9,"TotalCharges":199.8}'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8732,
  "risk_level": "HIGH",
  "recommendation": "Immediate retention action — offer discount or loyalty bonus."
}
```

---

## ⚙️ CI/CD (GitHub Actions)

Push to `main` → Lint + Tests → Build Docker → Push DockerHub → Deploy AWS EC2

**Required Secrets:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`

---

## 👤 Author
**Your Name** | AI/ML Engineer
Update with your name, email, LinkedIn and GitHub links!
