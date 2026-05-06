# 🛡️ Fraud Detection MLOps Pipeline

> **Production-grade Credit Card Fraud Detection system with end-to-end MLOps on Azure ML Pipelines**

[![CI/CD](https://github.com/YOUR_USERNAME/fraud-detection-mlops/actions/workflows/main.yml/badge.svg)](https://github.com/YOUR_USERNAME/fraud-detection-mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧱 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FRAUD DETECTION MLOps ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────────────────────────┐   │
│  │          │    │          │    │           AZURE ML PIPELINE             │   │
│  │  KAGGLE  │───▶│  GITHUB  │───▶│                                         │   │
│  │ Dataset  │    │   Repo   │    │  ┌─────────┐  ┌──────────┐  ┌───────┐  │   │
│  │          │    │          │    │  │ Ingest  │─▶│Preprocess│─▶│ Train │  │   │
│  └──────────┘    └────┬─────┘    │  └─────────┘  └──────────┘  └───┬───┘  │   │
│                       │          │                                  │      │   │
│                       ▼          │  ┌──────────┐  ┌──────────┐     │      │   │
│                 ┌──────────┐     │  │ Register │◀─│ Evaluate │◀────┘      │   │
│                 │  GITHUB  │     │  │  Model   │  │  + Gate  │            │   │
│                 │ ACTIONS  │     │  └──────────┘  └──────────┘            │   │
│                 │  CI/CD   │     └─────────────────────────────────────────┘   │
│                 └────┬─────┘                                                   │
│                      │                                                         │
│                      ▼                                                         │
│                ┌───────────┐     ┌─────────────────────────────────────────┐   │
│                │  DOCKER   │     │            AKS CLUSTER                  │   │
│                │  BUILD &  │────▶│                                         │   │
│                │   PUSH    │     │  ┌─────────┐  ┌───────┐  ┌──────────┐  │   │
│                │  (ACR)    │     │  │ Pod (1) │  │Pod (2)│  │ Pod (N)  │  │   │
│                └───────────┘     │  │ FastAPI │  │FastAPI│  │ FastAPI  │  │   │
│                                  │  └────┬────┘  └───┬───┘  └────┬─────┘  │   │
│                                  │       └───────────┼───────────┘        │   │
│                                  │                   │ /predict           │   │
│                                  │                   │ /health            │   │
│                                  │                   │ /metrics           │   │
│                                  │  ┌────────────────┴────────────────┐   │   │
│                                  │  │    LoadBalancer Service         │   │   │
│                                  │  │    + HPA Autoscaling            │   │   │
│                                  │  └────────────────┬────────────────┘   │   │
│                                  └───────────────────┼────────────────────┘   │
│                                                      │                        │
│                                                      ▼                        │
│                                  ┌─────────────────────────────────────────┐   │
│                                  │           MONITORING STACK              │   │
│                                  │  ┌────────────┐    ┌───────────────┐    │   │
│                                  │  │ Prometheus │───▶│   Grafana     │    │   │
│                                  │  │  Scraping  │    │  Dashboards   │    │   │
│                                  │  └────────────┘    └───────────────┘    │   │
│                                  │  • Request rate     • Error rate        │   │
│                                  │  • Latency p50/95   • Fraud rate       │   │
│                                  │  • Model status     • Active requests  │   │
│                                  │  • Alert rules      • Probability dist │   │
│                                  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
```
Kaggle API → Raw CSV → Feature Engineering → SMOTE Balancing → XGBoost Training
    → Model Evaluation (Quality Gate) → Model Registration → Docker Image → AKS Deployment
    → FastAPI /predict endpoint → Prometheus Metrics → Grafana Dashboard
```

---

## 📁 Project Structure

```
fraud-detection-mlops/
├── .github/
│   └── workflows/
│       └── main.yml                    # CI/CD pipeline (5 jobs)
├── api/
│   ├── __init__.py
│   ├── app.py                          # FastAPI application with Prometheus
│   └── schemas.py                      # Pydantic request/response models
├── azure_ml/
│   ├── __init__.py
│   ├── config.py                       # Azure ML configuration
│   ├── environment.py                  # Compute & environment setup
│   ├── pipeline.py                     # ML pipeline definition (4 steps)
│   └── steps/
│       ├── data_ingestion_step.py      # Step 1: Kaggle download
│       ├── preprocessing_step.py       # Step 2: Feature engineering
│       ├── training_step.py            # Step 3: XGBoost training
│       └── evaluation_step.py          # Step 4: Evaluation + quality gate
├── data/
│   └── .gitkeep
├── kubernetes/
│   ├── deployment.yaml                 # K8s deployment (3 replicas)
│   ├── service.yaml                    # LoadBalancer service
│   └── hpa.yaml                        # Autoscaler (2-10 pods)
├── monitoring/
│   ├── grafana/
│   │   ├── dashboard.json              # Pre-built dashboard (10 panels)
│   │   ├── dashboard_provider.yml      # Auto-provisioning
│   │   └── datasource.yml              # Prometheus datasource
│   └── prometheus/
│       ├── alert_rules.yml             # 6 alert rules
│       └── prometheus.yml              # Scrape configuration
├── scripts/
│   ├── download_dataset.sh             # Bash dataset downloader
│   ├── run_pipeline.py                 # Local pipeline runner
│   ├── setup_azure.sh                  # Azure infrastructure setup
│   └── setup_kaggle.py                 # Kaggle credential setup
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py               # Kaggle API download & validation
│   ├── preprocessing.py                # Feature engineering & SMOTE
│   ├── train.py                        # XGBoost training with MLflow
│   ├── evaluate.py                     # Evaluation & quality gates
│   └── inference.py                    # Production inference engine
├── tests/
│   ├── __init__.py
│   ├── test_api.py                     # API schema tests
│   └── test_preprocessing.py           # Preprocessing pipeline tests
├── .dockerignore
├── .env.example                        # Environment variable template
├── .gitignore
├── config.yaml                         # Project configuration
├── docker-compose.yml                  # Full local stack
├── Dockerfile                          # Multi-stage production build
├── requirements-dev.txt                # Dev/test dependencies
├── requirements.txt                    # Production dependencies
├── setup.py                            # Package setup
└── README.md                           # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Azure CLI (for cloud deployment)
- Kaggle account with API key

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-mlops.git
cd fraud-detection-mlops

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configure Kaggle Credentials

```bash
# Option A: Environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option B: Run setup script
python scripts/setup_kaggle.py
```

### 3. Run Local Pipeline

```bash
# Full pipeline: download → preprocess → train → evaluate
python scripts/run_pipeline.py

# Or run steps individually:
python src/data_ingestion.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py
```

### 4. Serve the API

```bash
# Direct
python -m api.app

# Or with uvicorn
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "V1": -2.312, "V2": 1.952, "V3": -1.610, "V4": 3.998,
    "V5": -0.522, "V6": -1.427, "V7": -2.537, "V8": 1.392,
    "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.900,
    "V13": -0.595, "V14": -4.290, "V15": 0.390, "V16": -1.141,
    "V17": -2.830, "V18": -0.017, "V19": 0.416, "V20": 0.126,
    "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
    "V25": 0.045, "V26": 0.178, "V27": 0.261, "V28": -0.143,
    "Amount": 0.0
  }'
```

Response:
```json
{
  "fraud_probability": 0.9234,
  "is_fraud": true,
  "threshold": 0.5,
  "risk_level": "CRITICAL",
  "latency_ms": 3.45
}
```

---

## 🐳 Docker Deployment

### Build & Run API Only

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 -v ./artifacts:/app/artifacts:ro fraud-detection-api
```

### Full Stack (API + Prometheus + Grafana)

```bash
# Copy and edit environment variables
cp .env.example .env

# Start all services
docker-compose up -d

# Access:
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin / FraudDetection2024!)
```

---

## ☁️ Azure Deployment

### 1. Provision Infrastructure

```bash
# Login to Azure
az login

# Set environment variables
export AZURE_RESOURCE_GROUP=fraud-detection-rg
export AZURE_LOCATION=eastus
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# Run infrastructure setup
bash scripts/setup_azure.sh
```

This creates:
- Resource Group
- Azure Container Registry (ACR)
- Azure ML Workspace
- AKS Cluster (3 nodes)
- Kubernetes secrets

### 2. Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets):

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | Service principal JSON (`az ad sp create-for-rbac --sdk-auth`) |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `KAGGLE_USERNAME` | Kaggle username |
| `KAGGLE_KEY` | Kaggle API key |

### 3. Push to GitHub

```bash
git add .
git commit -m "Initial commit: fraud detection MLOps pipeline"
git push origin main
```

The CI/CD pipeline will automatically:
1. Run linting & unit tests
2. Download dataset from Kaggle
3. Build & push Docker image to ACR
4. Submit Azure ML training pipeline
5. Deploy to AKS

---

## ⚙️ Azure ML Pipeline

The pipeline runs 4 sequential steps on Azure ML compute:

| Step | Script | Description |
|------|--------|-------------|
| **1. Data Ingestion** | `data_ingestion_step.py` | Downloads dataset from Kaggle API |
| **2. Preprocessing** | `preprocessing_step.py` | Feature engineering, scaling, SMOTE |
| **3. Training** | `training_step.py` | XGBoost training with MLflow logging |
| **4. Evaluation** | `evaluation_step.py` | Quality gate check (ROC-AUC ≥ 0.95, PR-AUC ≥ 0.70) |

```bash
# Submit pipeline manually
python azure_ml/pipeline.py
```

---

## 📊 Monitoring

### Prometheus Metrics

The API exposes these metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_api_requests_total` | Counter | Total requests by method/endpoint/status |
| `fraud_api_request_duration_seconds` | Histogram | Request latency distribution |
| `fraud_predictions_total` | Counter | Predictions by result (fraud/legit) |
| `fraud_prediction_probability` | Histogram | Fraud probability distribution |
| `fraud_model_loaded` | Gauge | Model load status (0/1) |
| `fraud_api_active_requests` | Gauge | Current in-flight requests |

### Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| `HighErrorRate` | Error rate > 5% for 5m | Critical |
| `HighLatency` | p95 > 500ms for 10m | Warning |
| `APIDown` | Target unreachable for 2m | Critical |
| `AnomalousHighFraudRate` | Fraud rate > 10% for 15m | Warning |
| `ModelNotLoaded` | Model gauge = 0 for 1m | Critical |
| `NoPredictions` | Zero predictions for 30m | Warning |

### Grafana Dashboard

Pre-configured dashboard with 10 panels:
- Error rate, P95 latency, Request rate, Model status (stat panels)
- Prediction rate by result, Fraud detection rate (time series)
- Fraud probability distribution (histogram)
- Latency percentiles (p50/p90/p95/p99)
- Active requests, Request rate by endpoint

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=api --cov-report=html

# Lint
flake8 src/ api/ --max-line-length=120
black --check src/ api/
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check (K8s probes) |
| `GET` | `/model/info` | Model metadata |
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 10,000) |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

---

## 📐 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | XGBoost (XGBClassifier) |
| **Dataset** | Kaggle Credit Card Fraud Detection (284,807 transactions) |
| **Features** | 28 PCA components + 5 engineered features |
| **Class Balance** | SMOTE oversampling (50% ratio) |
| **Scaling** | RobustScaler |
| **Quality Gate** | ROC-AUC ≥ 0.95, PR-AUC ≥ 0.70 |

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `Hour` | Hour of day extracted from Time |
| `Log_Amount` | log(1 + Amount) |
| `Amount_Bin` | Binned amount (5 categories) |
| `V1_V2` | Interaction: V1 × V2 |
| `V1_V3` | Interaction: V1 × V3 |

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
