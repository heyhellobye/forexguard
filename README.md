# ForexGuard — Real-Time Trader Anomaly Detection Engine

> **AI/ML Internship Assessment Prototype**  
> Detects suspicious trader behaviour using an ensemble of unsupervised ML models,
> a streaming simulation layer, and a production-grade FastAPI inference server.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Model Explanation](#model-explanation)
6. [Feature Engineering](#feature-engineering)
7. [API Reference](#api-reference)
8. [Streaming Simulation](#streaming-simulation)
9. [Docker Deployment](#docker-deployment)
10. [MLflow Tracking](#mlflow-tracking)
11. [Assumptions, Trade-offs & Limitations](#assumptions-trade-offs--limitations)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                   EVENT SOURCES                          │
│  Client Portal Events  │  WebTrader Events  │  Replay   │
└────────────┬─────────────────────────────────────────────┘
             │ async micro-batches
             ▼
┌──────────────────────────────────────────────────────────┐
│         STREAMING LAYER  (streaming/simulator.py)        │
│  EventStreamSimulator → rolling per-user state           │
│  heuristic rule scorer → real-time alert emission        │
└────────────┬─────────────────────────────────────────────┘
             │ per-user event history
             ▼
┌──────────────────────────────────────────────────────────┐
│       FEATURE ENGINEERING  (features/engineering.py)     │
│  ~30 per-user aggregate features + LSTM time sequences   │
└────────────┬─────────────────────────────────────────────┘
             │
       ┌─────┴──────────────────┐
       ▼                        ▼
┌─────────────────┐    ┌────────────────────────┐
│ BASELINE MODELS │    │  ADVANCED MODEL         │
│ Isolation Forest│    │  LSTM Autoencoder       │
│ LOF             │    │  (T=12 × F=30 seqs)     │
│ SHAP explain.   │    │  Reconstruction error   │
└────────┬────────┘    └──────────┬─────────────┘
         │  score∈[0,1]           │  score∈[0,1]
         └──────────┬─────────────┘
                    ▼
         ┌────────────────────────┐
         │  ENSEMBLE SCORER       │
         │  IF×0.35+LOF×0.25      │
         │  +LSTM×0.40            │
         └──────────┬─────────────┘
                    ▼
         ┌────────────────────────┐
         │  ALERT GENERATOR       │
         │  Severity + Summary    │
         │  Top features          │
         └──────────┬─────────────┘
                    ▼
         ┌────────────────────────┐
         │  FastAPI  (:8000)      │
         │  /score /predict       │
         │  /alerts /stream       │
         └────────────────────────┘
```

---

## Directory Structure

```
forexguard/
├── config.py                  # All hyper-parameters & paths
├── train.py                   # End-to-end training pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── generate.py            # Synthetic 50 000-event generator
│   ├── events.csv             # Generated after train.py
│   └── labels.csv             # Ground-truth anomaly labels
│
├── features/
│   └── engineering.py         # Per-user + LSTM sequence builder
│
├── models/
│   ├── baseline.py            # IsolationForest + LOF w/ SHAP
│   └── lstm_ae.py             # LSTM Autoencoder (PyTorch)
│
├── streaming/
│   └── simulator.py           # Async event-stream replay
│
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py             # Pydantic schemas
│
├── utils/
│   └── alerts.py              # Alert dataclass + templates
│
└── saved_models/              # Serialised models (post-training)
    ├── isolation_forest.pkl
    ├── lof.pkl
    ├── lstm_ae.pt
    └── training_summary.json
```

---

## Quick Start

```bash
cd forexguard
pip install -r requirements.txt
python train.py                          # ~5-10 min on CPU
uvicorn api.main:app --port 8000 --reload
open http://localhost:8000/docs
```

---

## Step-by-Step Setup

### Prerequisites
- Python 3.10 or 3.11
- pip >= 23
- 4 GB RAM minimum (8 GB recommended for LSTM training)
- CUDA GPU optional

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train all models

```bash
python train.py
```

This single command:
- Generates `data/events.csv` (~50 000 events, 500 users, ~6% anomalous)
- Builds `data/features.csv` (~30 features per user)
- Trains Isolation Forest → `saved_models/isolation_forest.pkl`
- Trains LOF             → `saved_models/lof.pkl`
- Trains LSTM Autoencoder → `saved_models/lstm_ae.pt`
- Prints AUROC / Precision / Recall / F1 per model + ensemble
- Saves `saved_models/training_summary.json`

Force-regenerate fresh data:
```bash
python train.py --regen
```

### 3 — Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4 — Test the endpoints

**Score a known user:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_0042"}'
```

**Online inference from raw features:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "new_user_999",
    "features": {
      "n_unique_ips": 4.2,
      "volume_spike_ratio": 8.5,
      "small_deposit_ratio": 3.1
    }
  }'
```

**All alerts above threshold:**
```bash
curl "http://localhost:8000/alerts?threshold=0.55&limit=20"
```

**Start stream + poll status:**
```bash
curl http://localhost:8000/stream/start
curl http://localhost:8000/stream/status
```

**Run stream standalone:**
```bash
python -m streaming.simulator
```

**Interactive docs:**
```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

---

## Model Explanation

### Baseline 1 — Isolation Forest
**Why chosen:** Industry standard for tabular anomaly detection. Isolates anomalies by randomly partitioning the feature space; anomalies require fewer splits (shorter path length). Handles high-dimensional data efficiently in O(n log n).

**Details:**
- 200 trees, contamination=0.06
- SHAP TreeExplainer for per-sample feature contributions
- Score normalised to [0,1]

### Baseline 2 — Local Outlier Factor (LOF)
**Why chosen:** Captures *local* density anomalies — a trader deviating from their peer neighbourhood rather than the global population. Useful for detecting users who behave unusually vs. similar accounts (same size, frequency tier).

**Details:**
- `novelty=True` for inference on unseen data
- n_neighbors=20, contamination=0.06
- Permutation-based feature importance

### Advanced — LSTM Autoencoder
**Why chosen:** Temporal dependencies (e.g. sudden volume spike after weeks of normal activity) are invisible to static tabular models. An autoencoder trained only on normal users learns to reconstruct normal sequences; anomalous sequences yield high reconstruction error.

**Architecture:**
```
Input  (B, T=12, F=30)
  └─ Encoder LSTM (2 layers, H=64)
       └─ FC bottleneck → latent=16
            └─ FC expand → H=64
                 └─ Decoder LSTM → Output (B, T=12, F=30)
```
- Trained on normal users only
- Threshold = 95th percentile of training reconstruction errors
- Per-feature reconstruction error = explanation

### Ensemble Scoring
```
score = 0.35 × IF  +  0.25 × LOF  +  0.40 × LSTM_AE
```
Weights are configurable in `config.py`. LSTM receives highest weight as it uniquely captures temporal context.

---

## Feature Engineering

~30 per-user features, z-score normalised:

| Feature | Signal Detected |
|---|---|
| `n_unique_ips` | IP diversity / account sharing |
| `n_unique_countries` | Geo-switching |
| `ip_switch_rate` | Rapid IP rotation |
| `simultaneous_login_ratio` | Concurrent multi-IP logins |
| `unusual_hour_ratio` | Off-hours access (00:00–06:00) |
| `volume_spike_ratio` | Trade volume spike (max/mean) |
| `instrument_concentration` | Single-instrument overuse |
| `avg_inter_trade_min` | High-frequency trading |
| `min_inter_trade_min` | Latency arbitrage |
| `pnl_std` | Erratic / suspicious PnL |
| `win_rate` | Consistent abnormal profit |
| `small_deposit_ratio` | Structuring (<$1 000 deposits) |
| `deposit_per_trade` | Deposit cycling ratio |
| `deposit_withdrawal_ratio` | Wash-cycling indicator |
| `max_navigation_rate` | Bot-like navigation speed |
| `session_duration_std` | Irregular session patterns |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model status + user count |
| `POST` | `/score` | Score user by ID |
| `POST` | `/predict` | Score from raw feature dict |
| `GET` | `/alerts` | All flagged users (sorted) |
| `GET` | `/alerts/{user_id}` | Single-user alert detail |
| `GET` | `/stream/start` | Launch async stream sim |
| `GET` | `/stream/status` | Stream progress + alerts |

**Example alert response:**
```json
{
  "user_id": "user_0123",
  "ensemble_score": 0.812,
  "severity": "HIGH",
  "model_scores": {
    "isolation_forest": 0.79,
    "lof": 0.74,
    "lstm_ae": 0.87
  },
  "top_features": {
    "volume_spike_ratio": 0.621,
    "n_unique_ips": 0.418,
    "small_deposit_ratio": 0.310
  },
  "summary": "HIGH RISK: User user_0123 displays suspicious activity (81%). Key signals: Trade volume spike; Multiple IPs; Structuring deposits. Flag for compliance review within 24 hours.",
  "action_required": "Place account under enhanced monitoring. Schedule compliance review within 24 h.",
  "flags": [
    "Trade volume spike (≥10× baseline)",
    "Multiple distinct IP addresses detected",
    "High ratio of sub-$1000 deposits (structuring signal)"
  ]
}
```

---

## Streaming Simulation

`streaming/simulator.py` replays the event CSV asynchronously at 1000× speed in micro-batches of 50 events. Per batch:

1. Events appended to per-user in-memory state
2. Heuristic rule scorer checks 5 fast signals
3. Score ≥ 0.60 → alert emitted
4. Alerts accessible at `/stream/status`

**To plug in real Kafka/Redpanda:** replace `_event_source()` with a `confluent-kafka` consumer. The rest of the pipeline is unchanged.

---

## Docker Deployment

```bash
# Build and start API + MLflow
docker compose up --build

# With Redpanda (Kafka-compatible)
docker compose --profile kafka up --build

# Run training inside container first
docker compose run --rm api python train.py
docker compose up api
```

Services:

| Service | Port | URL |
|---|---|---|
| ForexGuard API | 8000 | http://localhost:8000/docs |
| MLflow UI | 5000 | http://localhost:5000 |
| Redpanda | 9092 | kafka://localhost:9092 |

---

## MLflow Tracking

```bash
pip install mlflow
mlflow ui --port 5000
open http://localhost:5000
```

Tracked per training run:
- **Params:** `n_users`, `n_features`, `anomaly_ratio`
- **Metrics:** `auroc`, `precision`, `recall`, `f1` per model
- **Artifacts:** `training_summary.json`

---

## Assumptions, Trade-offs & Limitations

### Assumptions
- Injected anomaly patterns (IP cycling, deposit washing, volume spikes, structuring, bot navigation) represent realistic forex fraud scenarios. Real distributions would differ in frequency and magnitude.
- Features computed over the full observation window. Production would use a 7-day or 30-day rolling window.
- Labels exist only for offline evaluation; models are trained fully unsupervised.

### Trade-offs

| Decision | Alternative | Reason |
|---|---|---|
| Per-user aggregate features | Event-level stream features | Simpler; sufficient for batch compliance review |
| SHAP TreeExplainer on IF | LIME | Faster; architecturally consistent |
| Heuristic scorer in stream | Full ML inference in stream | Avoids model I/O latency in hot path |
| Async file replay | Real Kafka consumer | Zero infrastructure dependency; trivially swappable |
| Fixed ensemble weights | Learned meta-model | Avoids label leakage; more robust with sparse anomaly labels |

### Limitations
- LSTM uses aggregate feature buckets rather than raw event embeddings, losing fine-grained temporal resolution.
- No graph/network features (shared IPs across users, mirror trades). These require a graph DB + GNN-based detection.
- LOF degrades in very high dimensions; PCA pre-projection would improve it.
- No concept drift detection — models need retraining weekly in production.
- LSTM streaming inference requires sequence materialisation (~50–200 ms latency per user per batch).
