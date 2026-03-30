---
title: ForexGuard
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# ForexGuard — Real-Time Trader Anomaly Detection Engine

> **AI/ML Internship Assessment Submission**
> A production-ready anomaly detection system for forex broker compliance teams.
> Detects suspicious trader behaviour in real time using an ensemble of unsupervised ML models,
> streaming simulation, Kafka/RabbitMQ alert publishing, and Gemini LLM-generated risk narratives.

---

## Links

| | |
|---|---|
| **Live Demo** | https://Lavanya-777-forexguard.hf.space/docs |
| **GitHub** | https://github.com/heyhellobye/forexguard |
| **API Health** | https://Lavanya-777-forexguard.hf.space/health |
| **All Alerts** | https://Lavanya-777-forexguard.hf.space/alerts |

---

## Architecture Overview

```
EVENT SOURCES
Client Portal Events | WebTrader Events | Stream Replay
         |
         v async micro-batches
STREAMING LAYER  (streaming/simulator.py)
EventStreamSimulator -> rolling per-user state -> alert emission
         |
         v
FEATURE ENGINEERING  (features/engineering.py + cross_user.py)
59 per-user features + 5 cross-user graph features
Login | Trading | Financial | KYC | Session | News | Graph
         |
    +----+--------------------+
    v                         v
BASELINE MODELS          ADVANCED MODEL
Isolation Forest         LSTM Autoencoder
LOF                      (T=12 x F=64 sequences)
SHAP explanations        Reconstruction error score
    |                         |
    +----------+--------------+
               v
         ENSEMBLE SCORER
         IF x0.35 + LOF x0.25 + LSTM x0.40
         + Disagreement Score
               |
               v
         ALERT GENERATOR
         4-tier severity (LOW/MEDIUM/HIGH/CRITICAL)
         46 compliance flag rules
         Gemini LLM risk narratives
               |
         +-----+-----+
         v           v
      FastAPI    Message Brokers
    10 endpoints  Kafka + RabbitMQ
```

---

## Directory Structure

```
forexguard/
├── config.py                    # All hyperparameters and paths
├── train.py                     # End-to-end training pipeline (8 steps)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml           # API + Redpanda + RabbitMQ + MLflow
├── data/
│   └── generate.py              # Synthetic 50 000-event generator (16 anomaly types)
├── features/
│   ├── engineering.py           # 59 per-user behavioural features
│   └── cross_user.py            # 5 cross-user graph features
├── models/
│   ├── baseline.py              # Isolation Forest + LOF with SHAP
│   └── lstm_ae.py               # LSTM Autoencoder (PyTorch)
├── streaming/
│   ├── simulator.py             # Async event-stream replay
│   ├── kafka_publisher.py       # Kafka/Redpanda alert publisher
│   ├── rabbitmq_publisher.py    # RabbitMQ alert publisher
│   └── alert_router.py          # Unified broker interface
├── api/
│   ├── main.py                  # FastAPI application (10 endpoints)
│   └── schemas.py               # Pydantic request/response models
└── utils/
    ├── alerts.py                # Alert generation + 46 flag rules
    └── llm_summary.py           # Gemini LLM risk narrative generator
```

---

## Quick Start

```bash
git clone https://github.com/heyhellobye/forexguard.git
cd forexguard
pip install -r requirements.txt
python train.py
python -m uvicorn api.main:app --port 8000 --reload
```

Open http://localhost:8000/docs

### Enable Gemini LLM Summaries (optional)
```bash
pip install google-generativeai
set GEMINI_API_KEY=your-key-here
```
Get a free key at https://aistudio.google.com/app/apikey

### Run with Docker (includes Kafka + RabbitMQ)
```bash
docker compose up --build
```

---

## Dataset

Synthetic dataset of ~51,000 events across 500 users. 8% anomalous users with 16 injected anomaly patterns:

| Type | Pattern | Category |
|---|---|---|
| A | Rapid IP/geo switching | Login |
| B | Deposit -> minimal trade -> withdrawal | Financial |
| C | Trade volume spike (20x) | Trading |
| D | High-frequency small deposits (structuring) | Financial |
| E | Bot-like navigation (30s intervals) | Behavioural |
| F | Simultaneous multi-IP logins | Login |
| G | IP hub (multiple accounts share one IP) | Network |
| H | Bonus abuse cycle | Financial |
| I | Rapid KYC changes before withdrawal | Account |
| J | Brute force login then immediate withdrawal | Login |
| K | Trades aligned with news events | Temporal |
| L | Dormancy then sudden large withdrawal | Financial |
| M | Impossible travel (Mumbai to London in 12 min) | Login |
| N | Micro-deposit probing before large transfer | Financial |
| O | Off-hours automation (1AM-5AM perfect timing) | Behavioural |
| P | First-session expert trading (no learning curve) | Behavioural |

---

## Feature Engineering

64 total features per user (59 individual + 5 cross-user):

| Group | Key Features | Anomaly Detected |
|---|---|---|
| Login | n_unique_ips, ip_switch_rate, simultaneous_login_ratio | A, F |
| Impossible Travel | impossible_travel_count, impossible_travel_min_gap | M |
| Brute Force | failed_login_ratio, brute_force_pattern | J |
| Trading | volume_spike_ratio, inter_trade_regularity, trade_volume_cusum | C |
| Off-hours | off_hours_trade_ratio, off_hours_interval_std | O |
| First Session | first_session_volume_ratio, first_session_n_trades | P |
| News Alignment | news_aligned_trade_ratio, news_aligned_trade_count | K |
| Financial | small_deposit_ratio, deposit_per_trade | B, D |
| Micro-probing | micro_deposit_count, micro_before_large | N |
| Dormancy | dormancy_before_withdrawal_days | L |
| Bonus Abuse | n_bonus_claims, bonus_to_deposit_ratio | H |
| KYC | kyc_before_withdrawal_count, kyc_interval_min | I |
| Session/Bot | max_navigation_rate, session_interval_regularity | E |
| Cross-user | ip_hub_score, sync_trade_ratio, mirror_trade_score, withdrawal_cluster_score | G |

---

## Models

### Baseline 1 — Isolation Forest
Isolates anomalies by randomly partitioning the feature space. Anomalies require fewer splits (shorter average path length). Handles high-dimensional tabular data efficiently.
- 200 trees, contamination=0.08
- SHAP TreeExplainer for per-sample feature contributions

### Baseline 2 — Local Outlier Factor (LOF)
Captures local density anomalies — traders deviating from their peer neighbourhood rather than the global population. Effective for detecting users who behave unusually compared to similar accounts.
- n_neighbors=20, novelty=True for inference on new data

### Advanced — LSTM Autoencoder (PyTorch)
Captures temporal dependencies invisible to static models. Trained only on normal users — anomalous sequences produce high reconstruction error.

Architecture: Input (B, T=12, F=64) -> Encoder LSTM (2 layers, H=64) -> Bottleneck (latent=16) -> Decoder LSTM -> Output (B, T=12, F=64)

Threshold set at 95th percentile of training reconstruction errors. Per-feature reconstruction error serves as the explanation.

### Ensemble
```
score = 0.35 x IF  +  0.25 x LOF  +  0.40 x LSTM_AE
```
Also computes ensemble disagreement score (std dev of three model scores). High disagreement automatically flags for human review.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Model status + broker connectivity |
| POST | /score | Score a known user by ID |
| POST | /predict | Score from raw feature dict (online inference) |
| GET | /alerts | All flagged users sorted by severity |
| GET | /alerts/{user_id} | Single user detailed alert |
| GET | /alerts/llm/{user_id} | Alert + Gemini LLM narrative |
| GET | /alerts/llm/top/{n} | Top N alerts with LLM narratives |
| GET | /stream/start | Start async streaming simulation |
| GET | /stream/status | Stream progress + live alerts |
| GET | /broker/status | Kafka + RabbitMQ connection status |
| POST | /broker/test | Publish test alert to brokers |

---

## Streaming and Message Brokers

The streaming simulator replays 51,000 events asynchronously at 1000x speed. Alerts are published to:

Kafka/Redpanda topics: forexguard.alerts.login, forexguard.alerts.trading, forexguard.alerts.deposit, forexguard.alerts.withdrawal, forexguard.alerts.critical

RabbitMQ topic exchange with same routing keys — compliance teams bind queues with wildcards (forexguard.alerts.# for all alerts, forexguard.alerts.critical for critical only).

Both brokers degrade gracefully if not running.

---

## Deployment

### Local
```bash
pip install -r requirements.txt
python train.py
python -m uvicorn api.main:app --port 8000 --reload
```

### Docker
```bash
docker compose up --build
```

Services: API on :8000, Redpanda Console on :8080, RabbitMQ UI on :15672, MLflow on :5000

### Live (HuggingFace Spaces)
```
https://Lavanya-777-forexguard.hf.space/docs
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML | PyTorch, scikit-learn, SHAP |
| Data | Pandas, NumPy, SciPy |
| API | FastAPI, Uvicorn, Pydantic |
| Streaming | Async Python, confluent-kafka, pika (RabbitMQ) |
| LLM | Google Gemini 1.5 Flash |
| Tracking | MLflow |
| Infra | Docker, HuggingFace Spaces |

---

## Assumptions, Trade-offs and Limitations

### Assumptions
- Synthetic anomaly patterns represent realistic forex fraud scenarios
- Features computed over full observation window (production would use 7-30 day rolling window)
- Models trained fully unsupervised — labels used only for offline evaluation

### Trade-offs
- Heuristic scorer used in stream instead of full ML inference to avoid 50-200ms latency in hot path
- SHAP explainer rebuilt at training time only — not persisted — fallback uses raw feature magnitudes
- TCP probe before Kafka initialisation prevents confluent-kafka stderr flooding when broker is offline

### Limitations
- Cross-user graph features computed in-memory at training time — would need Neo4j in production
- LSTM uses aggregate 4-hour time buckets rather than raw event embeddings
- No concept drift detection — models should be retrained weekly in production
- HuggingFace free tier sleeps after 15 min inactivity — first request after sleep takes ~30 seconds
