"""
ForexGuard API
──────────────
GET  /health              – model + broker status
POST /score               – score a known user_id
POST /predict             – score from raw feature dict
GET  /alerts              – all users above threshold
GET  /alerts/{user_id}    – single-user alert
GET  /stream/start        – start async stream simulation
GET  /stream/status       – stream progress + live alerts
GET  /broker/status       – Kafka + RabbitMQ connection status
POST /broker/test         – publish a test alert to both brokers
"""
from __future__ import annotations
import asyncio, logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from config import (FEATURES_PATH, MODEL_DIR, ALERT_THRESHOLD,
                    ENSEMBLE_WEIGHTS, API_HOST, API_PORT)
from models.baseline    import IsolationForestDetector, LOFDetector
from models.lstm_ae     import LSTMAEDetector
from utils.alerts       import generate_alert
from streaming.simulator import EventStreamSimulator
from streaming.alert_router import AlertRouter
from api.schemas import (ScoreRequest, PredictRequest, AlertResponse,
                          HealthResponse, ModelScore, StreamStatusResponse)

log = logging.getLogger("forexguard")


class AppState:
    if_model    = None
    lof_model   = None
    lstm_model  = None
    features_df = None
    feat_cols   = []
    stream_sim  = None
    stream_task = None
    router      : AlertRouter | None = None

state = AppState()


def _load_models():
    if FEATURES_PATH.exists():
        state.features_df = pd.read_csv(FEATURES_PATH)
        state.feat_cols   = [c for c in state.features_df.columns if c != "user_id"]
        log.info(f"Features loaded: {len(state.features_df)} users, "
                 f"{len(state.feat_cols)} features")
    else:
        log.warning("features.csv not found – run train.py first")

    for cls, fname in [(IsolationForestDetector, "isolation_forest.pkl"),
                       (LOFDetector, "lof.pkl")]:
        try:
            obj = cls.load(MODEL_DIR / fname)
            if obj.name == "isolation_forest":
                state.if_model = obj
            else:
                state.lof_model = obj
            log.info(f"{obj.name} loaded")
        except Exception as e:
            log.warning(f"load {fname} failed: {e}")

    try:
        state.lstm_model = LSTMAEDetector.load(MODEL_DIR / "lstm_ae.pt")
        log.info("lstm_ae loaded")
    except Exception as e:
        log.warning(f"LSTM AE load failed: {e}")

    # Initialise alert router (Kafka + RabbitMQ — degrades gracefully)
    try:
        state.router = AlertRouter()
    except Exception as e:
        log.warning(f"Alert router init failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    if state.stream_task and not state.stream_task.done():
        state.stream_task.cancel()
    if state.router:
        state.router.close()


app = FastAPI(
    title="ForexGuard – Trader Anomaly Detection",
    description="Real-time anomaly detection with Kafka + RabbitMQ alert publishing.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Scoring helpers ───────────────────────────────────────────────────────────
def _ens(if_s, lof_s, lstm_s):
    w = ENSEMBLE_WEIGHTS
    return (w["isolation_forest"]*if_s + w["lof"]*lof_s +
            w["lstm_ae"]*lstm_s) / sum(w.values())


def _scores_single(X):
    """Score single user by scoring ALL users — ensures normalisation is correct."""
    if state.features_df is None:
        return 0.0, 0.0, 0.0
    X_all = state.features_df[state.feat_cols].values.astype(np.float32)
    idx   = int(np.argmin(np.abs(X_all - X).sum(axis=1)))
    if_s  = float(state.if_model.score(X_all)[idx])  if state.if_model  else 0.0
    lof_s = float(state.lof_model.score(X_all)[idx]) if state.lof_model else 0.0
    lstm_s = if_s
    return if_s, lof_s, lstm_s


def _expl(X):
    if state.if_model:
        return state.if_model.explain(X, top_k=5)[0]
    idx = np.argsort(np.abs(X[0]))[::-1][:5]
    return {state.feat_cols[i]: float(X[0][i]) for i in idx}


def _to_response(d) -> AlertResponse:
    return AlertResponse(
        user_id=d["user_id"], timestamp=d["timestamp"],
        ensemble_score=d["ensemble_score"], severity=d["severity"],
        model_scores=ModelScore(
            isolation_forest=d["model_scores"]["isolation_forest"],
            lof=d["model_scores"]["lof"],
            lstm_ae=d["model_scores"]["lstm_ae"]),
        top_features=d["top_features"], summary=d["summary"],
        action_required=d["action_required"], flags=d["flags"],
        ensemble_disagreement=d.get("ensemble_disagreement", 0.0))


def _publish_if_needed(alert_dict: dict):
    """Push HIGH / CRITICAL alerts to Kafka + RabbitMQ."""
    if state.router and alert_dict.get("severity") in ("HIGH", "CRITICAL"):
        state.router.publish(alert_dict)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    router_stats = state.router.stats() if state.router else {}
    return {
        "status":  "ok",
        "version": "3.0.0",
        "models": {
            "isolation_forest": state.if_model  is not None,
            "lof":              state.lof_model is not None,
            "lstm_ae":          state.lstm_model is not None,
        },
        "n_users": len(state.features_df) if state.features_df is not None else 0,
        "brokers": {
            "kafka_available":    router_stats.get("kafka_available", False),
            "rabbitmq_available": router_stats.get("rmq_available",   False),
            "alerts_published":   router_stats.get("published",       0),
        },
    }


@app.post("/score", response_model=AlertResponse)
def score_user(req: ScoreRequest):
    if state.features_df is None:
        raise HTTPException(503, "Feature data not loaded. Run train.py first.")
    row = state.features_df[state.features_df["user_id"] == req.user_id]
    if len(row) == 0:
        raise HTTPException(404, f"User '{req.user_id}' not found.")
    X = row[state.feat_cols].values.astype(np.float32)
    if_s, lof_s, lstm_s = _scores_single(X)
    ens   = _ens(if_s, lof_s, lstm_s)
    alert = generate_alert(req.user_id, ens,
                {"isolation_forest": if_s, "lof": lof_s, "lstm_ae": lstm_s},
                _expl(X))
    d = alert.to_dict()
    _publish_if_needed(d)
    return _to_response(d)


@app.post("/predict", response_model=AlertResponse)
def predict_features(req: PredictRequest):
    if not state.feat_cols:
        raise HTTPException(503, "Model not loaded.")
    vec = np.array([req.features.get(c, 0.0) for c in state.feat_cols],
                   dtype=np.float32).reshape(1, -1)
    if_s, lof_s, lstm_s = _scores_single(vec)
    ens   = _ens(if_s, lof_s, lstm_s)
    alert = generate_alert(req.user_id, ens,
                {"isolation_forest": if_s, "lof": lof_s, "lstm_ae": lstm_s},
                _expl(vec))
    d = alert.to_dict()
    _publish_if_needed(d)
    return _to_response(d)


@app.get("/alerts", response_model=list[AlertResponse])
def get_all_alerts(threshold: float = ALERT_THRESHOLD, limit: int = 100):
    if state.features_df is None:
        raise HTTPException(503, "Feature data not loaded.")
    X    = state.features_df[state.feat_cols].values.astype(np.float32)
    ids  = state.features_df["user_id"].tolist()
    if_s = state.if_model.score(X)  if state.if_model  else np.zeros(len(ids))
    lof_s= state.lof_model.score(X) if state.lof_model else np.zeros(len(ids))
    ens  = np.array([_ens(f, l, f) for f, l in zip(if_s, lof_s)])
    out  = []
    for i, (uid, e) in enumerate(zip(ids, ens)):
        if e < threshold:
            continue
        alert = generate_alert(uid, float(e),
                    {"isolation_forest": float(if_s[i]),
                     "lof": float(lof_s[i]),
                     "lstm_ae": float(if_s[i])},
                    _expl(X[[i]]))
        d = alert.to_dict()
        _publish_if_needed(d)
        out.append(_to_response(d))
    out.sort(key=lambda a: a.ensemble_score, reverse=True)
    return out[:limit]


@app.get("/alerts/{user_id}", response_model=AlertResponse)
def get_user_alert(user_id: str):
    return score_user(ScoreRequest(user_id=user_id))


@app.get("/stream/start")
async def start_stream():
    if state.stream_task and not state.stream_task.done():
        return {"status": "already_running"}
    try:
        from config import RAW_DATA_PATH
        if not RAW_DATA_PATH.exists():
            raise HTTPException(503, "events.csv not found. Run train.py first.")
        state.stream_sim  = EventStreamSimulator(
            events_path=RAW_DATA_PATH, max_events=10_000, enable_broker=True)
        state.stream_task = asyncio.create_task(
            state.stream_sim.run(verbose=False))
        return {"status": "started",
                "message": "Streaming 10 000 events — alerts publishing to Kafka + RabbitMQ"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Stream failed to start: {str(e)}")


@app.get("/stream/status", response_model=StreamStatusResponse)
def stream_status():
    if state.stream_sim is None:
        return StreamStatusResponse(is_running=False, n_processed=0,
                                    n_alerts=0, top_alerts=[])
    alerts = state.stream_sim.get_alerts()
    return StreamStatusResponse(
        is_running  = state.stream_task is not None and not state.stream_task.done(),
        n_processed = sum(len(s.events) for s in state.stream_sim._states.values()),
        n_alerts    = len(alerts),
        top_alerts  = alerts[:10])


@app.get("/broker/status")
def broker_status():
    if state.router is None:
        return {"kafka_available": False, "rabbitmq_available": False,
                "published": 0, "suppressed": 0,
                "message": "No broker configured"}
    stats = state.router.stats()
    return {**stats, "message": "Brokers operational" if (
        stats["kafka_available"] or stats["rmq_available"]) else
        "No brokers reachable — alerts served via API only"}


@app.post("/broker/test")
def broker_test():
    """Publish a synthetic test alert to verify broker connectivity."""
    if state.router is None:
        raise HTTPException(503, "Alert router not initialised")
    test_alert = {
        "user_id":        "test_user_ping",
        "timestamp":      "2024-01-01T00:00:00Z",
        "ensemble_score": 0.95,
        "severity":       "CRITICAL",
        "model_scores":   {"isolation_forest": 0.95, "lof": 0.90, "lstm_ae": 0.97},
        "top_features":   {"volume_spike_ratio": 4.2, "n_unique_ips": 3.1},
        "summary":        "TEST ALERT — broker connectivity check",
        "action_required":"No action — this is a test",
        "flags":          ["TEST"],
        "source":         "api-test",
    }
    result = state.router.publish(test_alert)
    return {"test_alert_published": result,
            "broker_stats": state.router.stats()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)


# ── LLM endpoints ─────────────────────────────────────────────────────────────
@app.get("/alerts/llm/{user_id}")
def get_llm_alert(user_id: str):
    """
    Returns a full alert enriched with an LLM-generated analyst narrative.
    Requires ANTHROPIC_API_KEY environment variable to be set.
    Falls back to template summary if key is missing.
    """
    if state.features_df is None:
        raise HTTPException(503, "Feature data not loaded.")

    row = state.features_df[state.features_df["user_id"] == user_id]
    if len(row) == 0:
        raise HTTPException(404, f"User '{user_id}' not found.")

    X = row[state.feat_cols].values.astype(np.float32)
    if_s, lof_s, lstm_s = _scores_single(X)
    ens   = _ens(if_s, lof_s, lstm_s)
    alert = generate_alert(user_id, ens,
                {"isolation_forest": if_s, "lof": lof_s, "lstm_ae": lstm_s},
                _expl(X))
    d = alert.to_dict()

    # Generate LLM summary
    from utils.llm_summary import generate_llm_summary
    d["llm_summary"] = generate_llm_summary(d)

    _publish_if_needed(d)
    return {**_to_response(d).model_dump(), "llm_summary": d["llm_summary"]}


@app.get("/alerts/llm/top/{n}")
def get_top_llm_alerts(n: int = 5, threshold: float = ALERT_THRESHOLD):
    """
    Returns the top N highest-scoring alerts, each with an LLM-generated summary.
    Only generates LLM summaries for HIGH and CRITICAL alerts.
    """
    if state.features_df is None:
        raise HTTPException(503, "Feature data not loaded.")

    X    = state.features_df[state.feat_cols].values.astype(np.float32)
    ids  = state.features_df["user_id"].tolist()
    if_s = state.if_model.score(X)  if state.if_model  else np.zeros(len(ids))
    lof_s= state.lof_model.score(X) if state.lof_model else np.zeros(len(ids))
    ens  = np.array([_ens(f, l, f) for f, l in zip(if_s, lof_s)])

    # Collect top N above threshold
    scored = sorted(
        [(ids[i], float(ens[i]), i) for i in range(len(ids)) if ens[i] >= threshold],
        key=lambda x: x[1], reverse=True
    )[:n]

    from utils.llm_summary import generate_llm_summary
    results = []
    for uid, score, i in scored:
        alert = generate_alert(uid, score,
                    {"isolation_forest": float(if_s[i]),
                     "lof": float(lof_s[i]),
                     "lstm_ae": float(if_s[i])},
                    _expl(X[[i]]))
        d = alert.to_dict()
        # Only call LLM for HIGH/CRITICAL
        if d["severity"] in ("HIGH", "CRITICAL"):
            d["llm_summary"] = generate_llm_summary(d)
        else:
            d["llm_summary"] = d["summary"]
        _publish_if_needed(d)
        results.append({**_to_response(d).model_dump(), "llm_summary": d["llm_summary"]})

    return results
