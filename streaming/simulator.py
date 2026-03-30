"""
Async streaming event simulator.

Replays events.csv as if events arrive in real time.
Computes rolling risk score per user and:
  1. Emits alert via on_alert callback
  2. Publishes to Kafka + RabbitMQ via AlertRouter
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_PATH, ALERT_THRESHOLD

log = logging.getLogger("forexguard.simulator")


class UserState:
    def __init__(self, user_id: str):
        self.user_id    = user_id
        self.events     : list[dict] = []
        self.risk_score : float = 0.0
        self.n_alerts   : int = 0


def _heuristic_score(events: list[dict]) -> tuple[float, dict]:
    if not events:
        return 0.0, {}

    n_unique_ips = len({e.get("ip_address") for e in events if e.get("ip_address")})
    n_logins     = sum(1 for e in events if e.get("event_type") == "login")
    n_trades     = sum(1 for e in events if e.get("event_type") == "trade")
    volumes      = [e.get("trade_volume") or 0 for e in events if e.get("event_type") == "trade"]
    deposits     = [e.get("amount") or 0 for e in events if e.get("event_type") == "deposit"]
    nav_rates    = [e.get("navigation_rate") or 0 for e in events if e.get("event_type") == "session"]
    failed_logins= sum(1 for e in events if e.get("event_type") == "login"
                       and e.get("login_success") == False)

    # Impossible travel: consecutive logins from different countries < 120 min
    login_evts = sorted([e for e in events if e.get("event_type") == "login"
                         and e.get("country")], key=lambda x: x["timestamp"])
    impossible_travel = 0
    for i in range(1, len(login_evts)):
        prev, curr = login_evts[i-1], login_evts[i]
        if prev["country"] != curr["country"]:
            try:
                gap = (pd.Timestamp(curr["timestamp"]) -
                       pd.Timestamp(prev["timestamp"])).total_seconds() / 60
                if gap < 120:
                    impossible_travel += 1
            except Exception:
                pass

    feats, signals = {}, []

    # IP diversity
    if n_logins > 0:
        feats["ip_diversity"] = n_unique_ips / max(n_logins, 1)
        if n_unique_ips >= 3:
            signals.append(0.35)

    # Brute force
    if n_logins > 0:
        feats["failed_login_ratio"] = failed_logins / max(n_logins, 1)
        if failed_logins >= 5:
            signals.append(0.45)

    # Impossible travel
    feats["impossible_travel"] = float(impossible_travel)
    if impossible_travel >= 1:
        signals.append(0.50)

    # Volume spike
    if len(volumes) >= 3:
        vol_cv = float(np.std(volumes) / (np.mean(volumes) + 1e-9))
        feats["volume_cv"] = vol_cv
        if vol_cv > 3.0:
            signals.append(0.45)

    # Off-hours trading
    off_hours = sum(1 for e in events
                    if e.get("event_type") == "trade"
                    and hasattr(e.get("timestamp"), "hour")
                    and 1 <= pd.Timestamp(e["timestamp"]).hour <= 5)
    if n_trades > 0:
        feats["off_hours_ratio"] = off_hours / n_trades
        if feats["off_hours_ratio"] > 0.7:
            signals.append(0.40)

    # Structuring
    if deposits:
        small_ratio = sum(1 for d in deposits if d < 1000) / len(deposits)
        feats["small_deposit_ratio"] = small_ratio
        if small_ratio > 0.7 and len(deposits) > 5:
            signals.append(0.40)

    # Micro-deposit probing
    micro_deps = sum(1 for d in deposits if d < 10)
    large_deps = sum(1 for d in deposits if d > 10000)
    if micro_deps > 5 and large_deps > 0:
        feats["micro_probe"] = float(micro_deps)
        signals.append(0.45)

    # Bot navigation
    if nav_rates:
        max_nav = max(nav_rates)
        feats["max_navigation_rate"] = max_nav
        if max_nav > 50:
            signals.append(0.45)

    # Deposit cycling
    deposit_total = sum(deposits)
    feats["deposit_to_trade_ratio"] = deposit_total / max(n_trades, 1)
    if deposit_total > 20_000 and n_trades < 3:
        signals.append(0.50)

    score = float(np.clip(sum(signals), 0.0, 1.0))
    return score, feats


class EventStreamSimulator:
    def __init__(self, events_path: Path = RAW_DATA_PATH,
                 batch_size: int = 50, speed_up: float = 1000.0,
                 max_events: int | None = None,
                 enable_broker: bool = True):
        self.events_path   = events_path
        self.batch_size    = batch_size
        self.speed_up      = speed_up
        self.max_events    = max_events
        self.enable_broker = enable_broker
        self._states       : dict[str, UserState] = {}
        self._alert_log    : list[dict] = []
        self._router       = None

    def _init_router(self):
        if not self.enable_broker:
            return
        try:
            from streaming.alert_router import AlertRouter
            self._router = AlertRouter()
            log.info("[simulator] alert router initialised")
        except Exception as e:
            log.warning(f"[simulator] alert router unavailable: {e}")

    async def _event_source(self):
        df = pd.read_csv(self.events_path, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        if self.max_events:
            df = df.head(self.max_events)
        prev_ts = None
        for row in df.to_dict("records"):
            if prev_ts is not None:
                gap   = (row["timestamp"] - prev_ts).total_seconds()
                delay = gap / self.speed_up
                if delay > 0:
                    await asyncio.sleep(min(delay, 0.01))
            prev_ts = row["timestamp"]
            yield row

    def _process_batch(self, batch: list[dict]):
        for event in batch:
            uid = event["user_id"]
            if uid not in self._states:
                self._states[uid] = UserState(uid)
            self._states[uid].events.append(event)

        for uid in {e["user_id"] for e in batch}:
            state = self._states[uid]
            score, feats = _heuristic_score(state.events)
            state.risk_score = score

            if score >= ALERT_THRESHOLD:
                state.n_alerts += 1
                alert = {
                    "user_id":       uid,
                    "timestamp":     datetime.utcnow().isoformat() + "Z",
                    "score":         round(score, 4),
                    "ensemble_score": round(score, 4),
                    "n_events":      len(state.events),
                    "top_feats":     {k: round(v, 4) for k, v in feats.items()},
                    "top_features":  {k: round(v, 4) for k, v in feats.items()},
                    "severity":      ("CRITICAL" if score >= 0.85 else
                                      "HIGH"     if score >= 0.70 else "MEDIUM"),
                    "flags":         [k for k, v in feats.items() if v > 0],
                    "model_scores":  {"isolation_forest": score,
                                      "lof": score, "lstm_ae": score},
                    "source":        "stream",
                }
                self._alert_log.append(alert)
                yield alert

    async def run(self, on_alert=None, verbose: bool = True):
        self._init_router()
        batch, n_total = [], 0

        async for event in self._event_source():
            batch.append(event)
            n_total += 1

            if len(batch) >= self.batch_size:
                for alert in self._process_batch(batch):
                    if verbose:
                        print(f"[{alert['severity']}] {alert['user_id']} "
                              f"score={alert['score']:.3f} "
                              f"flags={list(alert['top_feats'].keys())[:3]}")
                    # Publish to Kafka + RabbitMQ
                    if self._router:
                        self._router.publish(alert)
                    if on_alert:
                        await on_alert(alert)
                batch = []

        for alert in self._process_batch(batch):
            if verbose:
                print(f"[{alert['severity']}] {alert['user_id']} "
                      f"score={alert['score']:.3f}")
            if self._router:
                self._router.publish(alert)
            if on_alert:
                await on_alert(alert)

        if self._router:
            self._router.close()

        if verbose:
            print(f"\n[simulator] {n_total:,} events | "
                  f"{len(self._alert_log)} alerts emitted")

    def get_alerts(self) -> list[dict]:
        return sorted(self._alert_log, key=lambda a: a["score"], reverse=True)

    def get_risk_scores(self) -> dict[str, float]:
        return {uid: st.risk_score for uid, st in self._states.items()}


async def _main():
    sim = EventStreamSimulator(max_events=5_000, enable_broker=True)
    await sim.run(verbose=True)
    for a in sim.get_alerts()[:3]:
        print(json.dumps(a, indent=2, default=str))

if __name__ == "__main__":
    import json
    asyncio.run(_main())
