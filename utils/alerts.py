"""
Alert generation covering all 16 anomaly types + ensemble disagreement scoring.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


class Severity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    user_id              : str
    timestamp            : str
    ensemble_score       : float
    severity             : Severity
    model_scores         : dict
    top_features         : dict
    summary              : str
    action_required      : str
    raw_flags            : list = field(default_factory=list)
    ensemble_disagreement: float = 0.0

    def to_dict(self):
        return {
            "user_id":               self.user_id,
            "timestamp":             self.timestamp,
            "ensemble_score":        round(self.ensemble_score, 4),
            "severity":              self.severity.value,
            "model_scores":          {k: round(v,4) for k,v in self.model_scores.items()},
            "top_features":          {k: round(v,4) for k,v in self.top_features.items()},
            "summary":               self.summary,
            "action_required":       self.action_required,
            "flags":                 self.raw_flags,
            "ensemble_disagreement": round(self.ensemble_disagreement, 4),
        }


def _severity(s):
    if s >= 0.85: return Severity.CRITICAL
    if s >= 0.70: return Severity.HIGH
    if s >= 0.55: return Severity.MEDIUM
    return Severity.LOW


def _ensemble_disagreement(model_scores: dict) -> float:
    """
    Std deviation of the three model scores.
    High disagreement = models see different signals = flag for human review.
    """
    vals = list(model_scores.values())
    return float(np.std(vals)) if len(vals) > 1 else 0.0


# All flag rules: (feature_name, threshold, sign, human message)
# sign='+' means feature >= threshold triggers flag
# sign='-' means feature <= threshold triggers flag
_FLAG_RULES = [
    # Login / access
    ("n_unique_ips",               2.0, "+", "Multiple distinct IP addresses detected"),
    ("n_unique_countries",         1.5, "+", "Logins from multiple countries"),
    ("ip_switch_rate",             1.5, "+", "Rapid IP switching between sessions"),
    ("simultaneous_login_ratio",   1.0, "+", "Concurrent logins from different IPs"),
    ("unusual_hour_ratio",         1.5, "+", "Significant off-hours login activity (00:00-06:00)"),
    ("n_unique_devices",           2.0, "+", "Frequent device switching detected"),
    # Impossible travel
    ("impossible_travel_count",    1.0, "+", "Impossible travel: login from distant cities within minutes"),
    ("impossible_travel_min_gap",  0.1, "+", "Login gap too short for physical travel between countries"),
    # Brute force
    ("failed_login_ratio",         2.0, "+", "High rate of failed login attempts"),
    ("brute_force_pattern",        0.5, "+", "Brute-force pattern: multiple failures then success"),
    # Trading
    ("volume_spike_ratio",         2.0, "+", "Trade volume spike (>=10x baseline)"),
    ("instrument_concentration",   1.5, "+", "Single-instrument trade concentration"),
    ("min_inter_trade_min",       -1.5, "-", "Extremely rapid trade execution (latency arbitrage)"),
    ("inter_trade_regularity",     2.0, "+", "Robotic inter-trade timing regularity"),
    ("pnl_std",                    2.0, "+", "Abnormal PnL volatility"),
    ("win_rate",                   2.0, "+", "Suspiciously consistent profit rate"),
    ("trade_volume_cusum",         2.0, "+", "CUSUM regime shift detected in trade volume"),
    # Off-hours automation
    ("off_hours_trade_ratio",      2.0, "+", "Abnormally high off-hours (01:00-05:00) trading"),
    ("off_hours_interval_std",    -1.5, "-", "Robotic regularity in off-hours trading intervals"),
    # Weekend
    ("weekend_trade_ratio",        2.0, "+", "Unusual weekend/holiday trading spike"),
    # First-session expert
    ("first_session_volume_ratio", 2.0, "+", "Expert-level volume in very first trading session"),
    ("first_session_n_trades",     2.0, "+", "Abnormally high trade count in first session"),
    # News
    ("news_aligned_trade_ratio",   2.0, "+", "Trades clustered around major news events"),
    ("news_aligned_trade_count",   1.5, "+", "Multiple news-event-aligned trades detected"),
    # Financial
    ("small_deposit_ratio",        1.5, "+", "High ratio of sub-$1000 deposits (structuring)"),
    ("deposit_per_trade",          2.0, "+", "Excessive deposits relative to trading activity"),
    ("deposit_withdrawal_ratio",   2.0, "+", "Deposit-to-withdrawal ratio anomaly"),
    ("deposit_to_withdrawal_hours",-1.5,"-", "Extremely rapid deposit-to-withdrawal cycle"),
    ("dormancy_before_withdrawal_days",2.0,"+","Large withdrawal after prolonged account dormancy"),
    ("micro_deposit_count",        2.0, "+", "Micro-deposit probing pattern detected"),
    ("micro_before_large",         1.0, "+", "Tiny test deposits preceding a large transfer"),
    ("n_bonus_claims",             2.0, "+", "Abnormally high bonus claim activity"),
    ("bonus_to_deposit_ratio",     2.0, "+", "Bonus abuse: bonus exceeds normal deposit ratio"),
    # KYC
    ("kyc_before_withdrawal_count",1.0, "+", "KYC profile changes immediately before withdrawal"),
    ("kyc_interval_min",          -1.5, "-", "Rapid sequential KYC modifications"),
    ("n_kyc_events",               2.0, "+", "Unusually high number of KYC change events"),
    # Session / bot
    ("max_navigation_rate",        2.0, "+", "Bot-like navigation speed detected"),
    ("session_interval_regularity",2.0, "+", "Robotic session timing regularity"),
    ("session_duration_std",      -1.5, "-", "Suspiciously uniform session durations"),
    # Graph / network
    ("ip_hub_score",               2.0, "+", "IP shared with many other accounts (hub behaviour)"),
    ("device_hub_score",           2.0, "+", "Device shared with multiple other accounts"),
    ("shared_ip_user_count",       2.0, "+", "Multiple accounts operating from same IP"),
    ("sync_trade_ratio",           2.0, "+", "Trades synchronised with other accounts"),
    ("mirror_trade_score",         2.0, "+", "Mirror trading pattern detected across accounts"),
    ("withdrawal_cluster_score",   2.0, "+", "Withdrawals clustered with multiple other accounts"),
]

_SUMMARIES = {
    Severity.CRITICAL: (
        "CRITICAL RISK: User {uid} shows highly anomalous behaviour "
        "(ensemble score {score:.0%}). Dominant signals: {flags}. "
        "Immediate compliance review is required."
    ),
    Severity.HIGH: (
        "HIGH RISK: User {uid} displays suspicious activity "
        "(ensemble score {score:.0%}). Key signals: {flags}. "
        "Flag for compliance review within 24 hours."
    ),
    Severity.MEDIUM: (
        "MEDIUM RISK: User {uid} has elevated anomaly indicators "
        "(ensemble score {score:.0%}). Notable signals: {flags}. "
        "Monitor and review if behaviour persists."
    ),
    Severity.LOW: (
        "LOW RISK: User {uid} triggered minor anomaly signals "
        "(ensemble score {score:.0%}). Signals: {flags}. "
        "Logged for trend monitoring."
    ),
}

_ACTIONS = {
    Severity.CRITICAL: "FREEZE account pending compliance investigation. Escalate to AML team immediately.",
    Severity.HIGH:     "Place account under enhanced monitoring. Schedule compliance review within 24 h.",
    Severity.MEDIUM:   "Apply additional KYC checks. Enable step-up authentication on next login.",
    Severity.LOW:      "Log and monitor. No immediate action required.",
}


def generate_alert(user_id, ensemble_score, model_scores, top_features):
    sev   = _severity(ensemble_score)
    flags = []
    for feat, thr, sign, msg in _FLAG_RULES:
        if feat in top_features:
            val = top_features[feat]
            if sign == "+" and val >= thr:
                flags.append(msg)
            elif sign == "-" and val <= thr:
                flags.append(msg)
    flags = list(dict.fromkeys(flags))
    if not flags:
        flags = [f"Anomalous pattern in: {', '.join(list(top_features.keys())[:3])}"]

    disagreement = _ensemble_disagreement(model_scores)

    # If models strongly disagree, note it in flags
    if disagreement > 0.25:
        flags.append(f"Model disagreement detected (std={disagreement:.2f}) — recommend human review")

    summary = _SUMMARIES[sev].format(
        uid=user_id, score=ensemble_score, flags="; ".join(flags[:3]))

    return Alert(
        user_id=user_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        ensemble_score=ensemble_score,
        severity=sev,
        model_scores=model_scores,
        top_features=top_features,
        summary=summary,
        action_required=_ACTIONS[sev],
        raw_flags=flags,
        ensemble_disagreement=disagreement,
    )


def bulk_alerts(user_ids, ensemble_scores, model_scores, top_features, threshold=0.0):
    alerts = [
        generate_alert(uid, float(s), ms, tf)
        for uid, s, ms, tf in zip(user_ids, ensemble_scores, model_scores, top_features)
        if float(s) >= threshold
    ]
    return sorted(alerts, key=lambda a: a.ensemble_score, reverse=True)
