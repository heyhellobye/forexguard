"""
Cross-user (graph / network-level) features.

  - ip_hub_score            : other users sharing same IPs
  - device_hub_score        : other users sharing same devices
  - sync_trade_ratio        : trades within 30s of another user same instrument
  - mirror_trade_score      : same instrument + direction + timing
  - shared_ip_user_count    : raw count of co-users on same IP
  - withdrawal_cluster_score: withdrawals within 24h of 2+ other users
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROSS_FEAT_PATH


def build_cross_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    login_df = df[df["event_type"] == "login"].copy()
    trade_df = df[df["event_type"] == "trade"].copy()
    wdr_df   = df[df["event_type"] == "withdrawal"].copy()
    all_uids = df["user_id"].unique().tolist()

    cross = {uid: {
        "ip_hub_score":             0.0,
        "device_hub_score":         0.0,
        "sync_trade_ratio":         0.0,
        "mirror_trade_score":       0.0,
        "shared_ip_user_count":     0.0,
        "withdrawal_cluster_score": 0.0,
    } for uid in all_uids}

    # ── IP hub score ──────────────────────────────────────────────────────────
    if len(login_df):
        ip_to_users = login_df.groupby("ip_address")["user_id"].apply(set).to_dict()
        for uid in all_uids:
            user_ips    = set(login_df[login_df["user_id"] == uid]["ip_address"].dropna())
            shared      = set()
            for ip in user_ips:
                if ip in ip_to_users:
                    shared |= ip_to_users[ip]
            shared.discard(uid)
            cross[uid]["ip_hub_score"]         = float(len(shared))
            cross[uid]["shared_ip_user_count"] = float(len(shared))

    # ── Device hub score ──────────────────────────────────────────────────────
    if len(login_df):
        dev_to_users = login_df.groupby("device_id")["user_id"].apply(set).to_dict()
        for uid in all_uids:
            user_devs = set(login_df[login_df["user_id"] == uid]["device_id"].dropna())
            shared    = set()
            for dev in user_devs:
                if dev in dev_to_users:
                    shared |= dev_to_users[dev]
            shared.discard(uid)
            cross[uid]["device_hub_score"] = float(len(shared))

    # ── Sync + mirror trades ──────────────────────────────────────────────────
    if len(trade_df):
        tdf = trade_df.sort_values("timestamp")
        for uid in all_uids:
            user_tr  = tdf[tdf["user_id"] == uid]
            other_tr = tdf[tdf["user_id"] != uid]
            if len(user_tr) == 0:
                continue
            window       = pd.Timedelta(seconds=30)
            sync_count   = 0
            mirror_count = 0
            for _, tr in user_tr.iterrows():
                same_inst = other_tr[other_tr["instrument"] == tr["instrument"]]
                nearby    = same_inst[
                    (same_inst["timestamp"] >= tr["timestamp"] - window) &
                    (same_inst["timestamp"] <= tr["timestamp"] + window)]
                if len(nearby):
                    sync_count += 1
                mirrors = nearby[nearby["trade_direction"] == tr["trade_direction"]]
                if len(mirrors):
                    mirror_count += 1
            n = max(len(user_tr), 1)
            cross[uid]["sync_trade_ratio"]   = sync_count / n
            cross[uid]["mirror_trade_score"] = mirror_count / n

    # ── Withdrawal clustering ─────────────────────────────────────────────────
    if len(wdr_df):
        wdf = wdr_df.sort_values("timestamp")
        for uid in all_uids:
            user_wdrs  = wdf[wdf["user_id"] == uid]
            other_wdrs = wdf[wdf["user_id"] != uid]
            if len(user_wdrs) == 0:
                continue
            window      = pd.Timedelta(hours=24)
            cluster_cnt = 0
            for _, w in user_wdrs.iterrows():
                nearby = other_wdrs[
                    (other_wdrs["timestamp"] >= w["timestamp"] - window) &
                    (other_wdrs["timestamp"] <= w["timestamp"] + window)]
                if len(nearby) >= 2:
                    cluster_cnt += 1
            cross[uid]["withdrawal_cluster_score"] = (
                cluster_cnt / max(len(user_wdrs), 1))

    cross_df = pd.DataFrame.from_dict(cross, orient="index")
    cross_df.index.name = "user_id"

    for col in cross_df.columns:
        mu = cross_df[col].mean()
        sd = cross_df[col].std()
        cross_df[col] = (cross_df[col] - mu) / (sd + 1e-9)

    cross_df.reset_index(inplace=True)
    cross_df.to_csv(CROSS_FEAT_PATH, index=False)
    print(f"[cross_user] {len(cross_df)} users x {len(cross_df.columns)-1} features "
          f"-> {CROSS_FEAT_PATH}")
    return cross_df
