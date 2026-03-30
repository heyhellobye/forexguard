"""
Per-user feature engineering (~55 behavioural features) + LSTM time sequences.

Feature groups:
  - Login / access (IP, device, country, hour, simultaneous, brute-force)
  - Impossible travel (min minutes between logins from distant countries)
  - Trading (volume, spike, instruments, PnL, inter-trade timing)
  - Off-hours automation (fraction of trades between 1-5AM, inter-event regularity)
  - First-session behaviour (trades in first 24h vs later)
  - News-event alignment
  - Financial (deposits, withdrawals, structuring, bonus abuse, cycling)
  - Micro-deposit probing (tiny deposits before large transfer)
  - KYC (burst changes before withdrawal)
  - Session / bot detection
  - CUSUM regime changepoint score
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES_PATH, SEQ_BUCKET_HOURS, LSTM_SEQ_LEN, NEWS_WINDOW_MINUTES


# ── CUSUM helper ──────────────────────────────────────────────────────────────
def _cusum_score(series: pd.Series, drift: float = 0.5) -> float:
    """
    CUSUM changepoint score.
    Returns the maximum cumulative sum deviation — high value means
    the user's behaviour shifted abruptly at some point.
    """
    if len(series) < 4:
        return 0.0
    x   = series.values.astype(float)
    mu  = x.mean()
    sd  = x.std() + 1e-9
    z   = (x - mu) / sd
    s_pos, s_neg = 0.0, 0.0
    peak = 0.0
    for zi in z:
        s_pos = max(0.0, s_pos + zi - drift)
        s_neg = max(0.0, s_neg - zi - drift)
        peak  = max(peak, s_pos, s_neg)
    return float(peak)


def build_user_features(df: pd.DataFrame,
                        news_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]      = df["timestamp"].dt.hour
    df["weekday"]   = df["timestamp"].dt.weekday   # 0=Mon … 6=Sun

    news_ts = []
    if news_df is not None and len(news_df):
        news_ts = pd.to_datetime(news_df["timestamp"]).tolist()

    login_df    = df[df["event_type"] == "login"]
    trade_df    = df[df["event_type"] == "trade"]
    deposit_df  = df[df["event_type"] == "deposit"]
    withdraw_df = df[df["event_type"] == "withdrawal"]
    session_df  = df[df["event_type"] == "session"]
    kyc_df      = df[df["event_type"] == "kyc"]

    feats = []

    for uid, grp in df.groupby("user_id"):
        r = {"user_id": uid}

        # ── Login / access ───────────────────────────────────────────────────
        lg = login_df[login_df["user_id"] == uid]
        r["n_logins"]            = len(lg)
        r["n_unique_ips"]        = lg["ip_address"].nunique() if len(lg) else 0
        r["n_unique_devices"]    = lg["device_id"].nunique()  if len(lg) else 0
        r["n_unique_countries"]  = lg["country"].nunique()    if len(lg) else 0
        r["unusual_hour_ratio"]  = (lg["hour"] < 6).sum() / len(lg) if len(lg) else 0.0
        r["login_hour_std"]      = lg["hour"].std() if len(lg) > 1 else 0.0

        if len(lg) > 1:
            ls = lg.sort_values("timestamp")
            r["ip_switch_rate"] = ls["ip_address"].ne(
                ls["ip_address"].shift()).astype(int).mean()
            ls["bkt"] = ls["timestamp"].dt.floor("1min")
            r["simultaneous_login_ratio"] = (ls.groupby("bkt").size() > 1).mean()
        else:
            r["ip_switch_rate"]           = 0.0
            r["simultaneous_login_ratio"] = 0.0

        # Failed login / brute force
        if "login_success" in lg.columns:
            failed  = (lg["login_success"] == False).sum()
            r["failed_login_count"] = int(failed)
            r["failed_login_ratio"] = failed / max(len(lg), 1)
            lg_s = lg.sort_values("timestamp")
            burst_count = 0
            for _, row_i in lg_s.iterrows():
                if row_i.get("login_success") == True:
                    ws = row_i["timestamp"] - pd.Timedelta(minutes=10)
                    pf = lg_s[(lg_s["timestamp"] >= ws) &
                              (lg_s["timestamp"] < row_i["timestamp"]) &
                              (lg_s["login_success"] == False)]
                    if len(pf) >= 3:
                        burst_count += 1
            r["brute_force_pattern"] = float(burst_count)
        else:
            r["failed_login_count"] = 0
            r["failed_login_ratio"] = 0.0
            r["brute_force_pattern"] = 0.0

        # ── Impossible travel ─────────────────────────────────────────────────
        # Min minutes between consecutive logins from different countries
        if len(lg) > 1 and "country" in lg.columns:
            ls2 = lg.sort_values("timestamp").reset_index(drop=True)
            impossible_count = 0
            min_gap = 9999.0
            for i in range(1, len(ls2)):
                prev, curr = ls2.iloc[i-1], ls2.iloc[i]
                if (prev["country"] != curr["country"] and
                        pd.notna(prev["country"]) and pd.notna(curr["country"])):
                    gap_min = (curr["timestamp"] - prev["timestamp"]
                               ).total_seconds() / 60.0
                    if gap_min < 120:   # less than 2 hours = suspicious
                        impossible_count += 1
                        min_gap = min(min_gap, gap_min)
            r["impossible_travel_count"] = float(impossible_count)
            r["impossible_travel_min_gap"] = min_gap if min_gap < 9999 else 0.0
        else:
            r["impossible_travel_count"]   = 0.0
            r["impossible_travel_min_gap"] = 0.0

        # ── Trading ──────────────────────────────────────────────────────────
        tr = trade_df[trade_df["user_id"] == uid]
        r["n_trades"]               = len(tr)
        r["avg_trade_volume"]       = tr["trade_volume"].mean()  if len(tr) else 0.0
        r["trade_volume_std"]       = tr["trade_volume"].std()   if len(tr) > 1 else 0.0
        r["volume_spike_ratio"]     = (tr["trade_volume"].max() /
                                       (tr["trade_volume"].mean() + 1e-9)) if len(tr) else 0.0
        r["n_unique_instruments"]   = tr["instrument"].nunique() if len(tr) else 0
        r["instrument_concentration"] = (tr["instrument"].value_counts(normalize=True).iloc[0]
                                         if len(tr) else 0.0)
        r["avg_pnl"]                = tr["pnl"].mean()  if len(tr) else 0.0
        r["pnl_std"]                = tr["pnl"].std()   if len(tr) > 1 else 0.0
        r["win_rate"]               = (tr["pnl"] > 0).mean() if len(tr) else 0.0
        r["avg_lot_size"]           = tr["lot_size"].mean()   if len(tr) else 0.0
        r["avg_margin_used"]        = tr["margin_used"].mean() if len(tr) else 0.0

        if len(tr) > 1:
            ts_ = tr.sort_values("timestamp")
            deltas = ts_["timestamp"].diff().dt.total_seconds() / 60.0
            r["avg_inter_trade_min"]       = deltas.mean()
            r["min_inter_trade_min"]       = deltas.min()
            r["inter_trade_regularity"]    = 1.0 / (deltas.std() + 1e-9)   # high = bot-like
            r["trade_volume_cusum"]        = _cusum_score(ts_["trade_volume"].dropna())
        else:
            r["avg_inter_trade_min"]    = 0.0
            r["min_inter_trade_min"]    = 0.0
            r["inter_trade_regularity"] = 0.0
            r["trade_volume_cusum"]     = 0.0

        # Off-hours automation (1AM-5AM trades)
        if len(tr):
            off_hours = tr[tr["timestamp"].dt.hour.between(1, 5)]
            r["off_hours_trade_ratio"] = len(off_hours) / len(tr)
            # Inter-event std within off-hours sessions (low = robotic)
            if len(off_hours) > 2:
                oh_s    = off_hours.sort_values("timestamp")
                oh_gaps = oh_s["timestamp"].diff().dt.total_seconds().dropna()
                r["off_hours_interval_std"] = oh_gaps.std()
            else:
                r["off_hours_interval_std"] = 9999.0
        else:
            r["off_hours_trade_ratio"]  = 0.0
            r["off_hours_interval_std"] = 9999.0

        # Weekend trading
        if len(tr):
            weekend_trades = tr[tr["timestamp"].dt.weekday >= 5]
            r["weekend_trade_ratio"] = len(weekend_trades) / len(tr)
        else:
            r["weekend_trade_ratio"] = 0.0

        # First-session expert: high volume in first 24h vs rest
        if len(tr):
            first_ts   = tr["timestamp"].min()
            first_day  = tr[tr["timestamp"] <= first_ts + pd.Timedelta(hours=24)]
            later      = tr[tr["timestamp"] > first_ts + pd.Timedelta(hours=24)]
            first_vol  = first_day["trade_volume"].sum()
            later_vol  = later["trade_volume"].sum()
            r["first_session_volume_ratio"] = first_vol / (later_vol + 1e-9)
            r["first_session_n_trades"]     = float(len(first_day))
        else:
            r["first_session_volume_ratio"] = 0.0
            r["first_session_n_trades"]     = 0.0

        # News-event alignment
        if len(tr) and news_ts:
            window      = pd.Timedelta(minutes=NEWS_WINDOW_MINUTES)
            news_aligned = sum(
                1 for t in tr["timestamp"].tolist()
                if any(abs((t - n).total_seconds()) <= window.total_seconds()
                       for n in news_ts))
            r["news_aligned_trade_ratio"] = news_aligned / max(len(tr), 1)
            r["news_aligned_trade_count"] = float(news_aligned)
        else:
            r["news_aligned_trade_ratio"] = 0.0
            r["news_aligned_trade_count"] = 0.0

        # ── Financial ─────────────────────────────────────────────────────────
        dep = deposit_df[deposit_df["user_id"] == uid]
        wdr = withdraw_df[withdraw_df["user_id"] == uid]

        bonus_dep = dep[dep["transaction_type"] == "bonus"] if "transaction_type" in dep.columns else pd.DataFrame()
        r["n_bonus_claims"]         = len(bonus_dep)
        r["bonus_to_deposit_ratio"] = (bonus_dep["amount"].sum() /
                                        (dep["amount"].sum() + 1e-9)) if len(dep) else 0.0

        r["n_deposits"]           = len(dep)
        r["n_withdrawals"]        = len(wdr)
        r["total_deposited"]      = dep["amount"].sum()  if len(dep) else 0.0
        r["total_withdrawn"]      = wdr["amount"].sum()  if len(wdr) else 0.0
        r["avg_deposit_amount"]   = dep["amount"].mean() if len(dep) else 0.0
        r["deposit_withdrawal_ratio"] = float(r["total_deposited"]) / (float(r["total_withdrawn"]) + 1e-9)
        r["small_deposit_ratio"]  = (dep["amount"] < 1000).sum() / len(dep) if len(dep) else 0.0
        r["deposit_per_trade"]    = float(r["total_deposited"]) / (r["n_trades"] + 1e-9)

        # Micro-deposit probing: tiny deposits (< $10) before a large one
        if len(dep):
            micro_deps  = (dep["amount"] < 10.0).sum()
            large_deps  = (dep["amount"] > 10000.0).sum()
            r["micro_deposit_count"] = float(micro_deps)
            r["micro_before_large"]  = float(min(micro_deps, large_deps))
        else:
            r["micro_deposit_count"] = 0.0
            r["micro_before_large"]  = 0.0

        if len(dep) and len(wdr):
            last_dep  = dep["timestamp"].max()
            first_wdr = wdr["timestamp"].min()
            hours_gap = (first_wdr - last_dep).total_seconds() / 3600
            r["deposit_to_withdrawal_hours"] = max(hours_gap, 0.0)
        else:
            r["deposit_to_withdrawal_hours"] = 9999.0

        # Dormancy before withdrawal
        if len(wdr):
            last_active = grp[grp["event_type"].isin(["trade","login"])]
            if len(last_active):
                dormancy = (wdr["timestamp"].min() -
                            last_active["timestamp"].max()).total_seconds() / 86400
                r["dormancy_before_withdrawal_days"] = max(dormancy, 0.0)
            else:
                r["dormancy_before_withdrawal_days"] = 0.0
        else:
            r["dormancy_before_withdrawal_days"] = 0.0

        # ── KYC ──────────────────────────────────────────────────────────────
        kyc = kyc_df[kyc_df["user_id"] == uid]
        r["n_kyc_events"] = len(kyc)
        if len(kyc) and len(wdr):
            kyc_before_wdr = 0
            for _, w in wdr.iterrows():
                ws       = w["timestamp"] - pd.Timedelta(hours=48)
                rk       = kyc[(kyc["timestamp"] >= ws) &
                               (kyc["timestamp"] <= w["timestamp"])]
                if len(rk) >= 2:
                    kyc_before_wdr += 1
            r["kyc_before_withdrawal_count"] = float(kyc_before_wdr)
        else:
            r["kyc_before_withdrawal_count"] = 0.0

        if len(kyc) > 2:
            kyc_s = kyc.sort_values("timestamp")
            gaps  = kyc_s["timestamp"].diff().dt.total_seconds().dropna()
            r["kyc_interval_std"] = gaps.std()
            r["kyc_interval_min"] = gaps.min()
        else:
            r["kyc_interval_std"] = 9999.0
            r["kyc_interval_min"] = 9999.0

        # ── Session / bot ─────────────────────────────────────────────────────
        ss = session_df[session_df["user_id"] == uid]
        r["n_sessions"]            = len(ss)
        r["avg_session_duration"]  = ss["session_duration"].mean() if len(ss) else 0.0
        r["session_duration_std"]  = ss["session_duration"].std()  if len(ss) > 1 else 0.0
        r["avg_navigation_rate"]   = ss["navigation_rate"].mean()  if len(ss) else 0.0
        r["max_navigation_rate"]   = ss["navigation_rate"].max()   if len(ss) else 0.0

        # Inter-session interval regularity (bots are too regular)
        if len(ss) > 2:
            ss_s  = ss.sort_values("timestamp")
            s_gaps = ss_s["timestamp"].diff().dt.total_seconds().dropna()
            r["session_interval_regularity"] = 1.0 / (s_gaps.std() + 1e-9)
        else:
            r["session_interval_regularity"] = 0.0

        feats.append(r)

    feat_df = pd.DataFrame(feats).set_index("user_id")
    feat_df.fillna(0.0, inplace=True)

    # Clip sentinel values
    for col in ["deposit_to_withdrawal_hours","kyc_interval_std",
                "kyc_interval_min","off_hours_interval_std"]:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].clip(upper=9998)

    num = feat_df.select_dtypes(include=[np.number]).columns
    feat_df[num] = feat_df[num].apply(lambda c: (c - c.mean()) / (c.std() + 1e-9))
    feat_df.reset_index(inplace=True)

    if FEATURES_PATH:
        feat_df.to_csv(FEATURES_PATH, index=False)
        print(f"[features] {len(feat_df)} users x {len(feat_df.columns)-1} features "
              f"-> {FEATURES_PATH}")
    return feat_df


def build_user_sequences(df: pd.DataFrame,
                         user_features: pd.DataFrame) -> dict:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["bucket"]    = df["timestamp"].dt.floor(f"{SEQ_BUCKET_HOURS}h")
    feat_cols = [c for c in user_features.columns if c != "user_id"]
    n_feats   = len(feat_cols)
    seqs      = {}
    for uid in user_features["user_id"].unique():
        grp     = df[df["user_id"] == uid].sort_values("timestamp")
        base    = user_features[user_features["user_id"] == uid][feat_cols].values[0].astype(float)
        buckets = sorted(grp["bucket"].unique())
        vecs    = []
        for b in buckets:
            bdata = grp[grp["bucket"] == b]
            vec   = base.copy()
            if len(bdata):
                vec[0] = len(bdata)
            vecs.append(vec)
        if len(vecs) < LSTM_SEQ_LEN:
            vecs = [np.zeros(n_feats)] * (LSTM_SEQ_LEN - len(vecs)) + vecs
        else:
            vecs = vecs[-LSTM_SEQ_LEN:]
        seqs[uid] = np.array(vecs, dtype=np.float32)
    return seqs
