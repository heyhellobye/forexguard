"""
Synthetic forex event generator (~50 000 events, 500 users, ~10% anomalous).

Anomaly types:
  A – Rapid IP/geo switching
  B – Deposit -> minimal trade -> large withdrawal (cycling)
  C – Sudden trade-volume spike (20x)
  D – High-frequency small deposits (structuring)
  E – Bot-like navigation (regular 30s sessions)
  F – Simultaneous multi-IP logins
  G – IP hub (multiple accounts share one IP)
  H – Bonus abuse cycle
  I – Rapid KYC changes before withdrawal
  J – Multiple failed logins then success (brute force)
  K – Trades clustered around news events
  L – Dormancy + sudden large withdrawal
  M – Impossible travel (login from two distant cities within minutes)
  N – Micro-deposit probing (tiny deposits before large transfer)
  O – Off-hours automation (high activity during sleep hours, perfectly timed)
  P – First-session expert (brand new account, immediately expert trading)
"""
import random, uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (N_USERS, N_EVENTS, ANOMALY_RATIO, RANDOM_SEED,
                    RAW_DATA_PATH, LABELS_PATH, NEWS_PATH)

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

INSTRUMENTS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
               "XAUUSD","US30","NAS100","BTCUSD"]
COUNTRIES   = ["US","UK","DE","FR","IN","SG","AE","AU","JP","BR"]
DEVICES     = [f"device_{i:04d}" for i in range(2000)]
IPS_NORMAL  = [f"192.168.{rng.integers(0,255)}.{rng.integers(1,254)}" for _ in range(800)]
IPS_FOREIGN = [f"{rng.integers(10,220)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}" for _ in range(500)]
HUB_IPS     = [f"10.0.{rng.integers(0,10)}.{rng.integers(1,20)}" for _ in range(10)]

# City -> (IP prefix, country)
CITY_IPS = {
    "Mumbai":    ("103.21", "IN"),
    "London":    ("81.130", "UK"),
    "New York":  ("74.125", "US"),
    "Tokyo":     ("203.104","JP"),
    "Sydney":    ("203.30", "AU"),
    "Dubai":     ("185.93", "AE"),
}

START_TS  = datetime(2024, 1, 1)
END_TS    = datetime(2024, 6, 30)
TOTAL_SEC = int((END_TS - START_TS).total_seconds())

NEWS_EVENTS = [START_TS + timedelta(days=int(d))
               for d in rng.choice(180, 20, replace=False)]


def _ev(user_id, ts, event_type, **kw):
    row = dict(event_id=str(uuid.uuid4()), user_id=user_id, timestamp=ts,
               event_type=event_type, ip_address=None, device_id=None,
               country=None, instrument=None, trade_volume=None, lot_size=None,
               trade_direction=None, pnl=None, margin_used=None, amount=None,
               transaction_type=None, session_duration=None, page_count=None,
               navigation_rate=None, kyc_event=None, login_success=None,
               bonus_amount=None, city=None)
    row.update(kw)
    return row


def _normal(uid, n, home_ips, home_dev, home_country):
    evts, base = [], START_TS + timedelta(seconds=int(rng.integers(0, TOTAL_SEC//2)))
    for _ in range(n):
        base += timedelta(hours=float(rng.exponential(12)))
        et = rng.choice(["login","trade","deposit","withdrawal","session","kyc"],
                        p=[0.22,0.33,0.14,0.09,0.13,0.09])
        if et == "login":
            base = base.replace(hour=int(rng.integers(7,23)),
                                minute=int(rng.integers(0,60)))
            evts.append(_ev(uid, base, "login",
                ip_address=random.choice(home_ips), device_id=home_dev,
                country=home_country, login_success=True))
        elif et == "trade":
            vol = float(rng.lognormal(3.5, 0.7))
            lot = round(float(rng.uniform(0.01, 2.0)), 2)
            evts.append(_ev(uid, base, "trade",
                instrument=random.choice(INSTRUMENTS[:5]),
                trade_volume=round(vol, 2), lot_size=lot,
                trade_direction=random.choice(["BUY","SELL"]),
                pnl=round(float(rng.normal(0, vol*0.05)), 2),
                margin_used=round(lot*float(rng.uniform(50,200)), 2)))
        elif et == "deposit":
            evts.append(_ev(uid, base, "deposit",
                amount=round(float(rng.uniform(500,10000)), 2),
                transaction_type="deposit"))
        elif et == "withdrawal":
            evts.append(_ev(uid, base, "withdrawal",
                amount=round(float(rng.uniform(100,5000)), 2),
                transaction_type="withdrawal"))
        elif et == "session":
            dur   = float(rng.uniform(5,45))*60
            pages = int(rng.integers(5,40))
            evts.append(_ev(uid, base, "session",
                session_duration=round(dur,1), page_count=pages,
                navigation_rate=round(pages/(dur/60), 3)))
        elif et == "kyc":
            evts.append(_ev(uid, base, "kyc",
                kyc_event=random.choice(["document_upload","address_verify",
                                          "id_verify"])))
    return evts


# ── Anomaly injectors ─────────────────────────────────────────────────────────
def _anom_A(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(10,100)))
    for i in range(int(rng.integers(8,18))):
        ts = base + timedelta(hours=float(i)*rng.uniform(0.5,3))
        evts.append(_ev(uid, ts, "login",
            ip_address=random.choice(IPS_FOREIGN),
            device_id=random.choice(DEVICES[:50]),
            country=random.choice(COUNTRIES), login_success=True))
    return evts

def _anom_B(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(20,80)))
    for _ in range(int(rng.integers(3,6))):
        evts.append(_ev(uid, base, "deposit",
            amount=round(float(rng.uniform(20000,50000)), 2),
            transaction_type="deposit"))
        base += timedelta(hours=2)
    base += timedelta(days=2)
    evts.append(_ev(uid, base, "trade", instrument="EURUSD",
        trade_volume=0.5, lot_size=0.01, trade_direction="BUY",
        pnl=12.0, margin_used=50.0))
    base += timedelta(days=1)
    evts.append(_ev(uid, base, "withdrawal",
        amount=round(float(rng.uniform(40000,90000)), 2),
        transaction_type="withdrawal"))
    return evts

def _anom_C(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(5,40)))
    for _ in range(15):
        base += timedelta(hours=float(rng.exponential(8)))
        evts.append(_ev(uid, base, "trade", instrument="EURUSD",
            trade_volume=round(float(rng.uniform(50,200)), 2),
            lot_size=0.1, trade_direction="BUY",
            pnl=float(rng.normal(0,20)), margin_used=100.0))
    base += timedelta(days=1)
    for _ in range(20):
        base += timedelta(minutes=float(rng.uniform(1,10)))
        evts.append(_ev(uid, base, "trade",
            instrument=random.choice(INSTRUMENTS),
            trade_volume=round(float(rng.uniform(8000,25000)), 2),
            lot_size=round(float(rng.uniform(10,50)), 1),
            trade_direction=random.choice(["BUY","SELL"]),
            pnl=float(rng.normal(500,300)),
            margin_used=float(rng.uniform(5000,20000))))
    return evts

def _anom_D(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(30,100)))
    for _ in range(int(rng.integers(30,60))):
        base += timedelta(hours=float(rng.uniform(2,8)))
        evts.append(_ev(uid, base, "deposit",
            amount=round(float(rng.uniform(200,950)), 2),
            transaction_type="deposit"))
    return evts

def _anom_E(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(10,60)))
    for _ in range(int(rng.integers(40,80))):
        base += timedelta(seconds=30)
        dur   = float(rng.uniform(0.5,2))*60
        pages = int(rng.integers(80,200))
        evts.append(_ev(uid, base, "session",
            session_duration=round(dur,1), page_count=pages,
            navigation_rate=round(pages/(dur/60), 3),
            ip_address=random.choice(IPS_NORMAL[:10]),
            device_id=DEVICES[0]))
    return evts

def _anom_F(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(15,90)))
    for _ in range(int(rng.integers(5,12))):
        ts = base + timedelta(seconds=float(rng.uniform(0,60)))
        evts.append(_ev(uid, ts, "login",
            ip_address=random.choice(IPS_FOREIGN),
            device_id=random.choice(DEVICES[:100]),
            country=random.choice(COUNTRIES), login_success=True))
        base += timedelta(days=float(rng.uniform(1,5)))
    return evts

def _anom_G(uid, shared_ip):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(5,60)))
    for _ in range(int(rng.integers(10,25))):
        base += timedelta(hours=float(rng.uniform(1,12)))
        evts.append(_ev(uid, base, "login",
            ip_address=shared_ip, device_id=random.choice(DEVICES[:20]),
            country="US", login_success=True))
    return evts

def _anom_H(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(10,80)))
    for _ in range(int(rng.integers(3,7))):
        dep_amt = round(float(rng.uniform(1000,5000)), 2)
        evts.append(_ev(uid, base, "deposit",
            amount=dep_amt, transaction_type="deposit"))
        base += timedelta(hours=1)
        evts.append(_ev(uid, base, "deposit",
            amount=round(dep_amt*0.3, 2), transaction_type="bonus",
            bonus_amount=round(dep_amt*0.3, 2)))
        base += timedelta(hours=2)
        evts.append(_ev(uid, base, "trade", instrument="EURUSD",
            trade_volume=1.0, lot_size=0.01, trade_direction="BUY",
            pnl=0.5, margin_used=10.0))
        base += timedelta(hours=1)
        evts.append(_ev(uid, base, "withdrawal",
            amount=round(dep_amt*1.25, 2), transaction_type="withdrawal"))
        base += timedelta(days=float(rng.uniform(3,10)))
    return evts

def _anom_I(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(20,100)))
    for _ in range(int(rng.integers(4,8))):
        base += timedelta(hours=float(rng.uniform(0.5,3)))
        evts.append(_ev(uid, base, "kyc",
            kyc_event=random.choice(["name_change","address_change",
                                      "phone_change","email_change",
                                      "bank_account_change"])))
    base += timedelta(hours=2)
    evts.append(_ev(uid, base, "withdrawal",
        amount=round(float(rng.uniform(50000,200000)), 2),
        transaction_type="withdrawal"))
    return evts

def _anom_J(uid, home_ip):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(10,90)))
    for _ in range(int(rng.integers(8,20))):
        base += timedelta(seconds=float(rng.uniform(5,30)))
        evts.append(_ev(uid, base, "login",
            ip_address=random.choice(IPS_FOREIGN),
            device_id=random.choice(DEVICES[:200]),
            country=random.choice(COUNTRIES), login_success=False))
    base += timedelta(minutes=2)
    evts.append(_ev(uid, base, "login",
        ip_address=home_ip, device_id=random.choice(DEVICES[:200]),
        country="US", login_success=True))
    base += timedelta(minutes=5)
    evts.append(_ev(uid, base, "withdrawal",
        amount=round(float(rng.uniform(10000,50000)), 2),
        transaction_type="withdrawal"))
    return evts

def _anom_K(uid):
    evts = []
    for news_ts in random.sample(NEWS_EVENTS, k=min(8, len(NEWS_EVENTS))):
        offset = timedelta(minutes=float(rng.uniform(-3,3)))
        ts = news_ts + offset
        evts.append(_ev(uid, ts, "trade",
            instrument=random.choice(["EURUSD","GBPUSD","XAUUSD","US30"]),
            trade_volume=round(float(rng.uniform(5000,20000)), 2),
            lot_size=round(float(rng.uniform(5,30)), 1),
            trade_direction=random.choice(["BUY","SELL"]),
            pnl=round(float(rng.uniform(200,2000)), 2),
            margin_used=round(float(rng.uniform(2000,15000)), 2)))
    return evts

def _anom_L(uid):
    evts, base = [], START_TS + timedelta(days=int(rng.integers(5,20)))
    for _ in range(int(rng.integers(5,10))):
        base += timedelta(days=float(rng.uniform(1,3)))
        evts.append(_ev(uid, base, "trade", instrument="EURUSD",
            trade_volume=round(float(rng.uniform(100,500)), 2),
            lot_size=0.1, trade_direction="BUY",
            pnl=float(rng.normal(0,20)), margin_used=100.0))
    # Long dormancy 60-120 days
    base += timedelta(days=float(rng.uniform(60,120)))
    evts.append(_ev(uid, base, "withdrawal",
        amount=round(float(rng.uniform(30000,150000)), 2),
        transaction_type="withdrawal"))
    return evts

def _anom_M(uid):
    """Impossible travel: login from two geographically distant cities within 15 minutes."""
    evts, base = [], START_TS + timedelta(days=int(rng.integers(10,100)))
    city_pairs = [
        ("Mumbai", "London"), ("New York", "Tokyo"),
        ("Sydney", "Dubai"), ("London", "New York"),
    ]
    for _ in range(int(rng.integers(5,12))):
        city_a, city_b = random.choice(city_pairs)
        ip_a, country_a = CITY_IPS[city_a]
        ip_b, country_b = CITY_IPS[city_b]
        evts.append(_ev(uid, base, "login",
            ip_address=f"{ip_a}.{rng.integers(1,255)}.{rng.integers(1,255)}",
            device_id=random.choice(DEVICES[:100]),
            country=country_a, login_success=True, city=city_a))
        # Only 5-15 minutes later from a city thousands of miles away
        base += timedelta(minutes=float(rng.uniform(5,15)))
        evts.append(_ev(uid, base, "login",
            ip_address=f"{ip_b}.{rng.integers(1,255)}.{rng.integers(1,255)}",
            device_id=random.choice(DEVICES[:100]),
            country=country_b, login_success=True, city=city_b))
        base += timedelta(days=float(rng.uniform(2,8)))
    return evts

def _anom_N(uid):
    """Micro-deposit probing: many tiny deposits before a large transfer."""
    evts, base = [], START_TS + timedelta(days=int(rng.integers(20,80)))
    # Probe with tiny deposits
    for _ in range(int(rng.integers(10,25))):
        base += timedelta(minutes=float(rng.uniform(10,30)))
        evts.append(_ev(uid, base, "deposit",
            amount=round(float(rng.uniform(0.01, 5.0)), 2),
            transaction_type="deposit"))
    # Then large deposit
    base += timedelta(hours=2)
    evts.append(_ev(uid, base, "deposit",
        amount=round(float(rng.uniform(50000,200000)), 2),
        transaction_type="deposit"))
    base += timedelta(days=1)
    evts.append(_ev(uid, base, "withdrawal",
        amount=round(float(rng.uniform(45000,190000)), 2),
        transaction_type="withdrawal"))
    return evts

def _anom_O(uid):
    """Off-hours automation: heavy perfectly-timed activity during 1AM-5AM."""
    evts, base = [], START_TS + timedelta(days=int(rng.integers(5,50)))
    for _ in range(int(rng.integers(30,60))):
        # Always between 1AM and 5AM
        base += timedelta(days=1)
        hour = int(rng.integers(1, 5))
        ts   = base.replace(hour=hour, minute=0, second=0) + \
               timedelta(seconds=int(rng.integers(0, 3600)))
        # Perfectly timed — very low inter-event variance
        for j in range(int(rng.integers(5,15))):
            trade_ts = ts + timedelta(seconds=j*30)  # exactly 30s apart
            evts.append(_ev(uid, trade_ts, "trade",
                instrument=random.choice(INSTRUMENTS),
                trade_volume=round(float(rng.uniform(500,2000)), 2),
                lot_size=round(float(rng.uniform(0.5,5)), 1),
                trade_direction=random.choice(["BUY","SELL"]),
                pnl=float(rng.normal(50,20)), margin_used=500.0))
    return evts

def _anom_P(uid):
    """First-session expert: brand new account, immediately complex expert trading."""
    evts, base = [], START_TS + timedelta(days=int(rng.integers(1,5)))
    # First ever login
    evts.append(_ev(uid, base, "login",
        ip_address=random.choice(IPS_FOREIGN),
        device_id=random.choice(DEVICES[:100]),
        country=random.choice(COUNTRIES), login_success=True))
    base += timedelta(minutes=5)
    # Immediately high-volume multi-instrument expert trades
    for _ in range(int(rng.integers(20,40))):
        base += timedelta(minutes=float(rng.uniform(1,5)))
        evts.append(_ev(uid, base, "trade",
            instrument=random.choice(INSTRUMENTS),
            trade_volume=round(float(rng.uniform(5000,20000)), 2),
            lot_size=round(float(rng.uniform(5,20)), 1),
            trade_direction=random.choice(["BUY","SELL"]),
            pnl=round(float(rng.uniform(100,1000)), 2),
            margin_used=round(float(rng.uniform(2000,10000)), 2)))
    return evts


INJECTORS = [_anom_A, _anom_B, _anom_C, _anom_D, _anom_E, _anom_F,
             _anom_H, _anom_I, _anom_K, _anom_L, _anom_M, _anom_N,
             _anom_O, _anom_P]


def generate(save=True):
    n_anom   = max(10, int(N_USERS * ANOMALY_RATIO))
    anom_ids = {f"user_{i:04d}"
                for i in rng.choice(N_USERS, n_anom, replace=False)}
    all_uids = [f"user_{i:04d}" for i in range(N_USERS)]
    labels   = {uid: (1 if uid in anom_ids else 0) for uid in all_uids}

    hub_users = {}
    anom_list = list(anom_ids)
    for i, hub_ip in enumerate(HUB_IPS[:5]):
        for u in anom_list[i*8:(i+1)*8]:
            hub_users[u] = hub_ip

    eppu     = max(30, N_EVENTS // N_USERS)
    all_evts = []

    for uid in all_uids:
        home_ips  = random.sample(IPS_NORMAL, k=int(rng.integers(1,4)))
        home_dev  = random.choice(DEVICES[:500])
        home_ctry = random.choice(COUNTRIES[:6])
        n         = int(rng.integers(max(10, eppu-30), eppu+30))
        evts      = _normal(uid, n, home_ips, home_dev, home_ctry)

        if uid in anom_ids:
            if uid in hub_users:
                evts += _anom_G(uid, hub_users[uid])
            else:
                injector = random.choice(INJECTORS)
                if injector == _anom_J:
                    evts += _anom_J(uid, home_ips[0])
                else:
                    evts += injector(uid)

        all_evts.extend(evts)

    df = pd.DataFrame(all_evts)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    ldf      = pd.DataFrame(list(labels.items()), columns=["user_id","label"])
    news_df  = pd.DataFrame({"timestamp": NEWS_EVENTS})

    if save:
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        ldf.to_csv(LABELS_PATH, index=False)
        news_df.to_csv(NEWS_PATH, index=False)
        print(f"[generate] {len(df):,} events | {n_anom} anomalous | "
              f"16 anomaly types | {len(NEWS_EVENTS)} news events")

    return df, ldf, news_df


if __name__ == "__main__":
    generate()
