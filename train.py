"""
ForexGuard training pipeline.

Steps
-----
1. Generate / load synthetic events (now with 11 anomaly types)
2. Engineer per-user features (~45 features including KYC, failed-login, news)
3. Build cross-user graph features (IP hub, sync trades, mirror trades)
4. Merge feature matrices
5. Build LSTM sequences
6. Train Isolation Forest
7. Train LOF
8. Train LSTM Autoencoder
9. Evaluate ensemble vs ground truth
10. Save models + log to MLflow
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (RAW_DATA_PATH, LABELS_PATH, FEATURES_PATH, CROSS_FEAT_PATH,
                    NEWS_PATH, MODEL_DIR, ALERT_THRESHOLD, ENSEMBLE_WEIGHTS,
                    MLFLOW_EXPERIMENT, RANDOM_SEED)
from data.generate        import generate
from features.engineering import build_user_features, build_user_sequences
from features.cross_user  import build_cross_user_features
from models.baseline      import IsolationForestDetector, LOFDetector
from models.lstm_ae       import LSTMAEDetector

try:
    import mlflow
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False
    print("[train] MLflow not available - skipping experiment tracking")


def ens(if_s, lof_s, lstm_s):
    w = ENSEMBLE_WEIGHTS
    return (w["isolation_forest"]*if_s + w["lof"]*lof_s + w["lstm_ae"]*lstm_s) / sum(w.values())


def evaluate(scores, labels, name, threshold=ALERT_THRESHOLD):
    preds = (scores >= threshold).astype(int)
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    print(f"  [{name:25s}] AUROC={auroc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    if MLFLOW_OK:
        mlflow.log_metrics({f"{name}_auroc": auroc, f"{name}_precision": prec,
                            f"{name}_recall": rec, f"{name}_f1": f1})
    return auroc, prec, rec, f1


def train(force_regen=False):
    np.random.seed(RANDOM_SEED)

    # 1. Data
    if force_regen or not RAW_DATA_PATH.exists():
        print("\n[1/8] Generating synthetic data (11 anomaly types)...")
        events_df, labels_df, news_df = generate(save=True)
    else:
        print(f"\n[1/8] Loading cached data...")
        events_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
        labels_df = pd.read_csv(LABELS_PATH)
        news_df   = pd.read_csv(NEWS_PATH) if NEWS_PATH.exists() else pd.DataFrame()

    # 2. Per-user features (with news alignment)
    print("\n[2/8] Engineering per-user features...")
    if force_regen or not FEATURES_PATH.exists():
        feat_df = build_user_features(events_df, news_df)
    else:
        feat_df = pd.read_csv(FEATURES_PATH)

    # 3. Cross-user graph features
    print("\n[3/8] Building cross-user graph features...")
    if force_regen or not CROSS_FEAT_PATH.exists():
        cross_df = build_cross_user_features(events_df)
    else:
        cross_df = pd.read_csv(CROSS_FEAT_PATH)

    # 4. Merge
    print("\n[4/8] Merging feature matrices...")
    merged_df = feat_df.merge(cross_df, on="user_id", how="left")
    merged_df.fillna(0.0, inplace=True)

    feat_cols = [c for c in merged_df.columns if c != "user_id"]
    X         = merged_df[feat_cols].values.astype(np.float32)
    user_ids  = merged_df["user_id"].tolist()
    label_map = dict(zip(labels_df["user_id"], labels_df["label"]))
    y         = np.array([label_map.get(u, 0) for u in user_ids], dtype=int)

    print(f"  {len(user_ids)} users | {X.shape[1]} total features | "
          f"{y.sum()} anomalous ({y.mean()*100:.1f}%)")

    # Save merged features for API
    merged_df.to_csv(FEATURES_PATH, index=False)

    # 5. LSTM sequences
    print("\n[5/8] Building LSTM sequences...")
    sequences = build_user_sequences(events_df, merged_df)

    if MLFLOW_OK:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run = mlflow.start_run(run_name="forexguard_train")
        run.__enter__()
        mlflow.log_params({"n_users": len(user_ids), "n_features": X.shape[1],
                           "anomaly_ratio": float(y.mean()),
                           "anomaly_types": 11})

    # 6. Isolation Forest
    print("\n[6/8] Training Isolation Forest...")
    if_det    = IsolationForestDetector()
    if_det.fit(X, feature_names=feat_cols)
    if_scores = if_det.score(X)
    if_det.save()
    evaluate(if_scores, y, "isolation_forest")

    # 7. LOF
    print("\n[7/8] Training LOF...")
    lof_det    = LOFDetector()
    lof_det.fit(X, feature_names=feat_cols)
    lof_scores = lof_det.score(X)
    lof_det.save()
    evaluate(lof_scores, y, "lof")

    # 8. LSTM AE (normal users only)
    print("\n[8/8] Training LSTM Autoencoder...")
    normal_uids = [u for u, l in label_map.items() if l == 0]
    normal_seqs = {u: sequences[u] for u in normal_uids if u in sequences}
    lstm_det    = LSTMAEDetector()
    lstm_det.fit(normal_seqs, feature_names=feat_cols)
    lstm_score_map = lstm_det.score(sequences)
    lstm_scores    = np.array([lstm_score_map.get(u, 0.0) for u in user_ids],
                               dtype=np.float32)
    lstm_det.save()
    evaluate(lstm_scores, y, "lstm_ae")

    # Ensemble
    print("\n── Ensemble ─────────────────────────────────────────────")
    ens_scores = np.array([ens(f, l, lt)
                           for f, l, lt in zip(if_scores, lof_scores, lstm_scores)])
    auroc, _, _, f1 = evaluate(ens_scores, y, "ensemble")

    summary = {
        "n_users":        len(user_ids),
        "n_features":     X.shape[1],
        "n_anomalous":    int(y.sum()),
        "anomaly_types":  11,
        "ensemble_auroc": round(auroc, 4),
        "ensemble_f1":    round(f1, 4),
        "models":         ["isolation_forest", "lof", "lstm_ae"],
        "new_features": [
            "failed_login_count", "failed_login_ratio", "brute_force_pattern",
            "news_aligned_trade_ratio", "news_aligned_trade_count",
            "n_bonus_claims", "bonus_to_deposit_ratio",
            "deposit_to_withdrawal_hours", "n_kyc_events",
            "kyc_before_withdrawal_count", "kyc_interval_std", "kyc_interval_min",
            "ip_hub_score", "device_hub_score", "sync_trade_ratio",
            "mirror_trade_score", "shared_ip_user_count"
        ]
    }
    sp = MODEL_DIR / "training_summary.json"
    sp.write_text(json.dumps(summary, indent=2))

    if MLFLOW_OK:
        mlflow.log_artifact(str(sp))
        run.__exit__(None, None, None)

    print(f"\n✅  Done. {X.shape[1]} features | Models -> {MODEL_DIR}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="Regenerate data")
    args = ap.parse_args()
    train(force_regen=args.regen)
