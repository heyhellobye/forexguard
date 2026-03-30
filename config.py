from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DATA_PATH    = BASE_DIR / "data" / "events.csv"
LABELS_PATH      = BASE_DIR / "data" / "labels.csv"
FEATURES_PATH    = BASE_DIR / "data" / "features.csv"
CROSS_FEAT_PATH  = BASE_DIR / "data" / "cross_features.csv"
NEWS_PATH        = BASE_DIR / "data" / "news_events.csv"
MODEL_DIR        = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

N_USERS       = 500
N_EVENTS      = 50_000
ANOMALY_RATIO = 0.08
RANDOM_SEED   = 42

SEQ_BUCKET_HOURS = 4

IF_CONTAMINATION = 0.08
IF_N_ESTIMATORS  = 200

LOF_N_NEIGHBORS   = 20
LOF_CONTAMINATION = 0.08

LSTM_SEQ_LEN       = 12
LSTM_HIDDEN        = 64
LSTM_LATENT        = 16
LSTM_LAYERS        = 2
LSTM_DROPOUT       = 0.2
LSTM_EPOCHS        = 40
LSTM_BATCH         = 64
LSTM_LR            = 1e-3
LSTM_THRESHOLD_PCT = 95

ENSEMBLE_WEIGHTS = {"isolation_forest": 0.35, "lof": 0.25, "lstm_ae": 0.40}
ALERT_THRESHOLD  = 0.60

API_HOST = "0.0.0.0"
API_PORT = 8000

MLFLOW_EXPERIMENT = "forexguard-anomaly-detection"

# News event window: trades within this many minutes of a news event are flagged
NEWS_WINDOW_MINUTES = 5
