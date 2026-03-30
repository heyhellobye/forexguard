"""
Baseline anomaly detectors: IsolationForestDetector + LOFDetector.
Both expose: fit(X), score(X), predict(X), explain(X), save(), load()
"""
import warnings
from pathlib import Path
import joblib
import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IF_CONTAMINATION, IF_N_ESTIMATORS, LOF_N_NEIGHBORS, LOF_CONTAMINATION, MODEL_DIR, RANDOM_SEED
warnings.filterwarnings("ignore")

class IsolationForestDetector:
    name = "isolation_forest"
    def __init__(self):
        self.model  = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION,
                                      max_samples="auto", random_state=RANDOM_SEED, n_jobs=-1)
        self.scaler = MinMaxScaler()
        self.explainer = None
        self._feature_names = None
        self._trained = False

    def fit(self, X, feature_names=None):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._feature_names = feature_names
        sample = Xs[np.random.choice(len(Xs), min(512, len(Xs)), replace=False)]
        self.explainer = shap.TreeExplainer(self.model, sample, feature_perturbation="interventional")
        self._trained = True
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        inv = -self.model.score_samples(Xs)
        lo, hi = inv.min(), inv.max()
        return (inv - lo) / (hi - lo + 1e-9)

    def predict(self, X, threshold=0.5):
        return (self.score(X) >= threshold).astype(int)

    def explain(self, X, top_k=5):
        Xs = self.scaler.transform(X)
        fn = self._feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        out = []
        if self.explainer is not None:
            try:
                sv = self.explainer.shap_values(Xs)
                for row in sv:
                    idx = np.argsort(np.abs(row))[::-1][:top_k]
                    out.append({fn[i]: round(float(row[i]), 4) for i in idx})
                return out
            except Exception:
                pass
        for row in Xs:
            idx = np.argsort(np.abs(row))[::-1][:top_k]
            out.append({fn[i]: round(float(row[i]), 4) for i in idx})
        return out

    def save(self, path=None):
        path = path or MODEL_DIR / "isolation_forest.pkl"
        joblib.dump({"model": self.model, "scaler": self.scaler, "feature_names": self._feature_names}, path)
        print(f"[isolation_forest] saved → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_DIR / "isolation_forest.pkl"
        obj  = cls()
        b    = joblib.load(path)
        obj.model, obj.scaler, obj._feature_names = b["model"], b["scaler"], b["feature_names"]
        obj._trained = True
        return obj


class LOFDetector:
    name = "lof"
    def __init__(self):
        self.model  = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, contamination=LOF_CONTAMINATION,
                                         novelty=True, n_jobs=-1)
        self.scaler = MinMaxScaler()
        self._feature_names = None
        self._trained = False

    def fit(self, X, feature_names=None):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._feature_names = feature_names
        self._trained = True
        return self

    def score(self, X):
        Xs  = self.scaler.transform(X)
        inv = -self.model.score_samples(Xs)
        lo, hi = inv.min(), inv.max()
        return (inv - lo) / (hi - lo + 1e-9)

    def predict(self, X, threshold=0.5):
        return (self.score(X) >= threshold).astype(int)

    def explain(self, X, top_k=5):
        Xs  = self.scaler.transform(X)
        bs  = -self.model.score_samples(Xs)
        fn  = self._feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        out = []
        for idx in range(len(Xs)):
            row = Xs[[idx]]
            contribs = {}
            for fi, name in enumerate(fn):
                p = row.copy(); p[0, fi] = 0.0
                ps = float(-self.model.score_samples(p)[0])
                contribs[name] = round(float(bs[idx]) - ps, 4)
            top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
            out.append(dict(top))
        return out

    def save(self, path=None):
        path = path or MODEL_DIR / "lof.pkl"
        joblib.dump({"model": self.model, "scaler": self.scaler, "feature_names": self._feature_names}, path)
        print(f"[lof] saved → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_DIR / "lof.pkl"
        obj  = cls()
        b    = joblib.load(path)
        obj.model, obj.scaler, obj._feature_names = b["model"], b["scaler"], b["feature_names"]
        obj._trained = True
        return obj
