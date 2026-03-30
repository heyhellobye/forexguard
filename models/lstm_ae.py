"""
LSTM Autoencoder (PyTorch).
Trained on normal users only. High reconstruction error → anomaly.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LATENT, LSTM_LAYERS,
                    LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH, LSTM_LR,
                    LSTM_THRESHOLD_PCT, MODEL_DIR, RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class _LSTMAEModule(nn.Module):
    def __init__(self, input_size, hidden, latent, layers, dropout):
        super().__init__()
        self.input_size = input_size
        d = dropout if layers > 1 else 0
        self.encoder = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True, dropout=d)
        self.fc_enc  = nn.Linear(hidden, latent)
        self.fc_dec  = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, input_size, num_layers=layers, batch_first=True, dropout=d)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z        = self.fc_enc(h[-1])
        dec_in   = self.fc_dec(z.unsqueeze(1).expand(-1, x.size(1), -1))
        recon, _ = self.decoder(dec_in)
        return recon

class LSTMAEDetector:
    name = "lstm_ae"
    def __init__(self, input_size=None):
        self.input_size    = input_size
        self._module       = None
        self._threshold    = 0.5
        self._train_errors = None
        self._feature_names= None
        self._trained      = False

    def _build(self):
        self._module = _LSTMAEModule(self.input_size, LSTM_HIDDEN, LSTM_LATENT,
                                     LSTM_LAYERS, LSTM_DROPOUT).to(DEVICE)

    def fit(self, sequences: dict, feature_names=None):
        self._feature_names = feature_names
        X = np.stack(list(sequences.values()), axis=0).astype(np.float32)
        if self.input_size is None:
            self.input_size = X.shape[2]
        self._build()
        loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                            batch_size=LSTM_BATCH, shuffle=True, drop_last=False)
        optim  = torch.optim.Adam(self._module.parameters(), lr=LSTM_LR)
        crit   = nn.MSELoss()
        self._module.train()
        for epoch in range(LSTM_EPOCHS):
            total = 0.0
            for (b,) in loader:
                b = b.to(DEVICE)
                loss = crit(self._module(b), b)
                optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), 1.0)
                optim.step()
                total += loss.item() * len(b)
            if (epoch + 1) % 10 == 0:
                print(f"  [lstm_ae] epoch {epoch+1:3d}/{LSTM_EPOCHS} loss={total/len(X):.6f}")
        self._train_errors = self._recon_errors(X)
        self._threshold    = float(np.percentile(self._train_errors, LSTM_THRESHOLD_PCT))
        print(f"  [lstm_ae] threshold (p{LSTM_THRESHOLD_PCT}) = {self._threshold:.6f}")
        self._trained = True
        return self

    def _recon_errors(self, X):
        self._module.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(X), LSTM_BATCH):
                b = torch.from_numpy(X[i:i+LSTM_BATCH]).to(DEVICE)
                e = ((self._module(b) - b)**2).mean(dim=(1,2)).cpu().numpy()
                errs.extend(e.tolist())
        return np.array(errs, dtype=np.float32)

    def score(self, sequences: dict) -> dict:
        X = np.stack(list(sequences.values()), axis=0).astype(np.float32)
        e = self._recon_errors(X)
        lo = self._train_errors.min() if self._train_errors is not None else e.min()
        hi = self._train_errors.max() if self._train_errors is not None else e.max()
        norm = np.clip((e - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        return {uid: float(s) for uid, s in zip(sequences.keys(), norm)}

    def predict(self, sequences: dict, threshold=None) -> dict:
        thr = threshold or self._threshold
        return {uid: int(s >= thr) for uid, s in self.score(sequences).items()}

    def explain(self, sequences: dict, top_k=5) -> dict:
        self._module.eval()
        fn  = self._feature_names or [f"feat_{i}" for i in range(self.input_size)]
        out = {}
        for uid, seq in sequences.items():
            x = torch.from_numpy(seq[None].astype(np.float32)).to(DEVICE)
            with torch.no_grad():
                fe = ((self._module(x) - x)**2).mean(dim=1).squeeze().cpu().numpy()
            idx = np.argsort(fe)[::-1][:top_k]
            out[uid] = {fn[i]: round(float(fe[i]), 6) for i in idx}
        return out

    def save(self, path=None):
        path = path or MODEL_DIR / "lstm_ae.pt"
        torch.save({"input_size": self.input_size, "state_dict": self._module.state_dict(),
                    "threshold": self._threshold, "train_errors": self._train_errors,
                    "feature_names": self._feature_names}, path)
        print(f"[lstm_ae] saved → {path}")

    @classmethod
    def load(cls, path=None):
        path   = path or MODEL_DIR / "lstm_ae.pt"
        bundle = torch.load(path, map_location=DEVICE, weights_only=False)
        obj    = cls(input_size=bundle["input_size"])
        obj._build()
        obj._module.load_state_dict(bundle["state_dict"])
        obj._module.eval()
        obj._threshold     = bundle["threshold"]
        obj._train_errors  = bundle["train_errors"]
        obj._feature_names = bundle["feature_names"]
        obj._trained       = True
        return obj
