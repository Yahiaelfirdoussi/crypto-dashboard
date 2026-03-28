"""
scripts/train.py
Trains Quantile Regression + LSTM to predict next-day LOG RETURN.

Why log returns?
  - Stationary: no unit root, safe for regression
  - Additive: multi-period returns sum cleanly
  - Symmetric: gains/losses on same scale
  - Directly tradeable: sign = direction, magnitude = size

Target: log_return_next = log(Close_{t+1} / Close_t)  [forward-shifted by 1]

Usage:
    python3 scripts/train.py
    python3 scripts/train.py --tickers BTC-USD
    python3 scripts/train.py --years 4
"""
import argparse
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler   # StandardScaler for returns (not MinMax)

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from tensorflow import keras
from tensorflow.keras import layers, callbacks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import compute_indicators, BASE_FEATURES

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"}

MODEL_NAMES = {
    "BTC-USD": {
        "quantile": "quantile_regression_btc.pkl",
        "features": "selected_features_btc.pkl",
        "lstm":     "best_lstm_model_btc.keras",
        "scaler_X": "scaler_X_btc.save",
        "scaler_y": "scaler_y_btc.save",
    },
    "ETH-USD": {
        "quantile": "quant_reg_eth.pkl",
        "features": "selected_features_eth.pkl",
        "lstm":     "best_lstm_model_eth.keras",
        "scaler_X": "scaler_X_eth.save",
        "scaler_y": "scaler_y_eth.save",
    },
}

SEQ_LENGTH = 30    # LSTM lookback (30 days — returns have less autocorrelation than prices)
QUANTILE   = 0.5   # median — for point forecast; train 0.1/0.9 for intervals if needed
TEST_RATIO = 0.20  # chronological split — never shuffle return series

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(obj, filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if filename.endswith(".pkl"):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif filename.endswith(".save"):
        joblib.dump(obj, path)
    elif filename.endswith(".keras"):
        obj.save(path)
    print(f"  Saved → models/{filename}")


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Direction accuracy — most important metric for trading
    direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    # Information coefficient (rank correlation) — standard in quant finance
    from scipy.stats import spearmanr
    ic, _ = spearmanr(y_true, y_pred)

    print(f"  MAE (return):       {mae:.6f}  ({mae*100:.4f}%)")
    print(f"  RMSE (return):      {rmse:.6f}  ({rmse*100:.4f}%)")
    print(f"  R²:                 {r2:.4f}")
    print(f"  Direction accuracy: {direction_acc:.2f}%   ← key trading metric")
    print(f"  IC (Spearman ρ):    {ic:.4f}              ← >0.05 is useful in practice")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "DirAcc": direction_acc, "IC": ic}


# ── Data ──────────────────────────────────────────────────────────────────────

def download(ticker: str, years: int) -> pd.DataFrame:
    end   = pd.Timestamp.today()
    start = end - pd.Timedelta(days=years * 365)
    print(f"  Downloading {ticker} ({years}y)…")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = compute_indicators(df)

    # ── Target: NEXT DAY log return (shift -1 so row t holds tomorrow's return)
    df["target_logret"] = np.log(df["Close"] / df["Close"].shift(1)).shift(-1)

    df = df.dropna()
    print(f"  {len(df)} rows · target = next-day log return")
    print(f"  Return stats: mean={df['target_logret'].mean():.5f}  "
          f"std={df['target_logret'].std():.5f}  "
          f"skew={df['target_logret'].skew():.3f}")
    return df


# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(df: pd.DataFrame, n: int = 15) -> list:
    """Random Forest feature importance on the return target."""
    candidates = [c for c in BASE_FEATURES if c in df.columns]
    X = df[candidates].dropna()
    y = df["target_logret"].loc[X.index]

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp   = pd.Series(rf.feature_importances_, index=candidates).sort_values(ascending=False)
    feats = imp.head(n).index.tolist()
    print(f"  Top features: {feats[:6]}")
    return feats


# ── Quantile regression ───────────────────────────────────────────────────────

def train_quantile(df: pd.DataFrame, ticker: str) -> tuple:
    print("\n── Quantile Regression (target = next-day log return) ──")
    feats = select_features(df)

    X = sm.add_constant(df[feats])
    y = df["target_logret"]

    split = int(len(X) * (1 - TEST_RATIO))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model  = QuantReg(y_tr, X_tr).fit(q=QUANTILE, max_iter=5000)
    y_pred = np.squeeze(model.predict(X_te))

    print("  Test metrics:")
    metrics = print_metrics(y_te.values, y_pred)

    names = MODEL_NAMES[ticker]
    save(model, names["quantile"])
    save(feats, names["features"])
    return metrics, feats


# ── LSTM ──────────────────────────────────────────────────────────────────────

def build_lstm(seq_len: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_features))
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(32)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)   # linear — returns are unbounded
    m   = keras.Model(inp, out)
    # Huber loss — more robust to outlier return spikes than MSE
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.Huber(delta=0.01))
    return m


def make_sequences(X: np.ndarray, y: np.ndarray, seq: int):
    Xs, ys = [], []
    for i in range(seq, len(X)):
        Xs.append(X[i - seq:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(df: pd.DataFrame, ticker: str, feats: list) -> dict:
    print("\n── LSTM (target = next-day log return) ──")

    X_raw = df[feats].values
    y_raw = df["target_logret"].values

    # StandardScaler for features AND returns — returns are ~N(0, σ)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_sc     = scaler_X.fit_transform(X_raw)
    y_sc     = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    X_seq, y_seq  = make_sequences(X_sc, y_sc, SEQ_LENGTH)
    split         = int(len(X_seq) * (1 - TEST_RATIO))
    X_tr, X_te    = X_seq[:split], X_seq[split:]
    y_tr, y_te    = y_seq[:split], y_seq[split:]

    model = build_lstm(SEQ_LENGTH, len(feats))
    cb = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6, verbose=0),
    ]
    print("  Training…")
    model.fit(X_tr, y_tr,
              validation_split=0.1,
              epochs=150,
              batch_size=32,
              callbacks=cb,
              verbose=0)

    y_pred_sc = model.predict(X_te, verbose=0).flatten()
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
    y_true    = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()

    print("  Test metrics:")
    metrics = print_metrics(y_true, y_pred)

    names = MODEL_NAMES[ticker]
    save(model,    names["lstm"])
    save(scaler_X, names["scaler_X"])
    save(scaler_y, names["scaler_y"])
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def train_ticker(ticker: str, years: int):
    name = TICKERS[ticker]
    print(f"\n{'='*60}")
    print(f"  {name} ({ticker})")
    print(f"{'='*60}")

    df = download(ticker, years)
    q_metrics, feats = train_quantile(df, ticker)
    l_metrics        = train_lstm(df, ticker, feats)

    print(f"\n  ✓ {name} complete")
    print(f"  Quantile — DirAcc {q_metrics['DirAcc']:.1f}% | IC {q_metrics['IC']:.4f}")
    print(f"  LSTM     — DirAcc {l_metrics['DirAcc']:.1f}% | IC {l_metrics['IC']:.4f}")
    return {"quantile": q_metrics, "lstm": l_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS.keys()))
    parser.add_argument("--years",   type=int,  default=3)
    args = parser.parse_args()

    for ticker in args.tickers:
        if ticker not in TICKERS:
            print(f"Unknown ticker: {ticker}")
            continue
        train_ticker(ticker, args.years)

    print(f"\n{'='*60}")
    print("  Models saved to models/")
    print("  Run: streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
