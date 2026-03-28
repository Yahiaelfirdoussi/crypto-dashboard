"""
src/features.py
Shared feature engineering — used by both train.py and app.py
"""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators and z-score normalised versions."""
    df = df.copy()
    close = df["Close"]

    df["RSI_14"]      = RSIIndicator(close=close, window=14).rsi()
    macd              = MACD(close=close)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"]   = macd.macd_diff()
    df["SMA_50"]      = SMAIndicator(close=close, window=50).sma_indicator()
    df["SMA_200"]     = SMAIndicator(close=close, window=200).sma_indicator()
    df["EMA_21"]      = EMAIndicator(close=close, window=21).ema_indicator()
    bb                = BollingerBands(close=close, window=20)
    df["BOLL_MID_20"] = bb.bollinger_mavg()
    df["BOLL_UP_20"]  = bb.bollinger_hband()
    df["BOLL_LOW_20"] = bb.bollinger_lband()
    df["RET_1D"]      = close.pct_change()
    df["LOGRET_1D"]   = np.log(close / close.shift(1))
    df["VOLAT_14D"]   = df["RET_1D"].rolling(14).std()

    z_cols = [
        "RSI_14","MACD","MACD_signal","SMA_50","SMA_200","EMA_21",
        "BOLL_MID_20","BOLL_UP_20","BOLL_LOW_20",
        "RET_1D","LOGRET_1D","VOLAT_14D",
    ]
    for c in z_cols:
        if c in df.columns:
            df[f"Z_{c}"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)

    return df


BASE_FEATURES = [
    "RSI_14","MACD","MACD_signal","MACD_diff",
    "SMA_50","SMA_200","EMA_21",
    "BOLL_MID_20","BOLL_UP_20","BOLL_LOW_20",
    "RET_1D","LOGRET_1D","VOLAT_14D",
    "Z_RSI_14","Z_MACD","Z_MACD_signal","Z_SMA_50","Z_SMA_200","Z_EMA_21",
    "Z_BOLL_MID_20","Z_BOLL_UP_20","Z_BOLL_LOW_20",
    "Z_RET_1D","Z_LOGRET_1D","Z_VOLAT_14D",
]
