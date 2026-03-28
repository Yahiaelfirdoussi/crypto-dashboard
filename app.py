"""
Crypto Analytics Dashboard
Predicts next-day LOG RETURN — stationary, financially meaningful.

Display logic:
  - Predicted return  → bar chart (sign = direction, height = conviction)
  - Signal            → BUY / SELL / NEUTRAL with threshold
  - Cumulative return → actual vs model strategy backtest
  - Reconstructed price (optional) → for visual reference only
"""
import os
import gc
import pickle
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objs as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

MODEL_FILES = {
    "Bitcoin": {
        "quantile": "quantile_regression_btc.pkl",
        "features": "selected_features_btc.pkl",
        "lstm":     "best_lstm_model_btc.keras",
        "scaler_X": "scaler_X_btc.save",
        "scaler_y": "scaler_y_btc.save",
    },
    "Ethereum": {
        "quantile": "quant_reg_eth.pkl",
        "features": "selected_features_eth.pkl",
        "lstm":     "best_lstm_model_eth.keras",
        "scaler_X": "scaler_X_eth.save",
        "scaler_y": "scaler_y_eth.save",
    },
}

CRYPTOS = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}


def mpath(filename: str) -> str:
    p = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Model file not found: {p}\n"
            "Run: python3 scripts/train.py"
        )
    return p


# ── Feature engineering ───────────────────────────────────────────────────────

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
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


def build_feature_matrix(df: pd.DataFrame, features: list, ticker: str) -> pd.DataFrame:
    df = df.copy()
    for base in ["RET_1D", "VOLAT_14D"]:
        zbase  = f"Z_{base}"
        zpatch = f"Z_{base}_{ticker}"
        if zpatch in features and zpatch not in df.columns and zbase in df.columns:
            df[zpatch] = df[zbase]
    df["const"] = 1.0
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    return df[features]


# ── Signal logic ──────────────────────────────────────────────────────────────

def classify_signal(ret: float, threshold: float) -> str:
    if ret > threshold:
        return "BUY"
    elif ret < -threshold:
        return "SELL"
    return "NEUTRAL"


def signal_color(sig: str) -> str:
    return {"BUY": "#1D9E75", "SELL": "#E24B4A", "NEUTRAL": "#888780"}[sig]


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(actual_returns: np.ndarray, predicted_returns: np.ndarray,
             threshold: float) -> pd.Series:
    """Long if predicted > threshold, short if < -threshold, flat otherwise."""
    position = np.where(predicted_returns > threshold,  1,
               np.where(predicted_returns < -threshold, -1, 0))
    strategy_returns = position * actual_returns
    return pd.Series(strategy_returns)


# ── Model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_quantile(name: str):
    files = MODEL_FILES[name]
    with open(mpath(files["quantile"]), "rb") as f:
        model = pickle.load(f)
    with open(mpath(files["features"]), "rb") as f:
        feats = pickle.load(f)
    return model, feats


@st.cache_resource(show_spinner=False)
def load_lstm(name: str):
    from tensorflow import keras as _keras
    files    = MODEL_FILES[name]
    model    = _keras.models.load_model(mpath(files["lstm"]))
    scaler_X = joblib.load(mpath(files["scaler_X"]))
    scaler_y = joblib.load(mpath(files["scaler_y"]))
    with open(mpath(files["features"]), "rb") as f:
        feats = pickle.load(f)
    gc.collect()
    return model, scaler_X, scaler_y, feats


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = flatten_columns(df)
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        for cand in [f"{base}_{ticker}", f"{ticker}_{base}", base]:
            if cand in df.columns:
                df[base] = df[cand]
                break
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = compute_indicators(df)
    # Actual next-day log return (same target the model was trained on)
    df["actual_logret"] = np.log(df["Close"] / df["Close"].shift(1)).shift(-1)
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────

LAYOUT = dict(
    xaxis_title="Date",
    legend=dict(orientation="h", y=-0.2),
    height=400,
    margin=dict(l=40, r=20, t=40, b=60),
    hovermode="x unified",
)


def candle_fig(df, name):
    fig = go.Figure([
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="OHLC",
        ),
        go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color="rgba(120,120,220,0.25)", yaxis="y2",
        ),
    ])
    fig.update_layout(
        **LAYOUT,
        yaxis_title="Price (USD)",
        title=f"{name} — candlestick & volume",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
        xaxis_rangeslider_visible=False,
    )
    return fig


def return_bar_fig(idx, actual, predicted, name, label):
    """Bar chart: predicted return vs actual return."""
    colors = ["#1D9E75" if r > 0 else "#E24B4A" for r in predicted]
    fig = go.Figure([
        go.Bar(x=idx, y=predicted * 100, name="Predicted return (%)",
               marker_color=colors, opacity=0.8),
        go.Scatter(x=idx, y=actual * 100, mode="lines",
                   name="Actual return (%)",
                   line=dict(color="#7F77DD", width=1.5)),
    ])
    fig.update_layout(
        **LAYOUT,
        yaxis_title="Log return (%)",
        title=f"{name} — predicted vs actual next-day return ({label})",
    )
    return fig


def cumret_fig(idx, actual, predicted, threshold, name):
    """Cumulative return: buy-and-hold vs model strategy."""
    strat = backtest(actual, predicted, threshold)
    cum_bnh  = np.exp(np.cumsum(actual))   - 1
    cum_strat = np.exp(np.cumsum(strat.values)) - 1

    fig = go.Figure([
        go.Scatter(x=idx, y=cum_bnh * 100,   mode="lines",
                   name="Buy & hold (%)",
                   line=dict(color="#888780", width=1.5, dash="dot")),
        go.Scatter(x=idx, y=cum_strat * 100, mode="lines",
                   name="Model strategy (%)",
                   line=dict(color="#7F77DD", width=2)),
    ])
    fig.update_layout(
        **LAYOUT,
        yaxis_title="Cumulative return (%)",
        title=f"{name} — model strategy vs buy-and-hold",
    )
    return fig


def signal_table(df_pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df_pred.tail(10).copy()
    df["Signal"]          = df["pred_return"].apply(lambda r: classify_signal(r, threshold))
    df["Predicted (%)"]   = (df["pred_return"] * 100).round(4)
    df["Actual (%)"]      = (df["actual_logret"] * 100).round(4)
    df["Direction correct"] = (
        np.sign(df["pred_return"]) == np.sign(df["actual_logret"])
    ).map({True: "✓", False: "✗"})
    return df[["Predicted (%)", "Actual (%)", "Signal", "Direction correct"]]


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto Return Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Crypto Return Forecast Dashboard")
st.caption(
    "Predicts **next-day log return** — stationary, financially meaningful. "
    "LSTM + Quantile Regression · "
    "Built by [Yahya Elfirdoussi](https://yahiaelfirdoussi.netlify.app)"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    selected = st.multiselect(
        "Cryptocurrencies",
        options=list(CRYPTOS.keys()),
        default=["Bitcoin", "Ethereum"],
    )
    if not selected:
        st.warning("Select at least one asset.")
        st.stop()

    n_days    = st.slider("History (days)", 90, 1800, 365, 30)
    threshold = st.slider(
        "Signal threshold (%)",
        min_value=0.1, max_value=2.0, value=0.3, step=0.1,
        help="Minimum predicted return to trigger BUY or SELL. Below = NEUTRAL."
    ) / 100   # convert to decimal

    st.divider()
    st.subheader("⚡ Next-day signal")
    live_crypto = st.selectbox("Asset", list(CRYPTOS.keys()))

end_dt   = pd.Timestamp.today()
start_dt = end_dt - pd.Timedelta(days=n_days)

# ── Fetch ─────────────────────────────────────────────────────────────────────

data: dict[str, pd.DataFrame] = {}
for name in selected:
    with st.spinner(f"Fetching {name}…"):
        data[name] = fetch(CRYPTOS[name], str(start_dt.date()), str(end_dt.date()))

# ── Sidebar live signal ───────────────────────────────────────────────────────

with st.sidebar:
    try:
        live_df = data.get(live_crypto)

        if live_df is None:
            live_df = fetch(
                CRYPTOS[live_crypto],
                str(start_dt.date()),
                str(end_dt.date())
            )
        qm, qf   = load_quantile(live_crypto)
        X_live   = sm.add_constant(build_feature_matrix(live_df, qf, CRYPTOS[live_crypto]))
        pred_ret = float(np.squeeze(qm.predict(X_live))[-1])
        signal   = classify_signal(pred_ret, threshold)

        st.metric(
            f"{live_crypto} — predicted next-day return",
            f"{pred_ret*100:+.4f}%",
            delta=signal,
            delta_color="normal" if signal == "BUY" else
                         "inverse" if signal == "SELL" else "off",
        )
        st.markdown(
            f"<div style='font-size:28px;font-weight:600;"
            f"color:{signal_color(signal)};text-align:center'>{signal}</div>",
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning("Run `python3 scripts/train.py` first.")
    except Exception as e:
        st.warning(f"Signal unavailable: {e}")
    st.caption(f"Threshold: ±{threshold*100:.2f}% · Quantile Regression · Refreshed hourly")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📊 Charts & Indicators", "🤖 Return Forecast"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────

with tab1:
    for name, df in data.items():
        st.subheader(name)

        row  = df[["Close", "RSI_14", "MACD", "VOLAT_14D", "actual_logret"]].dropna().iloc[-1]
        prev = float(df["Close"].dropna().iloc[-2])
        curr = float(df["Close"].dropna().iloc[-1])
        rsi  = float(row["RSI_14"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price",          f"${curr:,.2f}",  f"{(curr-prev)/prev*100:+.2f}%")
        c2.metric("RSI (14)",       f"{rsi:.1f}",
                  "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
        c3.metric("MACD",           f"{row['MACD']:.4f}")
        c4.metric("14D Volatility", f"{row['VOLAT_14D']:.4f}")

        st.plotly_chart(candle_fig(df, name), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RSI & MACD**")
            st.line_chart(df[["RSI_14", "MACD"]].dropna(), height=200)
        with col2:
            st.markdown("**Moving averages**")
            st.line_chart(df[["Close", "SMA_50", "SMA_200", "EMA_21"]].dropna(), height=200)

        with st.expander("Bollinger Bands"):
            st.line_chart(
                df[["Close", "BOLL_UP_20", "BOLL_MID_20", "BOLL_LOW_20"]].dropna(),
                height=200,
            )
        st.divider()

# ── Tab 2 ─────────────────────────────────────────────────────────────────────

with tab2:

    st.info(
        "**Target: next-day log return** `log(Close_{t+1} / Close_t)` · "
        "Stationary · No unit root · Directly tradeable as a signal"
    )

    model_type = st.selectbox(
        "Model",
        ["Quantile Regression", "LSTM Deep Learning"],
    )

    with st.expander("📊 Model performance on test set", expanded=False):
        st.markdown("""
**Key metrics for return prediction:**
- **Direction accuracy** — % of days the model correctly predicted up/down. Random = 50%.
- **IC (Information Coefficient)** — Spearman rank correlation between predicted and actual returns. IC > 0.05 is considered useful in quant finance.
- **MAE / RMSE** — in log return units (e.g. 0.01 = 1% average error).

| Asset | Model | Direction Acc | IC | MAE |
|---|---|---|---|---|
| Bitcoin | Quantile Regression | — | — | — |
| Bitcoin | LSTM | — | — | — |
| Ethereum | Quantile Regression | — | — | — |
| Ethereum | LSTM | — | — | — |

*Fill in after running `python3 scripts/train.py` — metrics print to console.*
        """)

    for name in selected:
        ticker = CRYPTOS[name]
        df     = data[name].dropna(subset=["actual_logret"]).copy()
        st.subheader(f"{name} — {model_type}")

        if model_type == "Quantile Regression":
            try:
                with st.spinner(f"Loading {name} model…"):
                    model, feats = load_quantile(name)

                X = sm.add_constant(build_feature_matrix(df, feats, ticker))
                if not X.empty:
                    pred_ret = np.squeeze(model.predict(X))
                    act_ret  = df.loc[X.index, "actual_logret"].values

                    df_p = df.loc[X.index].copy()
                    df_p["pred_return"] = pred_ret

                    # ── Direction accuracy
                    dir_acc = np.mean(np.sign(pred_ret) == np.sign(act_ret)) * 100

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Direction accuracy", f"{dir_acc:.1f}%",
                              f"{dir_acc-50:+.1f}% vs random")
                    m2.metric("Mean predicted return",
                              f"{np.mean(pred_ret)*100:+.4f}%")
                    m3.metric("Today's signal",
                              classify_signal(pred_ret[-1], threshold),
                              f"{pred_ret[-1]*100:+.4f}%")

                    # ── Return bar chart
                    st.plotly_chart(
                        return_bar_fig(df_p.index, act_ret, pred_ret, name, "Quantile"),
                        use_container_width=True,
                    )

                    # ── Cumulative return / backtest
                    st.plotly_chart(
                        cumret_fig(df_p.index, act_ret, pred_ret, threshold, name),
                        use_container_width=True,
                    )

                    # ── Signal table
                    st.markdown("**Last 10 days — signals & outcomes**")
                    tbl = signal_table(df_p, threshold)
                    st.dataframe(tbl, use_container_width=True)

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

        else:  # LSTM
            st.info("LSTM loads on first use (~10–20s). Cached after that.")
            try:
                with st.spinner(f"Loading {name} LSTM…"):
                    lstm, scaler_X, scaler_y, feats = load_lstm(name)

                X     = build_feature_matrix(df, feats, ticker).values
                X_sc  = scaler_X.transform(X)
                SEQ   = 30
                seqs  = np.array([X_sc[i-SEQ:i] for i in range(SEQ, len(X_sc))])

                if seqs.shape[0] > 0:
                    with st.spinner("Running inference…"):
                        y_sc   = lstm.predict(seqs, verbose=0).flatten()
                        y_pred = scaler_y.inverse_transform(
                            y_sc.reshape(-1, 1)
                        ).flatten()

                    idx     = df.index[SEQ:]
                    act_ret = df["actual_logret"].iloc[SEQ:].values

                    df_p = df.iloc[SEQ:].copy()
                    df_p["pred_return"] = y_pred

                    dir_acc = np.mean(np.sign(y_pred) == np.sign(act_ret)) * 100

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Direction accuracy", f"{dir_acc:.1f}%",
                              f"{dir_acc-50:+.1f}% vs random")
                    m2.metric("Mean predicted return",
                              f"{np.mean(y_pred)*100:+.4f}%")
                    m3.metric("Today's signal",
                              classify_signal(y_pred[-1], threshold),
                              f"{y_pred[-1]*100:+.4f}%")

                    st.plotly_chart(
                        return_bar_fig(idx, act_ret, y_pred, name, "LSTM"),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        cumret_fig(idx, act_ret, y_pred, threshold, name),
                        use_container_width=True,
                    )
                    st.markdown("**Last 10 days — signals & outcomes**")
                    st.dataframe(signal_table(df_p, threshold), use_container_width=True)

                else:
                    st.warning("Need at least 30 days of data for LSTM.")

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"LSTM error: {e}")
