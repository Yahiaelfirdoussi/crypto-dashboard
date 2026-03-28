# 📈 Crypto Analytics Dashboard

> Real-time Bitcoin & Ethereum price analysis with ML-powered price prediction.  
> LSTM deep learning + Quantile Regression, live yfinance data, deployed on Streamlit.

🚀 **[Live Demo](https://your-app.streamlit.app)** ← *add your URL after deploying*

---

## Features

- **Candlestick charts** with volume overlay (BTC & ETH)
- **Technical indicators** — RSI, MACD, Bollinger Bands, SMA 50/200, EMA 21
- **Quantile Regression** — fast, interpretable next-day close forecast with confidence intervals
- **LSTM Deep Learning** — 60-day sequence model capturing long-range price patterns
- **Live sidebar metric** — real-time next-close prediction updated every hour
- **Error table** — actual vs predicted with absolute and percentage error per day

---

## Model results

> Run `python scripts/train.py` to reproduce. Results vary with training window.

### Bitcoin (BTC-USD)

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Quantile Regression | $640 | $801 | 0.74% | 0.998 |
| LSTM | $5,582 | $6,938 | 7.02% | 0.768 |

### Ethereum (ETH-USD)

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Quantile Regression | $36 | $47 | 1.25% | 0.996 |
| LSTM | $148 | $181 | 5.34% | 0.924 |

*Trained on 3 years of daily OHLCV data · Chronological 80/20 split · No data leakage*

---

## Project structure

```
crypto_dashboard/
├── app.py                        # Streamlit dashboard (entry point)
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Theme and server config
├── src/
│   └── features.py               # Shared feature engineering
├── scripts/
│   └── train.py                  # Full retraining pipeline (BTC + ETH)
├── models/                       # Saved model artefacts (auto-created)
│   ├── best_lstm_model_btc.keras
│   ├── best_lstm_model_eth.keras
│   ├── quantile_regression_btc.pkl
│   ├── quant_reg_eth.pkl
│   ├── scaler_X_btc.save / scaler_X_eth.save
│   ├── scaler_y_btc.save / scaler_y_eth.save
│   └── selected_features_btc.pkl / selected_features_eth.pkl
└── notebook/
    └── crypto_analysis.ipynb     # EDA and prototyping
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Yahiaelfirdoussi/crypto_dashboard
cd crypto_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (downloads live data automatically)
python scripts/train.py

# 4. Run the dashboard
streamlit run app.py
```

Training takes ~5–10 minutes on CPU. Models are saved to `models/` automatically.

To train only one asset:
```bash
python scripts/train.py --tickers BTC-USD
python scripts/train.py --tickers ETH-USD
python scripts/train.py --years 4        # use 4 years of history
```

---

## Deploy on Streamlit Cloud (free)

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

1. Push the repo to GitHub (including the `models/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select `Yahiaelfirdoussi/crypto_dashboard`
4. Set main file: `app.py`
5. Deploy — live in ~2 minutes

---

## Technical decisions

**Why LSTM?**  
Crypto prices are sequential time series. LSTMs capture long-range temporal dependencies (e.g. weekly patterns, trend reversals) that simpler models miss. The 60-day lookback window lets the model see ~2 months of context before each prediction.

**Why Quantile Regression?**  
A single-point prediction is misleading for volatile assets. Quantile regression at q=0.5 gives the median forecast. Running it at q=0.1 and q=0.9 gives lower/upper bounds — a confidence interval for risk-aware decision making.

**Why chronological split?**  
Random shuffling causes data leakage in time series: rolling features (MACD, RSI, moving averages) computed over future data give the model information it wouldn't have in production. The last 20% of trading days are held out as a strict test set.

**Why yfinance?**  
Live data means the dashboard is always current. No manual dataset updates needed — every launch fetches the latest OHLCV candles.

---

## Tech stack

`Python 3.11` · `TensorFlow / Keras` · `Scikit-learn` · `Statsmodels`  
`Streamlit` · `Plotly` · `yfinance` · `ta` · `Pandas` · `NumPy`

---

## Author

**Yahya Elfirdoussi** — Data Scientist & AI Engineer  
📧 [yahiaelfirdoussi7@gmail.com](mailto:yahiaelfirdoussi7@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/yahya-elfirdoussi) · [Portfolio](https://yahiaelfirdoussi.netlify.app) · [GitHub](https://github.com/yahiaelfirdoussi)
# crypto-dashboard
