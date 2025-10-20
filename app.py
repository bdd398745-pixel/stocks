import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Buy/Sell Signal", layout="wide")

st.title("📈 Intelligent Stock Buy/Sell Signal Dashboard (India)")

st.markdown("""
This tool scans Indian stocks, identifies top 10 trending ones,  
and shows **Buy / Sell / Hold** signals based on EMA, RSI, and MACD.
""")

# ------------------------
# Config
# ------------------------
TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","LT.NS",
    "ITC.NS","SBIN.NS","AXISBANK.NS","BAJAJFINSV.NS","KOTAKBANK.NS","MARUTI.NS",
    "HINDUNILVR.NS","SUNPHARMA.NS","TITAN.NS","POWERGRID.NS","NTPC.NS","ULTRACEMCO.NS",
    "JSWSTEEL.NS","ADANIENT.NS"
]

LOOKBACK_DAYS = 365
MIN_AVG_VOL = 300000
TARGET_PCT = 0.12
STOP_PCT = 0.06

# ------------------------
# Indicator functions
# ------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

# ------------------------
# Analyze each ticker
# ------------------------
def analyze_ticker(ticker):
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        return None, None

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"] = macd(df["Close"])
    df["VolumeMA20"] = df["Volume"].rolling(20).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    trend = latest["EMA20"] > latest["EMA50"]
    macd_cross = (prev["MACD"] < prev["MACD_signal"]) and (latest["MACD"] > latest["MACD_signal"])
    macd_bear = (prev["MACD"] > prev["MACD_signal"]) and (latest["MACD"] < latest["MACD_signal"])

    buy = sell = False
    reason = ""

    if trend and latest["RSI"] > 40 and latest["RSI"] < 65 and macd_cross:
        buy = True
        reason = "Trend + RSI + MACD bullish crossover"
    elif macd_bear or latest["RSI"] > 70 or latest["Close"] < latest["EMA20"]:
        sell = True
        reason = "MACD bearish / RSI high / below EMA20"
    else:
        reason = "Hold / Neutral"

    entry = latest["Close"]
    target = entry * (1 + TARGET_PCT)
    stop = entry * (1 - STOP_PCT)

    return {
        "Ticker": ticker,
        "Close": round(entry, 2),
        "RSI": round(latest["RSI"], 2),
        "Trend": "Up" if trend else "Down",
        "Signal": "BUY ✅" if buy else ("SELL 🚫" if sell else "HOLD ⚪"),
        "Reason": reason,
        "Target": round(target, 2),
        "Stop": round(stop, 2)
    }, df

# ------------------------
# Main app
# ------------------------
data_list = []
progress = st.progress(0)
for i, t in enumerate(TICKERS):
    res, df = analyze_ticker(t)
    if res:
        data_list.append(res)
    progress.progress((i+1)/len(TICKERS))

if not data_list:
    st.warning("No data found for selected tickers.")
    st.stop()

df_all = pd.DataFrame(data_list)
top10 = df_all.head(10)

st.subheader("🔝 Top 10 Stocks with Current Signals")
st.dataframe(top10, use_container_width=True)

# ------------------------
# Chart section
# ------------------------
selected = st.selectbox("📊 Select a stock to view details", top10["Ticker"].tolist())
if selected:
    _, df_chart = analyze_ticker(selected)
    if df_chart is not None:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_chart.index,
                                     open=df_chart["Open"], high=df_chart["High"],
                                     low=df_chart["Low"], close=df_chart["Close"],
                                     name="Price"))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["EMA20"], name="EMA20"))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["EMA50"], name="EMA50"))
        fig.update_layout(title=f"{selected} Price with EMA20 & EMA50",
                          xaxis_title="Date", yaxis_title="Price (INR)",
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**RSI and MACD:**")
        st.line_chart(df_chart[["RSI"]])
        st.line_chart(df_chart[["MACD", "MACD_signal"]])
