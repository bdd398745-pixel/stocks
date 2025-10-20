import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta  # for technical indicators

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Smart Stock Signals", layout="wide")
st.title("📈 Smart Stock Buy/Sell Signal Tracker")

# ----------------------------
# SELECT STOCKS
# ----------------------------
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "AXISBANK.NS"
]

st.sidebar.header("⚙️ Settings")
selected_ticker = st.sidebar.selectbox("Select a Stock", TICKERS)
period = st.sidebar.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=2)

# ----------------------------
# FETCH DATA
# ----------------------------
@st.cache_data
def get_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    df.dropna(inplace=True)
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    return df

df = get_data(selected_ticker, period)

# ----------------------------
# SIGNAL LOGIC
# ----------------------------
def generate_signals(df):
    if len(df) < 2:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    macd_cross = (
        (prev["MACD"].item() < prev["MACD_signal"].item()) and 
        (latest["MACD"].item() > latest["MACD_signal"].item())
    )
    macd_bear = (
        (prev["MACD"].item() > prev["MACD_signal"].item()) and 
        (latest["MACD"].item() < latest["MACD_signal"].item())
    )

    buy_signal = (latest["EMA20"] > latest["EMA50"]) and macd_cross
    sell_signal = (latest["EMA20"] < latest["EMA50"]) and macd_bear

    if buy_signal:
        return "BUY"
    elif sell_signal:
        return "SELL"
    else:
        return "HOLD"

signal = generate_signals(df)

# ----------------------------
# DISPLAY SIGNAL
# ----------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
    if signal == "BUY":
        st.success("✅ BUY Signal Detected")
    elif signal == "SELL":
        st.error("🔻 SELL Signal Detected")
    else:
        st.info("⏸ HOLD / WAIT")

# ----------------------------
# PRICE CHART
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="orange", width=1.5)))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="blue", width=1.5)))

fig.update_layout(
    title=f"{selected_ticker} Price & Signals",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# MACD CHART
# ----------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green")))
fig2.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="red")))
fig2.add_hline(y=0, line=dict(color="gray", dash="dot"))
fig2.update_layout(title="MACD Indicator", height=300, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# FOOTER
# ----------------------------
st.caption("📊 Data from Yahoo Finance | Built with ❤️ in Streamlit")

