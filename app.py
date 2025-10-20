# =====================================================
# 📊 SMART STOCK SIGNAL TRACKER (Enhanced Version)
# =====================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import ta  # Technical indicators library

# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(page_title="Smart Stock Signals", layout="wide")
st.title("📈 Smart Stock Buy/Sell Signal Tracker")

# -----------------------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------------------
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "AXISBANK.NS"
]

st.sidebar.header("⚙️ Settings")
selected_ticker = st.sidebar.selectbox("Select Stock", TICKERS)
period = st.sidebar.selectbox("Select Period", ["3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1h"], index=0)
show_rsi = st.sidebar.checkbox("Show RSI Indicator", value=True)

# -----------------------------------------------------
# FETCH DATA FROM YAHOO FINANCE
# -----------------------------------------------------
@st.cache_data
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    
    close = df["Close"].squeeze()  # ✅ Ensure 1D array

    # Add EMAs
    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    # Add MACD
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # Add RSI
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    return df

df = get_data(selected_ticker, period, interval)

# -----------------------------------------------------
# SIGNAL GENERATION
# -----------------------------------------------------
def generate_signals(df):
    if len(df) < 2:
        return "HOLD"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    macd_cross = (prev["MACD"] < prev["MACD_signal"]) and (latest["MACD"] > latest["MACD_signal"])
    macd_bear = (prev["MACD"] > prev["MACD_signal"]) and (latest["MACD"] < latest["MACD_signal"])

    buy_signal = (latest["EMA20"] > latest["EMA50"]) and macd_cross
    sell_signal = (latest["EMA20"] < latest["EMA50"]) and macd_bear

    if buy_signal:
        return "BUY"
    elif sell_signal:
        return "SELL"
    else:
        return "HOLD"

signal = generate_signals(df)

# -----------------------------------------------------
# SIGNAL COLUMN FOR HISTORY
# -----------------------------------------------------
df["Signal"] = np.where(
    (df["EMA20"] > df["EMA50"]) & (df["MACD"] > df["MACD_signal"]), "BUY",
    np.where((df["EMA20"] < df["EMA50"]) & (df["MACD"] < df["MACD_signal"]), "SELL", "HOLD")
)

# -----------------------------------------------------
# DISPLAY SIGNAL STATUS
# -----------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
    if signal == "BUY":
        st.success("✅ BUY Signal Detected")
    elif signal == "SELL":
        st.error("🔻 SELL Signal Detected")
    else:
        st.info("⏸ HOLD / WAIT")

# -----------------------------------------------------
# PRICE CHART WITH SIGNAL MARKERS
# -----------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="orange", width=1.5)))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="blue", width=1.5)))

# Add Buy/Sell Markers
buy_points = df[df["Signal"] == "BUY"]
sell_points = df[df["Signal"] == "SELL"]

fig.add_trace(go.Scatter(
    x=buy_points.index, y=buy_points["Close"],
    mode="markers", marker=dict(symbol="triangle-up", color="green", size=10),
    name="Buy Signal"
))
fig.add_trace(go.Scatter(
    x=sell_points.index, y=sell_points["Close"],
    mode="markers", marker=dict(symbol="triangle-down", color="red", size=10),
    name="Sell Signal"
))

fig.update_layout(
    title=f"{selected_ticker} Price & EMA Signals",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# MACD CHART
# -----------------------------------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green")))
fig2.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="red")))
fig2.add_hline(y=0, line=dict(color="gray", dash="dot"))
fig2.update_layout(title="MACD Indicator", height=300, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------
# RSI CHART (Optional)
# -----------------------------------------------------
if show_rsi:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="purple")))
    fig3.add_hline(y=70, line=dict(color="red", dash="dot"))
    fig3.add_hline(y=30, line=dict(color="green", dash="dot"))
    fig3.update_layout(title="RSI Indicator (14-day)", height=300, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------
# RECENT SIGNAL HISTORY
# -----------------------------------------------------
st.markdown("### 📋 Recent Signal History")
st.dataframe(df.tail(10)[["Close", "EMA20", "EMA50", "MACD", "MACD_signal", "Signal"]])

# -----------------------------------------------------
# DOWNLOAD OPTION
# -----------------------------------------------------
csv = df.to_csv().encode("utf-8")
st.download_button(
    label="💾 Download Data as CSV",
    data=csv,
    file_name=f"{selected_ticker}_signals.csv",
    mime="text/csv"
)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.caption("📊 Data from Yahoo Finance | Built with ❤️ using Streamlit")
