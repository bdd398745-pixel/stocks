import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Signal Dashboard", layout="wide")

# -----------------------------------------------------
# HELPER FUNCTION TO DOWNLOAD DATA
# -----------------------------------------------------
@st.cache_data
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns from yfinance (e.g. ('Close', 'TCS.NS'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # flatten

    # Make sure Close is a 1-D Series
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    df["Close"] = close

    return df


# -----------------------------------------------------
# CALCULATE TECHNICAL INDICATORS
# -----------------------------------------------------
def add_indicators(df):
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    return df


# -----------------------------------------------------
# GENERATE SIGNAL
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


# -----------------------------------------------------
# UI CONTROLS
# -----------------------------------------------------
st.title("📊 Stock Technical Signal Dashboard")

ticker_list = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
selected_ticker = st.selectbox("Select Stock", ticker_list)
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.selectbox("Select Interval", ["1d", "1h", "15m"], index=0)

# -----------------------------------------------------
# FETCH DATA
# -----------------------------------------------------
df = get_data(selected_ticker, period, interval)
if df.empty:
    st.error("No data returned from Yahoo Finance for this stock.")
    st.stop()

df = add_indicators(df)
signal = generate_signals(df)

# -----------------------------------------------------
# METRICS & SIGNAL DISPLAY
# -----------------------------------------------------
col1, col2 = st.columns([1, 3])

with col1:
    current_price = df["Close"].iloc[-1]

    # Fix: if Series (multi-column issue), convert safely
    if isinstance(current_price, pd.Series):
        current_price = float(current_price.squeeze())
    else:
        current_price = float(current_price)

    st.metric("Current Price", f"₹{current_price:.2f}")

    if signal == "BUY":
        st.success("✅ **BUY Signal Detected**")
    elif signal == "SELL":
        st.error("🚨 **SELL Signal Detected**")
    else:
        st.info("⏸️ **HOLD — No Clear Signal**")

with col2:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA20"], line=dict(color="blue", width=1.5), name="EMA 20"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA50"], line=dict(color="orange", width=1.5), name="EMA 50"
    ))

    fig.update_layout(
        title=f"{selected_ticker} Price Chart with EMAs",
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
st.subheader("📈 MACD Trend")
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green")))
macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="red", dash="dot")))

macd_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="MACD Value",
    template="plotly_white",
    height=300
)

st.plotly_chart(macd_fig, use_container_width=True)

# -----------------------------------------------------
# DATA TABLE (OPTIONAL)
# -----------------------------------------------------
st.dataframe(df.tail(10))
