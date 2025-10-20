import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import ta

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Smart Stock Signal Tracker", layout="wide")
st.title("📈 Smart Stock Buy/Sell Signal Tracker (India)")

st.sidebar.header("⚙️ Settings")

# -----------------------------------------------------
# SEARCH BAR INPUT
# -----------------------------------------------------
st.sidebar.write("Type NSE stock code, e.g., RELIANCE.NS, TCS.NS, ITC.NS")
selected_ticker = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE.NS").strip().upper()
period = st.sidebar.selectbox("Select Period", ["3mo", "6mo", "1y", "2y"], index=2)
interval = "1d"

# -----------------------------------------------------
# FETCH DATA SAFELY
# -----------------------------------------------------
@st.cache_data
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Ensure Close column exists
        if "Close" not in df.columns:
            return pd.DataFrame()
        # Add indicators
        df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
        df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        return df.dropna()
    except Exception as e:
        st.warning(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()

df = get_data(selected_ticker, period, interval)

if df.empty:
    st.error("❌ No valid data returned. Check the symbol (e.g., use RELIANCE.NS).")
    st.stop()

# -----------------------------------------------------
# SIGNAL GENERATION (for all points)
# -----------------------------------------------------
def generate_signals(df):
    df["Signal"] = "HOLD"
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        macd_cross = (prev["MACD"] < prev["MACD_signal"]) and (curr["MACD"] > curr["MACD_signal"])
        macd_bear = (prev["MACD"] > prev["MACD_signal"]) and (curr["MACD"] < curr["MACD_signal"])
        buy_signal = (curr["EMA20"] > curr["EMA50"]) and macd_cross
        sell_signal = (curr["EMA20"] < curr["EMA50"]) and macd_bear
        if buy_signal:
            df.at[df.index[i], "Signal"] = "BUY"
        elif sell_signal:
            df.at[df.index[i], "Signal"] = "SELL"
    return df

df = generate_signals(df)

latest_signal = df["Signal"].iloc[-1]
latest_price = df["Close"].iloc[-1]

# -----------------------------------------------------
# DISPLAY CURRENT STATUS
# -----------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Current Price", f"₹{latest_price:.2f}")
    if latest_signal == "BUY":
        st.success("✅ BUY Signal Detected")
    elif latest_signal == "SELL":
        st.error("🔻 SELL Signal Detected")
    else:
        st.info("⏸ HOLD / WAIT")

# -----------------------------------------------------
# PRICE CHART WITH SIGNAL MARKERS
# -----------------------------------------------------
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price",
    increasing_line_color="green",
    decreasing_line_color="red"
))

# EMA Lines
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="orange", width=1.5)))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="blue", width=1.5)))

# Buy markers
buy_points = df[df["Signal"] == "BUY"]
fig.add_trace(go.Scatter(
    x=buy_points.index,
    y=buy_points["Close"],
    mode="markers",
    name="Buy",
    marker=dict(symbol="triangle-up", color="green", size=12)
))

# Sell markers
sell_points = df[df["Signal"] == "SELL"]
fig.add_trace(go.Scatter(
    x=sell_points.index,
    y=sell_points["Close"],
    mode="markers",
    name="Sell",
    marker=dict(symbol="triangle-down", color="red", size=12)
))

fig.update_layout(
    title=f"{selected_ticker} Price Chart with Buy/Sell Signals",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    height=600,
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
# FOOTER
# -----------------------------------------------------
st.caption("📊 Data from Yahoo Finance | Built with ❤️ in Streamlit")
