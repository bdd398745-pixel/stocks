import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.set_page_config(page_title="📈 Smart Stock Analyzer", layout="wide")

# -----------------------------------------------------
# Helper function to safely fetch data
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()

        # Handle multi-level columns (common for yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # Drop rows with missing close
        if "Close" not in df.columns:
            raise KeyError("No 'Close' column found in Yahoo data.")

        return df

    except Exception as e:
        st.warning(f"⚠️ Data fetch failed: {e}")
        return pd.DataFrame()

# -----------------------------------------------------
# Technical indicator calculation
# -----------------------------------------------------
def compute_indicators(df):
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

    # Generate buy/sell signals (MACD crossover + RSI confirmation)
    df["Buy_Signal"] = np.where(
        (df["MACD"] > df["Signal"]) & (df["RSI"] < 60), df["Close"], np.nan
    )
    df["Sell_Signal"] = np.where(
        (df["MACD"] < df["Signal"]) & (df["RSI"] > 40), df["Close"], np.nan
    )

    return df

# -----------------------------------------------------
# Intelligent signal summary
# -----------------------------------------------------
def latest_signal(df):
    last_row = df.iloc[-1]
    if last_row["MACD"] > last_row["Signal"] and last_row["RSI"] < 60:
        return "BUY"
    elif last_row["MACD"] < last_row["Signal"] and last_row["RSI"] > 40:
        return "SELL"
    else:
        return "HOLD"

# -----------------------------------------------------
# UI Layout
# -----------------------------------------------------
st.title("📊 Smart Stock Analyzer (India)")

st.markdown("""
Analyze any Indian stock in real time with technical indicators — MACD, RSI, and intelligent buy/sell detection.
""")

with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input(
        "Search Stock (e.g. TCS.NS, RELIANCE.NS, HDFCBANK.NS):",
        value="TCS.NS",
        placeholder="Enter NSE stock symbol ending with .NS"
    )

    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    interval = st.selectbox("Select Interval", ["1d", "1h", "30m", "15m"], index=0)

# -----------------------------------------------------
# Fetch data
# -----------------------------------------------------
df = get_data(ticker, period, interval)
if df.empty:
    st.error("No data available for the entered ticker. Try again with a valid NSE code (e.g. INFY.NS).")
    st.stop()

df = compute_indicators(df)
signal = latest_signal(df)

# -----------------------------------------------------
# Layout metrics
# -----------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    current_price = float(df["Close"].iloc[-1])
    st.metric("Current Price", f"₹{current_price:,.2f}")

    if signal == "BUY":
        st.success("✅ BUY Signal Detected")
    elif signal == "SELL":
        st.error("🔻 SELL Signal Detected")
    else:
        st.info("⚪ HOLD — No clear trend yet")

# -----------------------------------------------------
# Plot graph with Buy/Sell markers
# -----------------------------------------------------
with col2:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode='lines', name='Close Price', line=dict(color='royalblue', width=2)
    ))

    # Buy markers
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Buy_Signal"],
        mode="markers", name="Buy Signal",
        marker=dict(symbol="triangle-up", size=12, color="limegreen")
    ))

    # Sell markers
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Sell_Signal"],
        mode="markers", name="Sell Signal",
        marker=dict(symbol="triangle-down", size=12, color="red")
    ))

    fig.update_layout(
        title=f"{ticker} — Price with Buy/Sell Indicators",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# Optional: Indicator chart below
# -----------------------------------------------------
with st.expander("📈 View Technical Indicators (MACD + RSI)"):
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color='orange')))
    macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal", line=dict(color='blue', dash='dot')))
    macd_fig.update_layout(title="MACD vs Signal Line", height=300, template="plotly_white")
    st.plotly_chart(macd_fig, use_container_width=True)

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color='purple')))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_fig.update_layout(title="RSI Indicator", height=300, template="plotly_white")
    st.plotly_chart(rsi_fig, use_container_width=True)
