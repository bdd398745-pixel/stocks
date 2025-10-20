# app.py
# =====================================================
# 📊 SMART STOCK SIGNAL TRACKER (Final, checked)
# =====================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta

# -----------------------------------------------------
# Page config
# -----------------------------------------------------
st.set_page_config(page_title="Smart Stock Signals", layout="wide")
st.title("📈 Smart Stock Buy/Sell Signal Tracker")

# -----------------------------------------------------
# Sidebar controls
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
show_volume = st.sidebar.checkbox("Show Volume on Chart", value=True)

# -----------------------------------------------------
# Data fetch + indicator computation
# -----------------------------------------------------
@st.cache_data
def get_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # Prefer explicit auto_adjust to avoid future warning. Set threads=False in some envs to avoid thread issues.
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure normal DataFrame (single ticker). If multiindex columns, try to reduce to single-level
    if isinstance(df.columns, pd.MultiIndex):
        # pick the first ticker's columns if multi-download happened by mistake
        df.columns = df.columns.get_level_values(-1)

    # Drop rows with any NaN (indicator lib may need this)
    df = df.dropna(how="all")
    df = df.dropna()  # remove rows where indicators would fail

    if df.empty:
        return pd.DataFrame()

    # Ensure Close is a 1-D Series aligned with df.index
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # if it's (n,1) DataFrame, convert to Series
        close = close.squeeze()

    # Now compute indicators (ta expects a 1-D Series)
    # Use try/except to avoid raising unexpected exceptions in some envs
    try:
        df["EMA20"] = ta.trend.EMAIndicator(close, window=20, fillna=False).ema_indicator()
        df["EMA50"] = ta.trend.EMAIndicator(close, window=50, fillna=False).ema_indicator()

        macd = ta.trend.MACD(close, fillna=False)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        df["RSI"] = ta.momentum.RSIIndicator(close, window=14, fillna=False).rsi()
    except Exception as e:
        # If indicators fail, return dataframe without them (caller will handle)
        st.warning(f"Indicator calculation failed: {e}")
        # Ensure the columns exist to avoid KeyError later
        for col in ["EMA20", "EMA50", "MACD", "MACD_signal", "RSI"]:
            if col not in df.columns:
                df[col] = np.nan

    return df

# Fetch data (with defensive handling)
df = get_data(selected_ticker, period, interval)
if df.empty:
    st.error("No data returned from Yahoo Finance for the selected ticker/period/interval. Try another selection.")
    st.stop()

# -----------------------------------------------------
# Signal generation (scalar-safe)
# -----------------------------------------------------
def _get_scalar(val):
    """
    Safely convert a pandas Scalar/1-element Series/ndarray to float.
    If val is already scalar, returns float(val).
    If val is nan or cannot be converted, returns np.nan.
    """
    try:
        # numpy arrays or pandas Series -> squeeze then float
        a = np.asarray(val).squeeze()
        # If a is an array of length > 1, pick last element (shouldn't happen for scalar cell)
        if isinstance(a, np.ndarray) and a.size > 1:
            a = a[-1]
        return float(a)
    except Exception:
        return np.nan

def generate_signals(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "HOLD"
    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    macd_latest = _get_scalar(latest_row.get("MACD", np.nan))
    macd_prev = _get_scalar(prev_row.get("MACD", np.nan))
    macd_signal_latest = _get_scalar(latest_row.get("MACD_signal", np.nan))
    macd_signal_prev = _get_scalar(prev_row.get("MACD_signal", np.nan))
    ema20_latest = _get_scalar(latest_row.get("EMA20", np.nan))
    ema50_latest = _get_scalar(latest_row.get("EMA50", np.nan))

    # If any essential value is NaN, return HOLD
    if np.isnan([macd_latest, macd_prev, macd_signal_latest, macd_signal_prev, ema20_latest, ema50_latest]).any():
        return "HOLD"

    macd_cross = (macd_prev < macd_signal_prev) and (macd_latest > macd_signal_latest)
    macd_bear = (macd_prev > macd_signal_prev) and (macd_latest < macd_signal_latest)

    buy_signal = (ema20_latest > ema50_latest) and macd_cross
    sell_signal = (ema20_latest < ema50_latest) and macd_bear

    if buy_signal:
        return "BUY"
    elif sell_signal:
        return "SELL"
    else:
        return "HOLD"

signal = generate_signals(df)

# -----------------------------------------------------
# Create vectorized Signal column for history table & markers
# -----------------------------------------------------
# Use boolean masks (these operate elementwise)
mask_buy = (df["EMA20"] > df["EMA50"]) & (df["MACD"] > df["MACD_signal"])
mask_sell = (df["EMA20"] < df["EMA50"]) & (df["MACD"] < df["MACD_signal"])
df["Signal"] = np.where(mask_buy, "BUY", np.where(mask_sell, "SELL", "HOLD"))

# -----------------------------------------------------
# Display current price and signal (robust scalar extraction)
# -----------------------------------------------------
def get_last_close_scalar(df: pd.DataFrame) -> float:
    raw = df["Close"].iloc[-1]
    try:
        val = np.asarray(raw).squeeze()
        if isinstance(val, np.ndarray) and val.size > 1:
            val = val[-1]
        return float(val)
    except Exception:
        # fallback: try using df["Close"].iat[-1]
        try:
            return float(df["Close"].iat[-1])
        except Exception:
            return np.nan

current_price = get_last_close_scalar(df)
col1, col2 = st.columns([1, 3])
with col1:
    if np.isnan(current_price):
        st.metric("Current Price", "N/A")
    else:
        st.metric("Current Price", f"₹{current_price:.2f}")

    if signal == "BUY":
        st.success("✅ BUY Signal Detected")
    elif signal == "SELL":
        st.error("🔻 SELL Signal Detected")
    else:
        st.info("⏸ HOLD / WAIT")

# -----------------------------------------------------
# Price chart (candlestick) with EMA overlays and optional volume + markers
# -----------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
))

# EMA traces (only if not all NaN)
if not df["EMA20"].isna().all():
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(width=1.5), hovertemplate="EMA20: %{y:.2f}<extra></extra>"))
if not df["EMA50"].isna().all():
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(width=1.5), hovertemplate="EMA50: %{y:.2f}<extra></extra>"))

# Volume as bar chart on secondary y-axis (optional)
if show_volume and "Volume" in df.columns:
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", yaxis="y2", opacity=0.2))

# Buy/Sell markers (plot only where appropriate)
buy_points = df[df["Signal"] == "BUY"]
sell_points = df[df["Signal"] == "SELL"]

if not buy_points.empty:
    fig.add_trace(go.Scatter(
        x=buy_points.index, y=buy_points["Close"],
        mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"),
        name="Buy Signal"
    ))
if not sell_points.empty:
    fig.add_trace(go.Scatter(
        x=sell_points.index, y=sell_points["Close"],
        mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
        name="Sell Signal"
    ))

# Layout adjustments for secondary y-axis when volume shown
layout = dict(
    title=f"{selected_ticker} Price & EMA Signals",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    height=520
)
if show_volume and "Volume" in df.columns:
    layout["yaxis2"] = dict(overlaying="y", side="right", showgrid=False, title="Volume", position=1.0, range=[0, df["Volume"].max() * 5])
    # note: volume scaling is visual — adjust as needed

fig.update_layout(**layout)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# MACD chart
# -----------------------------------------------------
fig2 = go.Figure()
if not df["MACD"].isna().all():
    fig2.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
if not df["MACD_signal"].isna().all():
    fig2.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
fig2.add_hline(y=0, line=dict(color="gray", dash="dot"))
fig2.update_layout(title="MACD Indicator", height=300, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------
# RSI chart (optional)
# -----------------------------------------------------
if show_rsi:
    fig3 = go.Figure()
    if "RSI" in df.columns and not df["RSI"].isna().all():
        fig3.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig3.add_hline(y=70, line=dict(color="red", dash="dot"))
    fig3.add_hline(y=30, line=dict(color="green", dash="dot"))
    fig3.update_layout(title="RSI Indicator (14-day)", height=300, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------
# Recent signal history (table)
# -----------------------------------------------------
st.markdown("### 📋 Recent Signal History")
display_cols = ["Close", "EMA20", "EMA50", "MACD", "MACD_signal", "RSI", "Signal"]
available_cols = [c for c in display_cols if c in df.columns]
# Round numeric columns for cleaner display
df_display = df[available_cols].copy()
for c in df_display.select_dtypes(include=[np.number]).columns:
    df_display[c] = df_display[c].round(4)
st.dataframe(df_display.tail(12))

# -----------------------------------------------------
# Download button
# -----------------------------------------------------
csv = df.to_csv().encode("utf-8")
st.download_button(
    label="💾 Download Data as CSV",
    data=csv,
    file_name=f"{selected_ticker}_signals.csv",
    mime="text/csv"
)

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.caption("📊 Data from Yahoo Finance | Built with ❤️ using Streamlit")
