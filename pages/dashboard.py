import socket
import requests
from io import StringIO
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timezone
socket.setdefaulttimeout(15)  # prevents hanging forever on cloud



# --- 🔐 LOGIN PROTECTION ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first 🔐")
    st.stop()

# --- ✅ CACHED MODEL TRAINING ---
@st.cache_resource
def train_model(x_train, y_train):
    model = Sequential()
    
    # Layer 1
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    
    # Layer 2
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

@st.cache_data(ttl=300)  # shorter TTL so empty results don't stick for long
def load_data(stock, start_date, end_date):
    stock = str(stock).strip().upper()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if end_date <= start_date:
        return pd.DataFrame()

    # --------- Stooq first (try .com and .pl) ----------
    def stooq_symbol(sym: str) -> str:
        s = sym.lower()
        # Stooq format for US stocks: aapl.us
        if s.isalpha():
            return f"{s}.us"
        return s

    sym = stooq_symbol(stock)
    headers = {"User-Agent": "Mozilla/5.0"}

    for base in ["https://stooq.com", "https://stooq.pl"]:
        try:
            url = f"{base}/q/d/l/?s={sym}&i=d"
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()

            df = pd.read_csv(StringIO(r.text))
            if not df.empty and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
                if not df.empty:
                    return df
        except Exception:
            pass

    # --------- Yahoo fallback ----------
    try:
        df = yf.download(
            stock,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()
        
def load_data_twelvedata(stock, start_date, end_date):
    api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    stock = str(stock).strip().upper()
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": stock,
        "interval": "1day",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "apikey": api_key,
        "outputsize": 5000,
        "format": "JSON",
    }

    r = requests.get(url, params=params, timeout=20)
    js = r.json()

    if "values" not in js:
        return pd.DataFrame()

    df = pd.DataFrame(js["values"])
    df.rename(columns={"datetime": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)

    return df
def get_company_info_twelvedata(symbol: str) -> dict:
    api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    if not api_key:
        return {}

    url = "https://api.twelvedata.com/quote"
    r = requests.get(url, params={"symbol": symbol, "apikey": api_key}, timeout=20)
    js = r.json()
    return js if isinstance(js, dict) else {}
# --- ✅ FORWARD TEST DB (LIVE ACCURACY LOG) ---
@st.cache_resource
def pred_db():
    conn = sqlite3.connect("predictions.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forward_predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            ticker TEXT,
            predicted_for DATE,
            predicted_close REAL,
            last_close REAL,
            created_at TEXT,
            actual_close REAL,
            evaluated_at TEXT
        )
    """)
    conn.commit()
    return conn

def next_market_day_using_yf(ticker: str, last_date):
    # No network call (works on Streamlit Cloud). Skips weekends only.
    d = pd.to_datetime(last_date).normalize() + pd.Timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += pd.Timedelta(days=1)
    return d

def log_prediction(conn, username, ticker, predicted_for, predicted_close, last_close):
    conn.execute("""
        INSERT INTO forward_predictions(
            username, ticker, predicted_for, predicted_close, last_close, created_at,
            actual_close, evaluated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, NULL, NULL)
    """, (
        username, ticker,
        str(pd.to_datetime(predicted_for).date()),
        float(predicted_close),
        float(last_close),
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()

def refresh_actuals(conn, username, ticker=None):
    q = """
        SELECT id, ticker, predicted_for
        FROM forward_predictions
        WHERE username=? AND actual_close IS NULL
    """
    params = [username]
    if ticker:
        q += " AND ticker=?"
        params.append(ticker)

    rows = conn.execute(q, params).fetchall()

    for row_id, tkr, pred_for in rows:
        pred_for = pd.to_datetime(pred_for).normalize()
        if st.button("🧪 Test data for predicted day"):
            test_df = load_data_twelvedata(stock, pred_for, pred_for + pd.Timedelta(days=7))
            st.write(test_df.head())

        df = load_data_twelvedata(tkr, pred_for, pred_for + pd.Timedelta(days=7))
        if df.empty:
            continue

        df.index = pd.to_datetime(df.index).normalize()
        if pred_for not in df.index:
            continue

        val = df.loc[pred_for, "Close"]
        actual_close = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

        conn.execute("""
            UPDATE forward_predictions
            SET actual_close=?, evaluated_at=?
            WHERE id=?
        """, (actual_close, datetime.now(timezone.utc).isoformat(), row_id))

    conn.commit()

def load_logs(conn, username, ticker=None):
    q = """
        SELECT predicted_for, ticker, last_close, predicted_close, actual_close, created_at, evaluated_at
        FROM forward_predictions
        WHERE username=?
    """
    params = [username]
    if ticker:
        q += " AND ticker=?"
        params.append(ticker)

    q += " ORDER BY predicted_for ASC"
    df = pd.read_sql_query(q, conn, params=params)
    df["predicted_for"] = pd.to_datetime(df["predicted_for"])
    return df



# --- GET USERNAME ---
username = st.session_state.get("username", "User")

# --- DARK THEME & CUSTOM CSS ---
st.markdown("""
<style>
    /* Main App */
    .stApp { background-color: #0F172A; color: white; }
    [data-testid="stSidebarNav"] { display: none; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #111827; }
    
    /* Sidebar Buttons */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #2563EB !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stDownloadButton button {
    background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
}
    color: white !important;
    border-radius: 8px;
    border: none;
    font-weight: bold;
    padding: 8px 16px;
}

.stDownloadButton button:hover {
    background-color: #1D4ED8 !important;
    color: white !important;
}        
    
    /* All Text */
    h1, h2, h3, h4, h5, h6, p, label { color: white !important; }
    
    /* Date picker fix */
    div[role="dialog"] { background-color: white !important; }
    div[role="dialog"] * { color: black !important; }
/* Fix top Streamlit header (Deploy + menu area) */
header[data-testid="stHeader"] {
  background: #0F172A !important;
}

/* Make the toolbar/menu icons visible */
header[data-testid="stHeader"] * {
  color: #E5E7EB !important;
}

header[data-testid="stHeader"] svg {
  fill: #E5E7EB !important;
  stroke: #E5E7EB !important;
}

/* Optional: remove the thin decoration line at very top */
div[data-testid="stDecoration"] {
  background: #0F172A !important;
}
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("📈 AI Stock Prediction Dashboard")
st.markdown("LSTM Deep Learning Based Stock Analysis System")

# --- SIDEBAR ---
st.sidebar.title("Settings")
debug_mode = st.sidebar.checkbox("Debug mode")
# --- Admin: Clear Streamlit cache ---
if st.sidebar.button("🧹 Clear Cache (Admin)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Profile Section
st.sidebar.markdown("## 👤 Profile")
st.sidebar.markdown(
    f"""
    <div style="text-align:center;">
    <img src="https://api.dicebear.com/7.x/initials/svg?seed={username}" width="80" style="border-radius:50%;">
    <h4 style="color:white;">{username}</h4>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---") 

# Logout Button
logout = st.sidebar.button("🔐 Logout")
if logout:
    st.session_state.logged_in = False
    st.rerun()

# Stock Selection
stock = st.sidebar.text_input("Enter Stock Symbol", "AAPL").strip().upper()
st.sidebar.markdown("Examples: AAPL, TSLA, RELIANCE.NS, BTC-USD")


# Date Selection
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
st.sidebar.markdown("---")
data_mode = st.sidebar.radio("Data Source", ["Online (Yahoo)", "Upload CSV"], index=0)

uploaded_file = None
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Chart Settings")

chart_type = st.sidebar.selectbox("Chart Type", ["Line (Close)", "Candlestick + Volume"])

indicators = st.sidebar.multiselect(
    "Indicators",
    ["EMA 20", "EMA 50", "Bollinger Bands (20,2)", "RSI 14"],
    default=["EMA 20", "EMA 50"]
)


# --- MAIN APP LOGIC ---
try:
    with st.spinner("Loading Stock Data..."):
        if data_mode == "Upload CSV":
            if uploaded_file is None:
                st.info("Upload a CSV file to continue. Columns required: Date, Open, High, Low, Close (Volume optional).")
                st.stop()

            data = pd.read_csv(uploaded_file)

            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"])
                data = data.set_index("Date")

            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(set(data.columns)):
                st.error(f"CSV must contain columns: {sorted(required)}. Found: {list(data.columns)}")
                st.stop()

            data = data.sort_index()

        else:
            data = load_data_twelvedata(stock, start_date, end_date)

    # after spinner ends, validate data once
    if data is None or data.empty:
        st.error(
            "Online data is not available on Streamlit Cloud (Yahoo may block cloud servers). "
            "Switch Data Source to 'Upload CSV'."
        )
        st.stop()

    # --- Indicators (for charts only) ---
    df = data.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    window = 20
    rolling_mean = df["Close"].rolling(window).mean()
    rolling_std = df["Close"].rolling(window).std()
    df["BB_upper"] = rolling_mean + 2 * rolling_std
    df["BB_lower"] = rolling_mean - 2 * rolling_std

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))
    # Company Information Section
    info = get_company_info_twelvedata(stock)

    st.subheader("🏢 Company Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Company:**", info.get("name", "N/A"))
        st.write("**Exchange:**", info.get("exchange", "N/A"))
    with col2:
        st.write("**Currency:**", info.get("currency", "N/A"))
        st.write("**Country:**", info.get("country", "N/A"))
    # Metrics Section
    st.subheader(f"📊 {stock} Latest Data")
    latest_price = float(data["Close"].iloc[-1])
    open_price = float(data["Open"].iloc[-1])
    high_price = float(data["High"].iloc[-1])
    low_price = float(data["Low"].iloc[-1])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${latest_price:.2f}")
    m2.metric("Open", f"${open_price:.2f}")
    m3.metric("High", f"${high_price:.2f}")
    m4.metric("Low", f"${low_price:.2f}")

    st.dataframe(data.tail(), use_container_width=True)

    # Download Data
    csv = data.to_csv().encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"{stock}.csv", "text/csv")

    st.subheader("📈 Professional Chart")

    if chart_type == "Line (Close)":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2)))

        if "EMA 20" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20"))
        if "EMA 50" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50"))
        if "Bollinger Bands (20,2)" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(width=1)))

        fig.update_layout(
            template="plotly_dark",
            height=520,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.05
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                name="Candles"
            ),
            row=1, col=1
        )

        if "EMA 20" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", line=dict(width=1.5)), row=1, col=1)
        if "EMA 50" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(width=1.5)), row=1, col=1)
        if "Bollinger Bands (20,2)" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(width=1)), row=1, col=1)

        if "Volume" in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.4), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=650,
            hovermode="x unified",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

   
        if "RSI 14" in indicators:
            st.subheader("📉 RSI (14)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI 14"))
            rsi_fig.add_hline(y=70, line_dash="dash")
            rsi_fig.add_hline(y=30, line_dash="dash")
            rsi_fig.update_layout(template="plotly_dark", height=300, hovermode="x unified")
            st.plotly_chart(rsi_fig, use_container_width=True)

    # --- 🤖 DATA PREPARATION & ML (LEAKAGE-FREE) ---
    close_values = data[["Close"]].values  # numpy array

    training_data_len = int(len(close_values) * 0.8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_values[:training_data_len])      # ✅ fit ONLY on train
    scaled_data = scaler.transform(close_values)      # transform all using train scaler

    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 60:]  # keep lookback overlap

    # Create Train Data
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Build and Train Model
    with st.spinner("Training LSTM Model..."):
        model = train_model(x_train, y_train)
    st.success("Model Training Completed ✅")

    # Create Test Data
    x_test = []
    y_test = close_values[training_data_len:]  # ✅ actual close values (unscaled)

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test).reshape(-1, 60, 1)

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    st.subheader("🎯 Model Accuracy")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    # MAE + MAPE (%)
    eps = 1e-8  # prevents divide-by-zero
    mae = float(np.mean(np.abs(y_test - predictions)))
    mape = float(np.mean(np.abs((y_test - predictions) / (y_test + eps))) * 100)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
    st.write(f"**Accuracy (approx = 100 - MAPE):** {100 - mape:.2f}%")

    # Plot Actual vs Predicted
    st.subheader("🤖 Actual vs Predicted Prices")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index[training_data_len:], y=y_test.flatten(), name='Actual Price'))
    fig3.add_trace(go.Scatter(x=data.index[training_data_len:], y=predictions.flatten(), name='Predicted Price'))
    fig3.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # Next Day Prediction
    last_60_days = scaled_data[-60:].reshape(1, 60, 1)
    next_day_prediction = model.predict(last_60_days)
    next_day_prediction = scaler.inverse_transform(next_day_prediction)

    st.subheader("🔮 Next Day Prediction")
    st.success(f"Predicted Next Close Price: **${next_day_prediction[0][0]:.2f}**")

    st.subheader("🧾 Forward Test (Live Accuracy Log)")

    if data_mode == "Upload CSV":
        st.info("Forward Test needs Online mode (to fetch actual close). Switch Data Source to 'Online'.")
    else:
        conn = pred_db()

        pred_for = next_market_day_using_yf(stock, data.index[-1])
        st.write("Prediction is for:", pred_for.date())

        colA, colB = st.columns(2)

        with colA:
            if st.button("✅ Save this prediction to log", type="primary"):
                log_prediction(
                    conn=conn,
                    username=username,
                    ticker=stock,
                    predicted_for=pred_for,
                    predicted_close=float(next_day_prediction[0][0]),
                    last_close=float(latest_price),
                )
                st.success("Saved. Come back later and click Refresh to evaluate.")

        with colB:
            if st.button("🔄 Refresh actual prices (evaluate pending logs)", type="primary"):
                refresh_actuals(conn, username=username, ticker=stock)
                st.success("Refreshed.")

        logs = load_logs(conn, username=username, ticker=stock)

        pending = int(logs["actual_close"].isna().sum()) if not logs.empty else 0
        evaluated = int(len(logs) - pending) if not logs.empty else 0
        st.caption(f"Forward Test Status → Evaluated: {evaluated} | Pending: {pending}")

        if logs.empty:
            st.info("No forward-test logs yet. Save a prediction first.")
        else:
            eval_df = logs.dropna(subset=["actual_close"]).copy()

            if not eval_df.empty:
                eval_df["abs_error"] = (eval_df["actual_close"] - eval_df["predicted_close"]).abs()
                eval_df["pct_error"] = (eval_df["abs_error"] / eval_df["actual_close"]) * 100

                rmse_live = np.sqrt(mean_squared_error(eval_df["actual_close"], eval_df["predicted_close"]))
                mae_live = eval_df["abs_error"].mean()
                mape_live = eval_df["pct_error"].mean()

                m1, m2, m3 = st.columns(3)
                m1.metric("Live RMSE", f"{rmse_live:.2f}")
                m2.metric("Live MAE", f"{mae_live:.2f}")
                m3.metric("Live MAPE", f"{mape_live:.2f}%")

                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(x=eval_df["predicted_for"], y=eval_df["actual_close"], name="Actual"))
                fig_live.add_trace(go.Scatter(x=eval_df["predicted_for"], y=eval_df["predicted_close"], name="Predicted"))
                fig_live.update_layout(template="plotly_dark", height=450, title="Forward Test: Actual vs Predicted")
                st.plotly_chart(fig_live, use_container_width=True)
            else:
                st.info("Logs exist but none have actual prices yet (Pending).")

            st.subheader("All Logs (including pending)")
            st.dataframe(logs, use_container_width=True)

    # Latest 10 Predictions Table (keep this OUTSIDE the logs if/else)
    st.subheader("📋 Latest Predictions Comparison")
    pred_df = pd.DataFrame({
        "Actual Price": y_test.flatten()[-10:],
        "Predicted Price": predictions.flatten()[-10:]
    })
    st.dataframe(pred_df, use_container_width=True)

    st.markdown("---")
    st.write("Built with ❤️ using Streamlit, TensorFlow, and Plotly")

except Exception as e:
    st.error("An Application Error Occurred")
    st.exception(e)
