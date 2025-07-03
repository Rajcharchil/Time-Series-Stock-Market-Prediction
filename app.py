import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from datetime import datetime

# --- Streamlit Dark Theme & Custom CSS ---
st.set_page_config(page_title="📊 Stock Forecast Dashboard", layout="wide", page_icon="📈")
st.markdown("""
    <style>
    body, .main, .block-container {
        background-color: #18191A !important;
        color: #E4E6EB !important;
    }
    .stSidebar {
        background-color: #242526 !important;
    }
    .st-bb, .st-c6, .st-cg, .st-cj, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        background-color: #242526 !important;
    }
    .stMetric {
        background-color: #242526 !important;
        border-radius: 10px;
        padding: 12px;
        color: #F7C873 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .css-1v0mbdj, .css-1d391kg {
        background-color: #242526 !important;
    }
    .stSlider > div[data-baseweb="slider"] > div {
        background: #3A3B3C !important;
    }
    .stButton > button {
        background-color: #3A3B3C !important;
        color: #F7C873 !important;
    }
    .stSelectbox > div {
        background-color: #3A3B3C !important;
        color: #F7C873 !important;
    }
    .stDateInput > div {
        background-color: #3A3B3C !important;
        color: #F7C873 !important;
    }
    .stAlert-success {
        background-color: #1B5E20 !important;
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("""
    <h1 style='color:#F7C873;'>📈 Stock Forecast Dashboard using LSTM</h1>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("""
    <h2 style='color:#F7C873;'>⚙️ Configuration</h2>
""", unsafe_allow_html=True)
stock_list = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'SBIN.NS', 'INFY.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'BHARTIARTL.NS', 'ADANIENT.NS', 'HINDUNILVR.NS']
stock_symbol = st.sidebar.selectbox("Select Stock Ticker Codes", stock_list, help="Choose the stock to forecast.")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"), help="Select the start date for historical data.")
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"), help="Select the end date for historical data.")
train_size_ratio = st.sidebar.slider("Training Data %", 50, 95, 80, help="Percentage of data used for training.")
epochs = st.sidebar.slider("Epochs", 5, 100, 20, help="Number of epochs for LSTM training.")

# --- Predict Button ---
predict_btn = st.button("🔮 Predict", help="Click to run LSTM prediction and show results.")

# --- Data Loading ---
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df[['Close']].dropna()

if predict_btn:
    with st.spinner("📥 Loading data and running prediction..."):
        df = load_data(stock_symbol, start_date, end_date)
        if df is None or df.empty:
            st.error("❌ No data found for the selected stock and date range.")
            st.stop()

        st.success("✅ Data Loaded Successfully")

        # --- Metrics Cards ---
        if df is not None and not df.empty:
            latest_price = float(df['Close'].iloc[-1])
            first_price = float(df['Close'].iloc[0])
            change = latest_price - first_price
            change_pct = (change / first_price) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Price", f"₹{latest_price:,.2f}", f"{change:+.2f}")
            col2.metric("% Change", f"{change_pct:+.2f}%")
            col3.metric("Data Points", f"{len(df)}")
        else:
            st.warning("No data available for metrics.")

        # --- Interactive Plotly Chart: Historical Close ---
        st.subheader("📊 Historical Close Price")
        # DataFrame check
        st.write("Data preview:", df.head())
        st.write("Data shape:", df.shape)

        # if len(df) < 2:
        #     st.warning("Not enough data to plot the historical close price. Please select a valid date range with more data.")
        # else:
        #     fig_hist = go.Figure()
        #     fig_hist.add_trace(go.Scatter(
        #         x=df.index,
        #         y=df['Close'],
        #         mode='lines+markers',
        #         name='Close Price',
        #         line=dict(color='#00eaff', width=3),
        #         marker=dict(size=6, color='#00eaff', line=dict(width=0.5, color='#18191A')),
        #         hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>'
        #     ))
        #     fig_hist.update_layout(
        #         plot_bgcolor="#000000",
        #         paper_bgcolor="#4E3919",
        #         font=dict(color='#E4E6EB', size=18),
        #         xaxis=dict(title='Date', showgrid=True, gridcolor='#333'),
        #         yaxis=dict(title='Price (INR)', showgrid=True, gridcolor='#333'),
        #         hovermode='x unified',
        #         margin=dict(l=30, r=30, t=40, b=30),
        #         legend=dict(
        #             orientation='h',
        #             yanchor='bottom',
        #             y=1.02,
        #             xanchor='right',
        #             x=1,
        #             font=dict(color='#E4E6EB')
        #         )
        #     )
        #     st.plotly_chart(fig_hist, use_container_width=True)

        # --- Preprocessing ---
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        # --- Create sequences ---
        def create_dataset(dataset, seq_len):
            X, y = [], []
            for i in range(seq_len, len(dataset)):
                X.append(dataset[i - seq_len:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        SEQ_LEN = 60
        X, y = create_dataset(scaled_data, SEQ_LEN)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        split = int(len(X) * train_size_ratio / 100)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # --- LSTM Forecast vs Actual Chart with Loading Spinner ---
        st.subheader(f"📉 LSTM Forecast vs Actual for {stock_symbol}")
        with st.spinner("⏳ Loading LSTM Forecast vs Actual graph..."):
            # --- LSTM Model ---
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)))
            model.add(LSTM(64))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

            # --- Forecast ---
            predictions = model.predict(X_test)
            predicted_prices = scaler.inverse_transform(predictions)
            y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            # --- Interactive Plotly Chart: Forecast vs Actual ---
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=np.arange(len(y_test_scaled)), y=y_test_scaled.flatten(),
                mode='lines', name='Actual Price', line=dict(color='lime'),
                hovertemplate='Index: %{x}<br>Actual: ₹%{y:,.2f}<extra></extra>'
            ))
            fig_forecast.add_trace(go.Scatter(
                x=np.arange(len(predicted_prices)), y=predicted_prices.flatten(),
                mode='lines', name='Predicted Price', line=dict(color='#F7C873', dash='dash'),
                hovertemplate='Index: %{x}<br>Predicted: ₹%{y:,.2f}<extra></extra>'
            ))
            fig_forecast.update_layout(
                plot_bgcolor='#18191A',
                paper_bgcolor='#18191A',
                font=dict(color='#E4E6EB'),
                xaxis=dict(title='Time'),
                yaxis=dict(title='Stock Price (INR)'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        # --- RMSE Metric Card ---
        rmse = np.sqrt(mean_squared_error(y_test_scaled, predicted_prices))
        st.markdown(f"""
            <div class='stMetric'>
                <h3 style='color:#F7C873;'>📊 LSTM RMSE: {rmse:.2f}</h3>
            </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <hr style='border:1px solid #333;'>
    <div style='text-align:center; color:#888;'>
        Made with ❤️ using Streamlit, Plotly, and LSTM | <b>CHARCHIL</b>
    </div>
""", unsafe_allow_html=True)

#  streamlit run app.py
# venv310\\Scripts\\activate


