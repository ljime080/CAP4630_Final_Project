import streamlit as st
import pandas as pd
import plotly.express as px
from ensemble_forecasts import AttentionLayer , run_full_forecasting_pipeline, check_if_ticker_models_exist



# --- Page Config ---
st.set_page_config(page_title="📈 Stock Forecasting Ensemble", layout="centered")
st.title("📊 30-Day Stock Price Forecasting")

# --- Initialize session state ---
if "forecast_requested" not in st.session_state:
    st.session_state.forecast_requested = False
if "ticker" not in st.session_state:
    st.session_state.ticker = "TSLA"
if "has_models" not in st.session_state:
    st.session_state.has_models = {}

# --- Ticker input ---
st.session_state.ticker = st.text_input("Enter stock ticker:", value=st.session_state.ticker).upper()

# --- Forecast trigger ---
if st.button("Run Forecast"):
    st.session_state.forecast_requested = True
    ticker = st.session_state.ticker
    st.session_state.has_models[ticker] = check_if_ticker_models_exist(ticker)

# --- Forecast logic ---
if st.session_state.forecast_requested:
    ticker = st.session_state.ticker
    forecast_horizon = 30

    if st.session_state.has_models.get(ticker, False):
        st.success(f"🧠 Aha! We have models for {ticker}.")
        st.markdown("📡 Loading pre-trained models...")
        st.markdown("📈 Generating forecast...")
        spinner_msg = "Working on your forecast..."
    else:
        st.warning(f"⏳ No models found for {ticker}. Training from scratch...")
        st.markdown("🛠️ Training deep learning models...")
        st.markdown("🔮 Forecasting the next 30 days...")
        spinner_msg = "This may take a moment..."

    with st.spinner(spinner_msg):
        forecast_df, metrics_df = run_full_forecasting_pipeline(
            ticker=ticker,
            forecast_horizon=forecast_horizon,
            window_size=60,
            epochs=50
        )
        st.session_state.has_models[ticker] = True

    st.success(f"✅ Forecast for {ticker} completed!")
    st.balloons()

    # --- Display metrics ---
    st.subheader("📋 Model Metrics")
    st.dataframe(metrics_df)

    # --- Plot forecasts ---
    st.subheader("📈 Forecast Plot")
    fig = px.line(
        forecast_df,
        x="Date",
        y="Forecast",
        color="Model",
        title=f"Next 30-Day Forecast for {ticker}",
        markers=True
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Forecasted Price", legend_title="Model" )
    st.plotly_chart(fig, use_container_width=False)

    # --- Download CSV ---
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecasts as CSV", data=csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")