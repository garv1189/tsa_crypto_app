import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from create_forecast_model_prophet import load_model, forecast_future, load_data
import os

def main():
    st.title("Cryptocurrency Time Series Analysis and Forecasting")

    # Load data
    df = load_data()
    symbols = df['Symbol'].unique()

    # Sidebar for user input
    selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", symbols)

    # Display time series plot
    st.header(f"Time Series Analysis - {selected_symbol}")
    time_series_plot = plt.figure(figsize=(12, 6))
    plt.imshow(plt.imread(f'visualizations/{selected_symbol}_time_series.png'))
    plt.axis('off')
    st.pyplot(time_series_plot)

    # Display ACF and PACF plots
    st.header(f"ACF and PACF - {selected_symbol}")
    acf_pacf_plot = plt.figure(figsize=(12, 10))
    plt.imshow(plt.imread(f'visualizations/{selected_symbol}_acf_pacf.png'))
    plt.axis('off')
    st.pyplot(acf_pacf_plot)

    # Load model and make predictions
    st.header("Future Predictions")
    model = load_model(f'models/{selected_symbol}_prophet_model.joblib')
    future_days = st.slider("Select number of days for prediction", 1, 60, 30)
    predictions = forecast_future(model, steps=future_days)

    # Plot predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictions['ds'], predictions['yhat'], label='Predicted')
    ax.fill_between(predictions['ds'], predictions['yhat_lower'], predictions['yhat_upper'], alpha=0.3)
    ax.set_title(f'{selected_symbol} Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Display predicted price
    last_prediction = predictions.iloc[-1]
    st.write(f"Predicted price for {last_prediction['ds'].date()}: ${last_prediction['yhat']:.2f}")
    st.write(f"Prediction range: ${last_prediction['yhat_lower']:.2f} - ${last_prediction['yhat_upper']:.2f}")

    # Display top gainer
    st.header("Top Gainer")
    latest_date = df['Date'].max()
    daily_returns = df[df['Date'] == latest_date].set_index('Symbol')['Returns']
    top_gainer = daily_returns.idxmax()
    st.write(f"Top gainer for {latest_date.date()}: {top_gainer} ({daily_returns[top_gainer]:.2%})")

    # Display dataset
    st.header(f"{selected_symbol} Dataset")
    st.dataframe(df[df['Symbol'] == selected_symbol])

if __name__ == "__main__":
    main()