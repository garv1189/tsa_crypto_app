import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os

def load_data(filename='../data/processed/processed_crypto_data.csv'):
    return pd.read_csv(filename, parse_dates=['Date'], index_col='Date')


def create_sarima_model(data, symbol, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    crypto_data = data[data['Symbol'] == symbol]['Close']

    # Fit the SARIMA model
    model = SARIMAX(crypto_data, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)

    return fitted_model


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    return joblib.load(filename)


def forecast_future(model, steps=30):
    forecast = model.forecast(steps)
    return forecast


def create_models_for_all_cryptos():
    df = load_data()
    symbols = df['Symbol'].unique()

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    for symbol in symbols:
        print(f"Creating SARIMA model for {symbol}")
        model = create_sarima_model(df, symbol)
        save_model(model, f'models/{symbol}_sarima_model.joblib')

    print("All models created and saved successfully.")


if __name__ == "__main__":
    create_models_for_all_cryptos()