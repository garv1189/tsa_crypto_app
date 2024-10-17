import pandas as pd
from prophet import Prophet
import joblib
import os


def load_data(filename='processed_crypto_data.csv'):
    return pd.read_csv(filename, parse_dates=['Date'])


def create_prophet_model(data, symbol):
    crypto_data = data[data['Symbol'] == symbol][['Date', 'Close']].copy()
    crypto_data.columns = ['ds', 'y']
    crypto_data['ds'] = crypto_data['ds'].dt.tz_localize(None)  # Remove timezone

    model = Prophet()
    model.fit(crypto_data)

    return model


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    return joblib.load(filename)


def forecast_future(model, steps=30):
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast.tail(steps)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def create_models_for_all_cryptos():
    df = load_data()
    symbols = df['Symbol'].unique()

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    for symbol in symbols:
        print(f"Creating Prophet model for {symbol}")
        model = create_prophet_model(df, symbol)
        save_model(model, f'models/{symbol}_prophet_model.joblib')

    print("All models created and saved successfully.")


if __name__ == "__main__":
    create_models_for_all_cryptos()

    # Example of loading a model and making predictions
    symbol = 'BTC-USD'  # You can change this to any available symbol
    loaded_model = load_model(f'models/{symbol}_prophet_model.joblib')
    future_predictions = forecast_future(loaded_model)
    print(f"Future predictions for {symbol}:")
    print(future_predictions)