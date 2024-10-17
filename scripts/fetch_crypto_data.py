import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_crypto_data(symbols=['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'ADA-USD'],
                      start_date='2020-01-01',
                      end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        df['Symbol'] = symbol
        data[symbol] = df

    return data

def save_data(data, filename='crypto_data.csv'):
    combined_data = pd.concat(data.values(), axis=0)
    combined_data.to_csv(filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    crypto_data = fetch_crypto_data()
    save_data(crypto_data)