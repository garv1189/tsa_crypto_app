import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os


def load_data(filename='processed_crypto_data.csv'):
    df = pd.read_csv(filename, parse_dates=['Date'])
    df['Date'] = df['Date'].dt.tz_localize(None)  # Remove timezone information
    return df.set_index('Date')


def plot_time_series(df, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df[df['Symbol'] == symbol]['Close'], label='Close Price')
    plt.plot(df[df['Symbol'] == symbol]['MA5'], label='5-day MA')
    plt.plot(df[df['Symbol'] == symbol]['MA7'], label='7-day MA')
    plt.title(f'{symbol} Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    return plt


def plot_acf_pacf(df, symbol):
    crypto_data = df[df['Symbol'] == symbol]['Close']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(crypto_data, ax=ax1, lags=40)
    ax1.set_title(f'ACF - {symbol}')

    plot_pacf(crypto_data, ax=ax2, lags=40)
    ax2.set_title(f'PACF - {symbol}')

    plt.tight_layout()
    return plt


def create_visualizations():
    df = load_data()
    symbols = df['Symbol'].unique()

    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)

    for symbol in symbols:
        print(f"Creating visualizations for {symbol}")

        # Time series plot
        time_series_plot = plot_time_series(df, symbol)
        time_series_plot.savefig(f'visualizations/{symbol}_time_series.png')
        plt.close()

        # ACF and PACF plots
        acf_pacf_plot = plot_acf_pacf(df, symbol)
        acf_pacf_plot.savefig(f'visualizations/{symbol}_acf_pacf.png')
        plt.close()

    print("All visualizations created successfully.")


if __name__ == "__main__":
    create_visualizations()