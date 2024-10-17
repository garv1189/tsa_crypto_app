import pandas as pd
import numpy as np


def load_and_clean_data(filename='crypto_data.csv'):
    df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')

    # Remove any duplicate entries
    df = df.drop_duplicates()

    # Handle missing values (if any)
    df = df.fillna(method='ffill')

    # Calculate daily returns
    df['Returns'] = df.groupby('Symbol')['Close'].pct_change()

    # Calculate volatility (30-day rolling standard deviation of returns)
    df['Volatility'] = df.groupby('Symbol')['Returns'].rolling(window=30).std().reset_index(0, drop=True)

    return df


def calculate_moving_averages(df):
    df['MA5'] = df.groupby('Symbol')['Close'].rolling(window=5).mean().reset_index(0, drop=True)
    df['MA7'] = df.groupby('Symbol')['Close'].rolling(window=7).mean().reset_index(0, drop=True)
    return df


def process_data(df):
    df = calculate_moving_averages(df)

    # Calculate other technical indicators here if needed

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    processed_df = process_data(df)
    processed_df.to_csv('processed_crypto_data.csv')
    print("Data processed and saved to processed_crypto_data.csv")