#import the Libraries
import numpy as np
import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

Tesla_df = pd.read_csv('../Data/TSLA_historical_data.csv')
Vanguard_BND_df = pd.read_csv('../Data/BND_historical_data.csv')
SPY_df = pd.read_csv('../Data/SPY_historical_data.csv')


# Define the tickers and date range
tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-01-01"
end_date = "2024-10-31"

def fetch_data(): 
    # Fetch and save data for each ticker
    for ticker in tickers:
        # Download historical data for each ticker
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Save each ticker's data to a CSV file in the specified directory
        file_path = os.path.join(f"{ticker}_historical_data.csv")
        df.to_csv(file_path)
        print(f"Data for {ticker} saved to {file_path}")

def visual_adjusted_close():
    # Select Adjusted Close prices for each asset
    prices = pd.DataFrame({
        'TSLA': Tesla_df['Adj Close'],
        'BND': Vanguard_BND_df['Adj Close'],
        'SPY': SPY_df['Adj Close']
    })

    # Drop rows with missing values
    prices.dropna(inplace=True)
    return prices
    

def data_cleaning(price):
    print(price.isnull().sum())

def percentage_change(prices):
    returns = prices.pct_change().dropna()
    return returns

def analysis_volatility(prices):
    rolling_mean = prices['TSLA'].rolling(window=30).mean()
    rolling_std = prices['TSLA'].rolling(window=30).std()
    return rolling_mean ,rolling_std

def detect_outlier(returns):
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = returns[(z_scores.abs() > 3)]
    return outliers

def scaling(prices):
    scaler = MinMaxScaler()
    prices_scaled = pd.DataFrame(scaler.fit_transform(prices), columns=prices.columns, index=prices.index)
    print(prices_scaled)

def analysis_days(returns):
    high_returns = returns[returns['TSLA'] > returns['TSLA'].quantile(0.95)]
    low_returns = returns[returns['TSLA'] < returns['TSLA'].quantile(0.05)]
    print(f"Low returns \n{low_returns}\n High returns\n {high_returns}")

def seasonality_trand(prices):
    decomposition = seasonal_decompose(prices['TSLA'], model='multiplicative', period=365)
    return decomposition

def value_risk(returns):
    # Calculate the 5% VaR for Tesla’s daily returns
    var_95 = returns['TSLA'].quantile(0.05)
    print(f"5% VaR for TSLA: {var_95}")

def sharped_ratio(returns):
    sharpe_ratio = returns['TSLA'].mean() / returns['TSLA'].std() * np.sqrt(252)  # Annualized
    print(f"Annualized Sharpe Ratio for TSLA: {sharpe_ratio}")