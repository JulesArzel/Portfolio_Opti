import yfinance as yf
import pandas as pd

def get_data(tickers, start_date, end_date):
    '''tickers = ['AAPL', 'XOM', 'JNJ', 'TSLA', 'F', 'MSFT']'''
    '''tickers = ['AAPL', 'XOM', 'JNJ', 'TSLA', 'F', 'MSFT', 'CVX', 'PFE']'''
    '''start_date = "2020-01-01"
    end_date = "2025-01-01"'''

    data = yf.download(tickers, start=start_date, end=end_date)
    
    data = data[["Close"]].rename(columns={"Close": "price"})
    data.sort_index()
    data.dropna(inplace=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)

    return data