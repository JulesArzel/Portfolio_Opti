import pandas as pd
import numpy as np
from features.markowitz import get_markowitz_weights
from features.regime import get_hmm_regimes, predict_future_regimes
from data.data import get_data 

def one_hot_encode(series, prefix):
    return pd.get_dummies(series, prefix=prefix)

def add_technical_features(prices, returns, window=5):
    
    vol = returns.rolling(window).std()
    vol.columns = [f'vol_{col}' for col in vol.columns]

    momentum = prices.pct_change(periods=window)
    momentum.columns = [f'mom_{col}' for col in momentum.columns]

    avg_return = returns.rolling(window).mean()
    avg_return.columns = [f'avg_ret_{col}' for col in avg_return.columns]

    zscore = (returns - returns.rolling(window).mean()) / returns.rolling(window).std()
    zscore.columns = [f'zscore_{col}' for col in zscore.columns]

    return pd.concat([vol, momentum, avg_return, zscore], axis=1)

def build_rl_features(tickers=['AAPL', 'XOM', 'JNJ', 'TSLA', 'F', 'MSFT'],
                      start_date='2020-01-01', end_date='2025-01-01',
                      tech_window=5):
    
    prices = get_data(tickers, start_date, end_date)
    returns = prices.pct_change().dropna()

    markowitz_weights = get_markowitz_weights(prices)
    hmm_regimes = get_hmm_regimes(prices)
    predicted_regimes = predict_future_regimes(prices)

    hmm_ohe = one_hot_encode(hmm_regimes, prefix='hmm')
    pred_ohe = one_hot_encode(predicted_regimes, prefix='pred')

    tech_features = add_technical_features(prices, returns, window=tech_window)

    features_df = pd.concat([hmm_ohe, pred_ohe, markowitz_weights, tech_features], axis=1)
    features_df = features_df.dropna()

    returns_df = returns.loc[features_df.index]

    assert features_df.shape[0] == returns_df.shape[0]
    assert not features_df.isnull().any().any()
    assert not returns_df.isnull().any().any()

    return returns_df, features_df
