import pandas as pd
import numpy as np
from features.markowitz import get_markowitz_weights
from features.regime import get_hmm_regimes, predict_future_regimes
from data.data import get_data 

def one_hot_encode(series, prefix):
    return pd.get_dummies(series, prefix=prefix)


def build_rl_features(tickers=['AAPL', 'XOM', 'JNJ', 'TSLA', 'F', 'MSFT'],start_date = '2020-01-01',end_date = '2025-01-01'):
    
    prices = get_data(tickers, start_date, end_date)
    returns = prices.pct_change().dropna()
    
    markowitz_weights = get_markowitz_weights(prices)
    hmm_regimes = get_hmm_regimes(prices)
    predicted_regimes = predict_future_regimes(prices)
    
    hmm_ohe = one_hot_encode(hmm_regimes, prefix='hmm')
    pred_ohe = one_hot_encode(predicted_regimes, prefix='pred')
    
    features_df = pd.concat([hmm_ohe, pred_ohe, markowitz_weights], axis=1)
    features_df = features_df.dropna() 
    returns_df = returns.loc[features_df.index]
    assert features_df.shape[0] == returns_df.shape[0]
    assert not features_df.isnull().any().any()
    assert not returns_df.isnull().any().any()
    
    return returns_df, features_df
