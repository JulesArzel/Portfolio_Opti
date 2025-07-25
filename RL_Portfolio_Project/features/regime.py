import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def get_hmm_regimes(prices: pd.DataFrame, n_states: int = 3) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1)).dropna()
    X = log_returns.mean(axis=1).values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(X)
    hidden_states = model.predict(X)

    return pd.Series(hidden_states, index=log_returns.index, name='HMM_Regime')

def predict_future_regimes(prices: pd.DataFrame, n_states: int = 3) -> pd.Series:

    log_returns = np.log(prices / prices.shift(1)).dropna()

    rolling_window = 10
    features = pd.DataFrame(index=log_returns.index)
    features['mean_return'] = log_returns.mean(axis=1)
    features['volatility'] = log_returns.rolling(rolling_window).std().mean(axis=1)
    features['momentum_5'] = prices.pct_change(5).mean(axis=1)
    features['momentum_10'] = prices.pct_change(10).mean(axis=1)

    features = features.dropna()

    regimes = get_hmm_regimes(prices, n_states=n_states)
    aligned = features.join(regimes, how='inner')

    X = aligned.drop(columns='HMM_Regime')
    y = aligned['HMM_Regime'].shift(-1).dropna()
    X = X.iloc[:-1] 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_scaled)

    return pd.Series(y_pred, index=X.index, name='Predicted_Regime')
