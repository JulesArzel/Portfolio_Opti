import numpy as np
import pandas as pd

def get_markowitz_weights(prices: pd.DataFrame, risk_free_rate: float = 0.0, window: int = 60) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    mean_returns = returns.rolling(window=window).mean()
    cov_matrices = returns.rolling(window=window).cov().dropna()

    weights_list = []
    index_list = []

    for i in range(window, len(returns)):
        mu = mean_returns.iloc[i - 1]
        date = returns.index[i]  
        try:
            cov_matrix = cov_matrices.xs(returns.index[i - 1], level=0).values

            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-3

            inv_cov = np.linalg.inv(cov_matrix)
            excess_returns = mu - risk_free_rate

            weights = inv_cov @ excess_returns
            weights = weights / np.sum(np.abs(weights))

            weights = np.clip(weights, -1.0, 1.0)
        except (KeyError, np.linalg.LinAlgError):
            weights = np.full(len(mu), 1 / len(mu))  # equal weights fallback

        weights_list.append(weights)
        index_list.append(date)

    weights_df = pd.DataFrame(weights_list, index=index_list, columns=prices.columns)
    return weights_df
