import numpy as np
import pandas as pd

def get_markowitz_weights(prices: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:

    returns = prices.pct_change().dropna()
    mean_returns = returns.rolling(window=60).mean()
    cov_matrices = returns.rolling(window=60).cov().dropna()

    weights_list = []

    for i in range(59, len(returns)):
        mu = mean_returns.iloc[i]

        date = returns.index[i]
        try:
            cov_matrix = cov_matrices.xs(date, level=0).values
            inv_cov = np.linalg.inv(cov_matrix)
            excess_returns = mu - risk_free_rate
            weights = inv_cov @ excess_returns
            weights /= np.sum(weights)
        except (KeyError, np.linalg.LinAlgError):
            weights = np.full(len(mu), 1 / len(mu))  # fallback to equal weights

        weights_list.append(weights)


    aligned_index = returns.index[59:]
    weights_df = pd.DataFrame(weights_list, index=aligned_index, columns=prices.columns)

    return weights_df
