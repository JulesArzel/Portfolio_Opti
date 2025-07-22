import numpy as np
import pandas as pd

def get_markowitz_weights(prices: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Compute daily Markowitz tangent portfolio weights for a given price DataFrame.
    Compared to the notebook, we use the closed form formula for the tangent portfolio weights and return otpimal weights
    for each day, on a 60 rolling days basis.

    Parameters:
    - prices: DataFrame with dates as index and stock prices as columns.
    - risk_free_rate: Optional float, default 0.0

    Returns:
    - DataFrame of daily weights (same index, same columns as prices)
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.rolling(window=60).mean()
    cov_matrices = returns.rolling(window=60).cov().dropna()

    weights_list = []

    for i in range(59, len(returns)):
        mu = mean_returns.iloc[i]
        cov = cov_matrices.iloc[i * len(prices.columns):(i + 1) * len(prices.columns)]
        cov_matrix = cov.values.reshape(len(prices.columns), len(prices.columns))

        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(mu))
            excess_returns = mu - risk_free_rate

            weights = inv_cov @ excess_returns
            weights /= np.sum(weights)

        except np.linalg.LinAlgError:
            weights = np.full(len(mu), 1 / len(mu))

        weights_list.append(weights)

    aligned_index = returns.index[59:]
    weights_df = pd.DataFrame(weights_list, index=aligned_index, columns=prices.columns)

    return weights_df
