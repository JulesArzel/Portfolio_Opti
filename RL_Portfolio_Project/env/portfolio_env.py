import numpy as np
import pandas as pd
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    """
    Custom OpenAI Gym environment for financial portfolio management.
    The agent observes features and decides portfolio weights.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, features: pd.DataFrame, returns: pd.DataFrame, 
                 initial_cash: float = 1.0, allow_short: bool = False, verbose: bool = False):
        super(PortfolioEnv, self).__init__()

        assert features.shape[0] == returns.shape[0], "Feature and return data must align on time axis."
        assert features.index.equals(returns.index), "Feature and return index must match."

        self.features = features
        self.returns = returns
        self.initial_cash = initial_cash
        self.allow_short = allow_short
        self.verbose = verbose

        self.n_assets = returns.shape[1]
        self.n_features = features.shape[1]
        self.T = len(features)

        # Define action space: portfolio weights for each asset
        # If shorting allowed, weights can be negative
        action_low = -1.0 if allow_short else 0.0
        self.action_space = spaces.Box(low=action_low, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        # Observation = feature vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.last_weights = np.ones(self.n_assets) / self.n_assets  # start equally weighted
        self.history = [self.portfolio_value]

        return self._get_observation()

    def step(self, action):
        action = np.array(action)

        if not self.allow_short:
            action = np.clip(action, 0, 1)

        # Normalize weights to sum to 1
        if action.sum() == 0:
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = action / np.sum(action)

        daily_returns = self.returns.iloc[self.current_step].values  # vector of asset returns for the day
        portfolio_return = np.dot(weights, daily_returns)

        self.portfolio_value *= (1 + portfolio_return)
        self.history.append(self.portfolio_value)

        # Save state
        self.last_weights = weights
        self.current_step += 1

        done = self.current_step >= (self.T - 1)
        reward = portfolio_return  # could use log(1 + return), or penalized version (r-lambda*sigma)

        obs = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': weights,
            'portfolio_return': portfolio_return,
            'step': self.current_step
        }

        if self.verbose:
            print(f"Step {self.current_step}: return={portfolio_return:.4f}, value={self.portfolio_value:.4f}")

        return obs, reward, done, info

    def _get_observation(self):
        return self.features.iloc[self.current_step].values.astype(np.float32)

    def render(self, mode='human'):
        print(f"Step {self.current_step}, Portfolio Value: {self.portfolio_value:.4f}")
