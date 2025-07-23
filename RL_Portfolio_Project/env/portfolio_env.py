import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class PortfolioEnv(gym.Env):
    #Custom Gymnasium environment

    metadata = {'render_modes': ['human']}

    def __init__(self, features: pd.DataFrame, returns: pd.DataFrame, 
                 initial_cash: float = 1.0, allow_short: bool = False, 
                 render_mode=None, verbose: bool = False):
        super().__init__()

        assert features.shape[0] == returns.shape[0], "Feature and return data must align on time axis."
        assert features.index.equals(returns.index), "Feature and return index must match."

        self.rewards_window = deque(maxlen=20)  
        self.window_size = 20
        self.risk_aversion = 0.1 
        self.features = features
        self.returns = returns
        self.initial_cash = initial_cash
        self.allow_short = allow_short
        self.verbose = verbose
        self.render_mode = render_mode

        self.n_assets = returns.shape[1]
        self.n_features = features.shape[1]
        self.T = len(features)

        # Action: portfolio weights
        action_low = -1.0 if allow_short else 0.0
        self.action_space = spaces.Box(low=action_low, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        # Observation: feature vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        self.history = [self.portfolio_value]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        observation = self._get_observation()
        info = {"portfolio_value": self.portfolio_value}
        return observation, info

    def step(self, action):
        action = np.array(action)

        if not self.allow_short:
            action = np.clip(action, 0, 1)

        weights = action / np.sum(action) if action.sum() > 0 else self.last_weights

        daily_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(weights, daily_returns)

        self.portfolio_value *= (1 + portfolio_return)
        self.history.append(self.portfolio_value)

        self.last_weights = weights
        self.current_step += 1

        terminated = self.current_step >= (self.T - 1)
        truncated = False
        
        self.rewards_window.append(portfolio_return)

        if len(self.rewards_window) >= self.window_size:
            sigma = np.std(self.rewards_window)
        else:
            sigma = 0.0

        reward = portfolio_return - self.risk_aversion * sigma

        obs = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': weights,
            'portfolio_return': portfolio_return,
            'step': self.current_step
        }

        if self.verbose:
            print(f"Step {self.current_step}: return={portfolio_return:.4f}, value={self.portfolio_value:.4f}")

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        return self.features.iloc[self.current_step].values.astype(np.float32)

    def render(self):
        if self.render_mode == 'human':
            print(f"Step {self.current_step}, Portfolio Value: {self.portfolio_value:.4f}")
