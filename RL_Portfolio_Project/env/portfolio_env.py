import gym
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, features, returns, initial_cash=1.0):
        self.features = features
        self.returns = returns
        self.initial_cash = initial_cash
        self.n_assets = returns.shape[1]
        self.reset()

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.weights = np.ones(self.n_assets) / self.n_assets  # Start equally weighted
        return self.features.iloc[self.current_step].values

    def step(self, action):
        # action = weights over assets
        prev_weights = self.weights
        self.weights = np.clip(action, 0, 1)
        self.weights /= np.sum(self.weights)  # normalize

        r_t = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(self.weights, r_t)
        self.portfolio_value *= (1 + portfolio_return)

        self.current_step += 1
        done = self.current_step >= len(self.features) - 1

        next_state = self.features.iloc[self.current_step].values
        reward = portfolio_return  # or risk-adjusted
        info = {"portfolio_value": self.portfolio_value}

        return next_state, reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Value: {self.portfolio_value:.4f}")
