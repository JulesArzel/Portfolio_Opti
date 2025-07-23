from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def train_rl_agent(env, timesteps=100_000, verbose=1):
    """
    Trains a PPO agent on the given portfolio environment.

    Parameters:
    - env: instance of PortfolioEnv
    - timesteps: how long to train
    - verbose: verbosity level (0, 1)

    Returns:
    - Trained PPO model
    """
    vec_env = DummyVecEnv([lambda: env])  # wrap for SB3
    model = PPO("MlpPolicy", vec_env, verbose=verbose)
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_agent(model, env):
    """
    Evaluate a trained RL agent in the portfolio environment.

    Returns:
    - history: portfolio value over time
    - info_list: list of step-level info dicts
    """
    obs, _ = env.reset()
    done = False
    history = [env.portfolio_value]
    info_list = []

    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append(info['portfolio_value'])
        info_list.append(info)

    return history, info_list

