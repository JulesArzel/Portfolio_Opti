from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def train_rl_agent(env, timesteps=100_000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model
