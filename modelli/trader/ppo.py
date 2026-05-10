from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from modelli.trader.traiding_env import MultiTickerTradingEnv
import numpy as np
import pandas as pd

class PPOTrader:
    def __init__(self, env: MultiTickerTradingEnv,config: dict = {}):
        check_env(env)
        self.env   = env
        self.total_timesteps = config.get("total_timesteps", 10000)
        self.model = PPO("MlpPolicy", env, verbose=1)

    def train(self, ):
        self.model.learn(total_timesteps=self.total_timesteps,progress_bar=True,log_interval=100,reset_num_timesteps=True)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    def test(self, env: MultiTickerTradingEnv, render: bool = False) -> pd.DataFrame:
        obs, _ = env.reset()
        done   = False

        while not done:
            action                          = self.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)  # ← unpack corretto
            done = terminated or truncated

            if render and (env.step_idx+1) % env.log_every == 0:
                env.render()
        
        history = env.get_history()
        
        return history

