from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from modelli.trader.traiding_env import MultiTickerTradingEnv
import numpy as np
import pandas as pd
import os

class PPOTrader:
    def __init__(self, env: MultiTickerTradingEnv, config: dict = {}, model_path: str = None):
        check_env(env)
        self.env   = env
        self.total_timesteps = config.get("total_timesteps", 10000)
        
        # Se esiste un modello salvato, caricalo; altrimenti creane uno nuovo
        if model_path and os.path.exists(model_path):
            print(f"Loading PPO model from: {model_path}")
            self.model = PPO.load(model_path, env=env)
        else:
            self.model = PPO("MlpPolicy", env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=True, log_interval=100, reset_num_timesteps=True)

    def save(self, path: str):
        """Salva i pesi del modello PPO"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        print(f"[OK] PPO model saved to: {path}")

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

            if render:
                env.render(only_on_trade=True)
        
        history = env.get_history()
        
        return history

