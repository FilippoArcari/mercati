import hydra
from omegaconf import OmegaConf
from modelli.predictor.get_data import get_data
import numpy as np

with hydra.initialize(version_base=None, config_path="config"):
    cfg = hydra.compose(config_name="config")
    
    df, n_context_features, scaler = get_data(
        cfg.data.tickers, cfg.data.start_date, cfg.data.end_date, 
        cfg.frequency, cfg.data.inflation_series, cfg.data.split_train
    )
    
    print("df head scaled:\n", df.iloc[:3, :3])
    
    # inverse transform df
    df_rescaled = scaler.inverse_transform(df.values)
    print("df head rescaled:\n", df_rescaled[:3, :3])
