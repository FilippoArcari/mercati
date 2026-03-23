import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_predictions(predictions: pd.DataFrame, targets: pd.DataFrame, step: str, results_dir: str) -> None:
    """
    Valuta le prestazioni del modello calcolando metriche di errore e visualizzando i risultati.

    Args:
        predictions (np.ndarray): Le previsioni del modello.
        targets (np.ndarray): I valori reali.
        step (str): La fase di valutazione ('train' o 'test').
        results_dir (str): La directory in cui salvare i grafici.

    
    """
    difference = predictions - targets
    
    difference["mae"] = difference.abs().mean(axis=1)
    difference["mse"] = (difference ** 2).mean(axis=1)
    difference["rmse"] = difference["mse"] ** 0.5
    
    os.makedirs(f"{results_dir}/{step}", exist_ok=True)
    for ticker in predictions.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(predictions.index, predictions[ticker], label='Predizioni', color='blue')
        plt.plot(targets.index, targets[ticker], label='Valori Reali', color='orange')
        plt.title(f'Predizioni vs Valori Reali per {ticker}')
        plt.text(0.02, 0.95, f'Max Error:{difference["mae"].max() :.2f}\n Mean Error:{difference["mae"].mean():.2f}\n Min Error:{difference["mae"].min():.2f}', 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.xlabel('Data')
        plt.ylabel('Prezzo')
        plt.legend()
        plt.grid()
        plt.savefig(f"{results_dir}/{step}/{ticker}.png")
        plt.close() 