#!/bin/bash

#This script is used to run pipeline on kaggle, with more epoch and more data

#Optimization 
python main.py step=optimize optuna.n_trials=150 optuna.max_epochs=500
#Train the model with the best hyperparameters found in the optimization step
 python main.py -m step=train,test prediction.epochs=500  prediction.batch_size=128 
#Trade the model on the test set and save the results
python main.py step=trade trader.total_timesteps=100000
python main.py step=trade_only trader.total_timesteps=100000