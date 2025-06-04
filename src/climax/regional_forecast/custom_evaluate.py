import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


######## Evaluating the model ##############
def evaluation_metrics(pred_test, y_test, feature_titles):
    print(f"\n------------ Evaluation Metrics ------------")
    print(f"MAE: {mean_absolute_error(y_test, pred_test):.4f}")
    print(f"MSE: {mean_squared_error(y_test, pred_test):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")
    print(f"R²: {r2_score(y_test, pred_test):.4f}")

    print(f"\n---------- Per Feature Evaluation ----------")
    
    mse_values = mean_squared_error(y_test, pred_test, multioutput='raw_values')
    rmse_values = np.sqrt(mse_values)


    ranges = y_test.max(axis=0) - y_test.min(axis=0)
    ranges = np.where(ranges == 0, 1e-8, ranges)  
    nrmse_range = rmse_values / ranges


    std_dev = y_test.std(axis=0)
    std_dev = np.where(std_dev == 0, 1e-8, std_dev)
    nrmse_std = rmse_values / std_dev


    mean_y = np.nanmean(y_test, axis=0)
    mean_y_safe = np.where(np.abs(mean_y) < 1e-8, 1e-8, mean_y)  # avoid zero mean
    nrmse_mean = rmse_values / mean_y_safe


    ## Percentage Error Calculation ##
    safe_y_test = np.where(y_test == 0, np.nan, y_test)
    percentage_errors = np.abs((y_test - pred_test) / np.abs(safe_y_test)) * 100

    # Compute average percentage error per feature
    avg_percentage_error = np.nanmean(percentage_errors, axis=0)

    # Suppress error% for features with very small mean values
    mean_y = np.nanmean(safe_y_test, axis=0)
    error_threshold = 1  # or adjust based on your data's scale

    # Mask error% where mean target is too small
    avg_percentage_error = np.where(np.abs(mean_y) < error_threshold, np.nan, avg_percentage_error)


    ### Symmetric Mean Absolute Percentage Error ########
    denominator = (np.abs(y_test) + np.abs(pred_test)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.abs(y_test - pred_test) / denominator * 100
    avg_smape = np.nanmean(smape, axis=0)


    metrics_table = pd.DataFrame({
    'Feature': feature_titles,
    'MSE': [f"{mse:.4f}" for mse in mse_values],
    'RMSE': [f"{rmse:.4f}" for rmse in rmse_values],
    'NRMSE_range': [f"{nrmse:.4f}" for nrmse in nrmse_range],
    'NRMSE_std': [f"{nrmse:.4f}" for nrmse in nrmse_std],
    'NRMSE_mean': [f"{nrmse:.4f}" for nrmse in nrmse_mean],
    'Error%': [f"{perror:.2f}" if not np.isnan(perror) else "--" for perror in avg_percentage_error],
    'sMAPE%': [f"{val:.2f}" for val in avg_smape],
    'Mean': [f"{mean:.2f}" for mean in mean_y]
    })

    print(metrics_table.to_string(index=False))
    print(f"--------------------------------------------\n")



######## Action Code ##############
def evaluate_model():

    pred_test = np.load("results/pred_test.npy")
    y_test = np.load("results/y_test.npy")

    with open("./finetuning_1.40625deg/weather_feature_titles.txt", "r") as f:
        feature_titles = [line.strip() for line in f if line.strip()]

    # Call your evaluation
    evaluation_metrics(pred_test, y_test, feature_titles)
