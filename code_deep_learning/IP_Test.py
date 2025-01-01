import gc
import os
import pandas as pd
from typing import List
import numpy as np
import logging

import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import optuna
from collections import OrderedDict

# Model Import
from IP_RNN import RNN
from IP_GRU import GRU
from IP_LSTM import LSTM

# Data Import
from IP_Data import resample_to_week, standardize_by_date_total, standardize_by_date, cal_label, load_all_data, load_data_window, train_val_split, preprocess_data, preprocess_data_randomstock

window_count = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = 'GRU'
path = '/home/hchenef/Code/factor_dat'
pred_save_dir = f"predictions/{mode}/MSEFunc/(周频训练 日频预测数据)"
os.makedirs(pred_save_dir, exist_ok=True)
train_start_date = pd.Timestamp("2010-01-01")
train_end_date = pd.Timestamp("2011-12-30")
pred_start_date = pd.Timestamp("2012-01-01")
pred_end_date = pd.Timestamp("2012-06-30")
half_year_offset = pd.DateOffset(months=6)
common_configs = {
        "input_size": 123,
        "output_size": 1,
        "num_layers": 2
    }
params_df = pd.read_csv(f'/home/hchenef/Code/predictions/{mode}/MSEFunc/(randomstock_分开标准化_预测半年)/model_summary.csv')
# 用于回测的文件
back_train_csv_path = os.path.join(pred_save_dir, f"train_{mode}_outsample.csv")
back_test_csv_path = os.path.join(pred_save_dir, f"test_{mode}_outsample.csv")
if not os.path.exists(back_train_csv_path):
    pd.DataFrame(columns=["pred"], index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Symbol"])).to_csv(back_train_csv_path, index=True)    
if not os.path.exists(back_test_csv_path):
    pd.DataFrame(columns=["pred"], index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Symbol"])).to_csv(back_test_csv_path, index=True)

while train_end_date < pd.Timestamp("2024-07-01"):
    hidden_dim = int(params_df['hidden_size'][window_count])
    dropout_dim = params_df['dropout_rate'][window_count]
    batch_dim = int(params_df['batch_size'][window_count])
    file = pred_start_date.date().strftime('%Y-%m-%d')
    train_window, pred_window = load_data_window(path, train_start_date, train_end_date, pred_start_date, pred_end_date)
    train_set, val_set = train_val_split(train_window)
    X_train, y_train, X_val, y_val, X_insample, y_insample, y_insample_df, X_test, y_target, y_target_df = preprocess_data_randomstock(train_set, val_set, pred_window, time_steps=5)
    test_loader = DataLoader(TensorDataset(X_test, y_target), batch_size = batch_dim, shuffle=False)
    model = GRU(input_size = common_configs["input_size"], hidden_size = hidden_dim, num_layers = common_configs["num_layers"], output_size = common_configs["output_size"], dropout_rate = dropout_dim)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model_dict = torch.load(f'/home/hchenef/Code/models/{mode}/MSEFunc/(randomstock_分开标准化_预测半年)/model_window_{file}.pth', weights_only=True, map_location=device)
    # new_state_dict = OrderedDict()
    # for k, v in model_dict.items():
    #     if k.startswith('module.'):
    #         k = k[7:]
    #     new_state_dict[k] = v
    model.load_state_dict(model_dict)
    model = model.to(device)
    # model = nn.DataParallel(model)
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
        # predictions = model(X_test.to(device)).squeeze()
    predictions_np = np.concatenate(predictions, axis=0)
    # predictions = predictions.cpu().numpy()
    y_target_df['y_pred'] = predictions_np
    y_target_df['y_target'].to_csv(back_train_csv_path, mode='a', header=False, index=True)
    y_target_df['y_pred'].to_csv(back_test_csv_path, mode='a', header=False, index=True)
    print(file)
    torch.cuda.empty_cache()
    train_start_date += half_year_offset
    train_end_date += half_year_offset
    pred_start_date += half_year_offset
    pred_end_date += half_year_offset
    window_count += 1

back_train_parquet_path = os.path.join(pred_save_dir, f"train_{mode}_outsample.parquet")
back_test_parquet_path = os.path.join(pred_save_dir, f"test_{mode}_outsample.parquet")
pd.read_csv(back_train_csv_path, index_col=0).to_parquet(back_train_parquet_path, index=True)
pd.read_csv(back_test_csv_path, index_col=0).to_parquet(back_test_parquet_path, index=True)

