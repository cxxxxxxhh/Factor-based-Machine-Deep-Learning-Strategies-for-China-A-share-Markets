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

# Model Import
from IP_RNN import RNN
from IP_GRU import GRU
from IP_LSTM import LSTM

# Data Import
from IP_Data import resample_to_week, standardize_by_date_total, standardize_by_date, cal_label, load_all_data, load_data_window, train_val_split, preprocess_data, preprocess_data_randomstock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, optimizer):
    """
    模型训练函数
    """
    criterion = nn.MSELoss()
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # loss = ic_loss(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
def evaluate_model(model, val_loader):
    """
    模型验证函数
    """
    criterion = nn.MSELoss()
    total_val_size = 0
    total_val_loss = 0
    model.eval()
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss = ic_loss(outputs, labels)
            total_val_loss += loss.item() * labels.shape[0]
            total_val_size += labels.shape[0]
    epoch_val_loss = total_val_loss / total_val_size
    return epoch_val_loss
def calculate_rank_ic(y_target_df):
    grouped = y_target_df.groupby(level='Date').apply(lambda x: spearmanr(x['y_pred'], x['y_target'])[0])
    return grouped.mean(), grouped.std()
def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """
    超参数优化函数，返回最佳配置。
    """
    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 96, 128, 160, 192, 224, 256, 288, 320])
        learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.005, 0.01, 0.03, 0.05]) 
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096, 8192, 16384])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3, step=0.05)
        
        # 更新配置
        best_config = {
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate
        }

        # 训练模型并返回验证损失
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_config["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=best_config["batch_size"], shuffle=False)
        if mode == "RNN":
            model = RNN(common_configs["input_size"], best_config["hidden_size"], common_configs["num_layers"], common_configs["output_size"], best_config["dropout_rate"]).to(device)
        elif mode == "LSTM":
            model = LSTM(common_configs["input_size"], best_config["hidden_size"], common_configs["num_layers"], common_configs["output_size"], best_config["dropout_rate"]).to(device)
        elif mode == "GRU":
            model = GRU(common_configs["input_size"], best_config["hidden_size"], common_configs["num_layers"], common_configs["output_size"], best_config["dropout_rate"]).to(device)
        optimizer = optim.Adam(model.parameters(), best_config["learning_rate"])
        best_val_loss = float("inf")
        for epoch in range(20):
            train_model(model, train_loader, optimizer)
            val_loss = evaluate_model(model, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 清理模型和数据加载器
        del model
        del train_loader
        del val_loader
        gc.collect()
        torch.cuda.empty_cache()
        return best_val_loss
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)  # 设定试验次数
    return study.best_params
def rolling_train_predict(path, train_start_date, train_end_date, pred_start_date, pred_end_date, half_year_offset, common_configs, mode):
    model_save_dir = f"models/{mode}/MSEFunc/(randomstock_分开标准化_预测半年)"
    pred_save_dir = f"predictions/{mode}/MSEFunc/(randomstock_分开标准化_预测半年)"
    log_save_dir = f"logs/{mode}/MSEFunc"
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)
    out_sample_csv_path = os.path.join(pred_save_dir, "out_sample_predictions.csv")
    in_sample_csv_path = os.path.join(pred_save_dir, "in_sample_predictions.csv")
    ic_summary_csv_path = os.path.join(pred_save_dir, "model_summary.csv")
    # 用于回测的文件
    back_test_csv_path = os.path.join(pred_save_dir, "prediction_xgb_outsample.csv")

    if not os.path.exists(in_sample_csv_path):
        pd.DataFrame(columns=["y_target", "y_pred"], index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Symbol"])).to_csv(in_sample_csv_path, index=True)
    if not os.path.exists(out_sample_csv_path):
        pd.DataFrame(columns=["y_target", "y_pred"], index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Symbol"])).to_csv(out_sample_csv_path, index=True)
    if not os.path.exists(back_test_csv_path):
        pd.DataFrame(columns=["pred"], index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Symbol"])).to_csv(back_test_csv_path, index=True)
    if not os.path.exists(ic_summary_csv_path):
        pd.DataFrame(columns=[
            "pred_start_date", "hidden_size", "learning_rate", "batch_size", 
            "dropout_rate", "in_sample_ic", "out_sample_ic", "ic_difference(out - in)"
        ]).to_csv(ic_summary_csv_path, index=False)
    
    window_count = 1
    while train_end_date < pd.Timestamp("2024-07-01"):
        # Set up logging
        log_file = "(randomstock_分开标准化_预测半年).log"
        log_path = os.path.join(log_save_dir, log_file)
        logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Rolling Window {window_count}: Training from {train_start_date.date()} to {train_end_date.date()}, Predicting {pred_start_date.date()} to {pred_end_date.date()}")
    
        train_window, pred_window = load_data_window(path, train_start_date, train_end_date, pred_start_date, pred_end_date)
        train_set, val_set = train_val_split(train_window)
        X_train, y_train, X_val, y_val, X_insample, y_insample, y_insample_df, X_test, y_target, y_target_df = preprocess_data_randomstock(train_set, val_set, pred_window, time_steps=5)
        logging.info("开始超参调优：")
        best_configs = optimize_hyperparameters(X_train, y_train, X_val, y_val)
        logging.info("Model configuration: %s", best_configs)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_configs["batch_size"], shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=best_configs["batch_size"], shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_target), batch_size=best_configs["batch_size"], shuffle=False)
        insample_loader = DataLoader(TensorDataset(X_insample, y_insample), batch_size=best_configs["batch_size"], shuffle=False)
        if mode == "RNN":
            model = RNN(common_configs["input_size"], best_configs["hidden_size"], common_configs["num_layers"], common_configs["output_size"], best_configs["dropout_rate"]).to(device)
        elif mode == "LSTM":
            model = LSTM(common_configs["input_size"], best_configs["hidden_size"], common_configs["num_layers"], common_configs["output_size"], best_configs["dropout_rate"]).to(device)
        elif mode == "GRU":
            model = GRU(common_configs["input_size"], best_configs["hidden_size"], common_configs["num_layers"], common_configs["output_size"], best_configs["dropout_rate"]).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=best_configs["learning_rate"])
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        stop_steps = 0
        num_epochs = 100
        early_stop = 10
        for epoch in range(num_epochs):
            logging.info("Epoch: %d", epoch)
            train_model(model, train_loader, optimizer)
            val_loss = evaluate_model(model, val_loader)
            logging.info("val loss: %.4f", val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_steps = 0
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
            else:
                stop_steps += 1
                if stop_steps >= early_stop:
                    logging.info("Early stopping triggered.")
                    break
            
        model_save_path = os.path.join(model_save_dir, f"model_window_{pred_start_date.date()}.pth")
        torch.save(best_model_state, model_save_path)
        logging.info("best model loss: %.4f, best model epoch: %d", best_val_loss, best_epoch)
        model.load_state_dict(best_model_state)
        model.eval()
        predictions_insample = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in insample_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions_insample.append(outputs.cpu().numpy())
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
        predictions_insample_np = np.concatenate(predictions_insample, axis=0)
        y_insample_df['y_pred'] = predictions_insample_np
        predictions_np = np.concatenate(predictions, axis=0)
        y_target_df['y_pred'] = predictions_np
        ic_mean_insample, ic_std_insample = calculate_rank_ic(y_insample_df)
        ic_mean_outsample, ic_std_outsample = calculate_rank_ic(y_target_df)
        logging.info("Window %d, Rank IC insample mean: %.4f, Rank IC insample std: %.4f", window_count, ic_std_insample, ic_mean_insample)
        logging.info("Window %d, Rank IC outsample mean: %.4f, Rank IC outsample std: %.4f", window_count, ic_std_outsample, ic_mean_outsample)
        predictions_save_path = os.path.join(pred_save_dir, f"predictions_window_{pred_start_date.date()}.csv")
        y_target_df.to_csv(predictions_save_path, index=True)
        
        # Append predictions to CSV
        y_insample_df.to_csv(in_sample_csv_path, mode='a', header=False, index=True)
        y_target_df.to_csv(out_sample_csv_path, mode='a', header=False, index=True)
        y_target_df['y_pred'].to_csv(back_test_csv_path, mode='a', header=False, index=True)
        # Append to IC summary DataFrame
        model_summary_row = pd.DataFrame({
            "pred_start_date": [pred_start_date.date()],
            "hidden_size": [best_configs["hidden_size"]],
            "learning_rate": [best_configs["learning_rate"]], 
            "batch_size": [best_configs["batch_size"]], 
            "dropout_rate": [best_configs["dropout_rate"]],
            "in_sample_ic": [ic_std_insample],
            "out_sample_ic": [ic_std_outsample],
            "ic_difference(out - in)": [ic_std_outsample - ic_std_insample]
        })
        model_summary_row.to_csv(ic_summary_csv_path, mode='a', header=False, index=False)


        torch.cuda.empty_cache()
        train_start_date += half_year_offset
        train_end_date += half_year_offset
        pred_start_date += half_year_offset
        pred_end_date += half_year_offset
        window_count += 1
    logging.info("Rolling train-predict completed.")
    in_sample_predictions_parquet_path = os.path.join(pred_save_dir, "in_sample_predictions.parquet")
    out_sample_predictions_parquet_path = os.path.join(pred_save_dir, "out_sample_predictions.parquet")
    back_test_parquet_path = os.path.join(pred_save_dir, "prediction_xgb_outsample.parquet")
    pd.read_csv(in_sample_csv_path, index_col=0).to_parquet(in_sample_predictions_parquet_path, index=True)
    pd.read_csv(out_sample_csv_path, index_col=0).to_parquet(out_sample_predictions_parquet_path, index=True)
    pd.read_csv(back_test_csv_path, index_col=0).to_parquet(back_test_parquet_path, index=True)
    logging.info("All predictions saved as Parquet.")

if __name__ == "__main__":
    path = '/home/hchenef/Code/factor_dat'
    # 初始训练和预测日期和模型
    mode = "GRU"
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
    rolling_train_predict(path, train_start_date, train_end_date, pred_start_date, pred_end_date, half_year_offset, common_configs, mode)