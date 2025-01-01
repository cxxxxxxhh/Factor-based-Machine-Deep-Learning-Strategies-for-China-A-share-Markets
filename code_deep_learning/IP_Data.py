from collections import deque
import logging
import os
import pandas as pd
from typing import List
import numpy as np
import random

import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
def resample_to_week(df):
    """周频重采样函数"""
    weekly_data = df.groupby('Symbol').resample('W-MON', level='Date').last()
    weekly_data = weekly_data.dropna()
    return weekly_data
def standardize_by_date_total(df: pd.DataFrame, columns_to_scale:List[str])-> pd.DataFrame:
    """
    根据天分组对数据进行截面z-score标准化处理。

    参数:
    - df: 未处理的数据集。
    - columns_to_scale: 要处理的列。

    返回:
    - pd.DataFrame: 标准化后的数据。
    """
    standardized_df = df.copy()
    for date, group in df.groupby('Date'):
        scaler = StandardScaler()
        standardized_values = scaler.fit_transform(group[columns_to_scale])
        standardized_df.loc[group.index, columns_to_scale] = standardized_values
    return standardized_df
def standardize_by_date(train_window: pd.DataFrame, pred_window: pd.DataFrame, columns_to_scale:List[str])-> pd.DataFrame:
    """
    根据天分组对数据进行截面z-score标准化处理。

    参数:
    - train_window & pred_window: 未处理数据的训练集和测试集。
    - columns_to_scale: 要处理的列。

    返回:
    - pd.DataFrame: 标准化后的数据。
    """
    standardized_train = train_window.copy()
    standardized_pred = pred_window.copy()
    for date, group in train_window.groupby('Date'):
        scaler = StandardScaler()
        standardized_values_train = scaler.fit_transform(group[columns_to_scale])
        standardized_train.loc[group.index, columns_to_scale] = standardized_values_train
    for date, group in pred_window.groupby('Date'):
        standardized_values_pred = scaler.transform(group[columns_to_scale])
        standardized_pred.loc[group.index, columns_to_scale] = standardized_values_pred
    return standardized_train, standardized_pred
def cal_label(df: pd.DataFrame, next: int, col_name = 'AdjVWAP')-> pd.DataFrame:
    """
    计算标签，并将结果替换至回当天的数据中。

    参数:
    - df: 未处理数据。
    - next: 预测的天数。eg: next = 1，则表示预测明天收益率。
    - col_name: 'AdjVWAP'计算的列名。

    返回:
    - pd.DataFrame: 计算未来收益率后的数据。
    """
    label = df[[col_name]].unstack(level='Symbol').pct_change(next).shift(-next-1).stack(level='Symbol')
    df[col_name] = label
    df = df[~df.index.get_level_values(0).isin(df.index.get_level_values(0).unique()[-next-1:])]
    return df
def load_all_data(path: str, start_year: int, end_year: int):
    """
    加载所有数据函数，并返回总数据集。

    参数:
    - path: 数据存放路径。
    - year: 加载数据的年份（起始年份和结尾年份）。
    """
    data_list = []
    for y in range(start_year, end_year + 1):
        file_path = os.path.join(path, f"factor_{y}.parquet")
        if os.path.exists(file_path):
            parquet_file = pq.ParquetFile(file_path)
            data = parquet_file.read().to_pandas()
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index(['Date','Symbol'],inplace=True)
            data.sort_index(level='Date', inplace=True)
            data_list.append(data)
        else:
            print(f"Warning: File {file_path} does not exist.")
    combined_data = pd.concat(data_list)
    return combined_data
def load_data_window(path: str, train_start_date: pd.Timestamp, train_end_date: pd.Timestamp, pred_start_date: pd.Timestamp, pred_end_date: pd.Timestamp):
    """
    加载训练集和测试集的数据函数，并返回两个数据集。

    参数:
    - path: 数据存放路径。
    - years: 加载数据的年份（训练集和测试集起始年份和结尾年份）。
    """
    data_frames = []
    for y in range(train_start_date.year, pred_end_date.year + 1):
        file_path = os.path.join(path, f"factor_{y}.parquet")
        if os.path.exists(file_path):
            parquet_file = pq.ParquetFile(file_path)
            data = parquet_file.read().to_pandas()
            data['Date'] = pd.to_datetime(data['Date'])
            data_frames.append(data)
        else:
            print(f"Warning: File {file_path} does not exist.")
    data_window = pd.concat(data_frames).sort_values(by=['Date', 'Symbol']).reset_index(drop=True)
    data_window.set_index(['Date','Symbol'],inplace=True)
    data_window.sort_index(level='Date', inplace=True)
    data_window = resample_to_week(data_window)
    data_window = data_window.reorder_levels(['Date', 'Symbol'])
    data_window = data_window.sort_index()
    data_window = cal_label(data_window, next = 1)
    train_window = data_window[(data_window.index.get_level_values('Date') >= train_start_date) & (data_window.index.get_level_values('Date') <= train_end_date)]
    pred_window = data_window[(data_window.index.get_level_values('Date') >= pred_start_date) & (data_window.index.get_level_values('Date') <= pred_end_date)]   
    train_window, pred_window = standardize_by_date(train_window, pred_window, data_window.columns.drop('AdjVWAP'))
    return train_window, pred_window
def train_val_split(train_window, half_year_splits=0.8):
    """
    训练集和验证集划分函数。

    参数:
    - data: 构建模型的数据，在其中划分训练集和验证集。
    - half_year_splits: 划分为五份，训练集占比为前四份。

    返回:
    - train_data, val_data: 训练集和验证集的数据。
    """
    unique_dates = train_window.index.get_level_values('Date').unique()
    split_index = int(len(unique_dates) * half_year_splits)
    train_dates = unique_dates[:split_index]
    val_dates = unique_dates[split_index:]
    train_set = train_window[train_window.index.get_level_values('Date').isin(train_dates)]
    val_set = train_window[train_window.index.get_level_values('Date').isin(val_dates)]
    return train_set, val_set
def preprocess_data(train_set: pd.DataFrame, val_set: pd.DataFrame, pred_window: pd.DataFrame, time_steps: int):
    """
    数据预处理函数，将数据分为特征和标签。
    
    参数:
    - train_set和val_set: 经过标准化和标签计算后的训练集和验证集数据。
    - pred_window: 经过标准化和标签计算后的预测集数据。

    返回:
    （总共的或者单股票的样本数需要groupby股票后遍历日期）
    - X_train: 训练特征数据：123个因子，包含时间步长为time_steps的样本。
    - y_train: 训练标签数据：next=1的收益率标签。
    - X_val: 验证特征数据：123个因子，包含时间步长为time_steps的样本
    - y_val: 验证标签数据：next=1的收益率标签。
    - X_test: 预测特征数据：123个因子，包含时间步长为time_steps的样本。
    - y_test: 预测标签数据：next=1的收益率标签。
    """
    ##### 没有单股票划分的部分代码
    # X = torch.tensor(train_window.iloc[:, 1:].values, dtype=torch.float32)
    # X = X.unfold(0, time_steps, 1).permute(0, 2, 1)
    # y = torch.tensor(train_window.iloc[time_steps-1:, 0].values, dtype=torch.float32).view(-1, 1)
    # X_test = torch.tensor(pred_window.iloc[:, 1:].values, dtype=torch.float32)
    # X_test = X_test.unfold(0, time_steps, 1).permute(0, 2, 1)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_target = [], []
    y_index = []
    X_insample, y_insample = [], []
    y_insample_index = []
    
    grouped_train = train_set.groupby(level='Symbol')
    stock_codes_train = list(grouped_train.groups.keys())
    # random.shuffle(stock_codes_train)

    for stock_code in stock_codes_train:
        group = grouped_train.get_group(stock_code).sort_index()
        values = group.values
        for i in range(time_steps, len(values)):
            X_train.append(values[i-time_steps:i, 1:])
            X_insample.append(values[i-time_steps:i, 1:])
            y_train.append(values[i-1, 0])
            y_insample.append(values[i-1, 0])
            current_index = group.index[i-1]
            y_insample_index.append(current_index)

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    logging.info("X_train shape: %s", X_train.shape)
    logging.info("y_train shape: %s", y_train.shape)

    grouped_val = val_set.groupby(level='Symbol')
    stock_codes_val = list(grouped_val.groups.keys())
    # random.shuffle(stock_codes_val)

    for stock_code in stock_codes_val:
        group = grouped_val.get_group(stock_code).sort_index()
        values = group.values
        for i in range(time_steps, len(values)):
            X_val.append(values[i-time_steps:i, 1:])
            X_insample.append(values[i-time_steps:i, 1:])
            y_val.append(values[i-1, 0])
            y_insample.append(values[i-1, 0])
            current_index = group.index[i-1]
            y_insample_index.append(current_index)
    X_val = np.array(X_val)
    y_val = np.array(y_val).reshape(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_insample = np.array(X_insample)
    y_insample = np.array(y_insample).reshape(-1, 1)
    X_insample = torch.tensor(X_insample, dtype=torch.float32)
    y_insample = torch.tensor(y_insample, dtype=torch.float32).view(-1, 1)
    y_insample_df = pd.DataFrame(y_insample, index=pd.MultiIndex.from_tuples(y_insample_index, names=['Date', 'Symbol']), columns=['y_target'])
    logging.info("X_val shape: %s", X_val.shape)
    logging.info("y_val shape: %s", y_val.shape)
    logging.info("X_insample shape: %s", X_insample.shape)
    logging.info("y_insample shape: %s", y_insample.shape)
    logging.info("y_insample_index shape: %s", len(y_insample_index))

    grouped_test = pred_window.groupby(level='Symbol')
    stock_codes_test = list(grouped_test.groups.keys())
    # random.shuffle(stock_codes_test)

    for stock_code in stock_codes_test:
        group = grouped_test.get_group(stock_code).sort_index()
        values = group.values
        for i in range(time_steps, len(values)):
            X_test.append(values[i-time_steps:i, 1:])
            y_target.append(values[i-1, 0])
            current_index = group.index[i-1]
            y_index.append(current_index)
    X_test = np.array(X_test)
    y_target = np.array(y_target).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_target = torch.tensor(y_target, dtype=torch.float32).view(-1, 1)
    y_target_df = pd.DataFrame(y_target, index=pd.MultiIndex.from_tuples(y_index, names=['Date', 'Symbol']), columns=['y_target'])
    logging.info("X_test shape: %s", X_test.shape)
    logging.info("y_target shape: %s", y_target.shape)
    logging.info("y_index shape: %s", len(y_index))
    return X_train, y_train, X_val, y_val, X_insample, y_insample, y_insample_df, X_test, y_target, y_target_df
def preprocess_data_randomstock(train_set: pd.DataFrame, val_set: pd.DataFrame, pred_window: pd.DataFrame, time_steps: int):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_target = [], []
    y_index = []
    X_insample, y_insample = [], []
    y_insample_index = []
    
    grouped_train = train_set.groupby(level='Symbol')
    stock_codes_train = list(grouped_train.groups.keys())
    random.shuffle(stock_codes_train)

    for stock_code in stock_codes_train:
        group = grouped_train.get_group(stock_code).sort_index()
        values = group.values
        for i in range(time_steps, len(values)):
            X_train.append(values[i-time_steps:i, 1:])
            X_insample.append(values[i-time_steps:i, 1:])
            y_train.append(values[i-1, 0])
            y_insample.append(values[i-1, 0])
            current_index = group.index[i-1]
            y_insample_index.append(current_index)

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    logging.info("X_train shape: %s", X_train.shape)
    logging.info("y_train shape: %s", y_train.shape)

    grouped_val = val_set.groupby(level='Symbol')
    stock_codes_val = list(grouped_val.groups.keys())
    random.shuffle(stock_codes_val)

    for stock_code in stock_codes_val:
        group = grouped_val.get_group(stock_code).sort_index()
        values = group.values
        for i in range(time_steps, len(values)):
            X_val.append(values[i-time_steps:i, 1:])
            X_insample.append(values[i-time_steps:i, 1:])
            y_val.append(values[i-1, 0])
            y_insample.append(values[i-1, 0])
            current_index = group.index[i-1]
            y_insample_index.append(current_index)
    X_val = np.array(X_val)
    y_val = np.array(y_val).reshape(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_insample = np.array(X_insample)
    y_insample = np.array(y_insample).reshape(-1, 1)
    X_insample = torch.tensor(X_insample, dtype=torch.float32)
    y_insample = torch.tensor(y_insample, dtype=torch.float32).view(-1, 1)
    y_insample_df = pd.DataFrame(y_insample, index=pd.MultiIndex.from_tuples(y_insample_index, names=['Date', 'Symbol']), columns=['y_target'])
    logging.info("X_val shape: %s", X_val.shape)
    logging.info("y_val shape: %s", y_val.shape)
    logging.info("X_insample shape: %s", X_insample.shape)
    logging.info("y_insample shape: %s", y_insample.shape)
    logging.info("y_insample_index shape: %s", len(y_insample_index))

    grouped_test = pred_window.groupby(level='Symbol')
    stock_codes_test = list(grouped_test.groups.keys())
    random.shuffle(stock_codes_test)

    for stock_code in stock_codes_test:
        group = grouped_test.get_group(stock_code).sort_index()
        values = group.values
        for i in range(time_steps, len(values)):
            X_test.append(values[i-time_steps:i, 1:])
            y_target.append(values[i-1, 0])
            current_index = group.index[i-1]
            y_index.append(current_index)
    X_test = np.array(X_test)
    y_target = np.array(y_target).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_target = torch.tensor(y_target, dtype=torch.float32).view(-1, 1)
    y_target_df = pd.DataFrame(y_target, index=pd.MultiIndex.from_tuples(y_index, names=['Date', 'Symbol']), columns=['y_target'])
    logging.info("X_test shape: %s", X_test.shape)
    logging.info("y_target shape: %s", y_target.shape)
    logging.info("y_index shape: %s", len(y_index))
    return X_train, y_train, X_val, y_val, X_insample, y_insample, y_insample_df, X_test, y_target, y_target_df