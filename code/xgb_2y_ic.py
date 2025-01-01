import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import numpy as np
import optuna
import operator
#from xgboost import XGBRegressor
import sys
import os
import time
from pandas import read_parquet
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

import pickle
import matplotlib.pyplot as plt
data_dir = "/home/cheam/IP_alpha/xgb_ic_2y/"
f=open(os.path.join(data_dir, "train.txt"),"w+")
data = pd.read_parquet('/home/cheam/IP_alpha/data/factor_all_std.parquet')
results_all_tunning = pd.DataFrame(columns=['test_start_date','eta','max_depth','colsample_bytree','colsample_bylevel','subsample','out_ic','in_ic','out_mse','in_mse'])
date_range = pd.date_range(start='2010-01-01', end='2025-02-01', freq='6M')# end应该是24-10-01
year = [
    
    (f"{date.year}-{date.month:02d}", 1 if date.month == 1 or date.month ==7 else 0) for date in date_range
]
pd.set_option('display.max_rows', None)  # 设置最大行数为 None，表示不限制
pd.set_option('display.max_columns', None)  # 设置最大列数为 None，表示不限制
class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()

def custom_ic_train(y_pred, dtrain):
    """
    Compute the gradient and second derivative of the Pearson correlation coefficient for training.

    This function calculates the first and second derivatives of the Pearson
    correlation coefficient with respect to the predicted values. These derivatives
    are useful for gradient-based optimization in machine learning models.

    Parameters:
    y_true (array-like): The actual values.
    y_pred (array-like): The predicted values by the model.

    Returns:
    tuple: A tuple containing:
        - float: The negative first derivative of the Pearson correlation coefficient.
        - float: The negative second derivative of the Pearson correlation coefficient.
    """
    epsilon = 1e-7

    y_true = dtrain.get_label()

    y_pred_mean = np.mean(y_pred)
    y_true_mean = np.mean(y_true)

    a = np.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) + epsilon
    b = np.sum((y_pred - y_pred_mean) ** 2) + epsilon
    c = np.sum((y_true - y_true_mean) ** 2) + epsilon
    r = a / (np.sqrt(b) * np.sqrt(c))

    drdx = r * ((y_true - y_true_mean) / a - (y_pred - y_pred_mean) / b)

    dadx = y_true - y_true_mean
    dbdx = 2.0 * (y_pred - y_pred_mean)

    part1 = drdx**2 / r
    part2 = (
        -(y_true - y_true_mean) / a**2 * dadx
        - (b * (1.0 - 1.0 / len(y_true)) - (y_pred - y_pred_mean) * dbdx) / b**2
    ) * r
    d2rdx2 = part1 + part2
    return -1.0 * drdx, -1.0 * d2rdx2
def XGBoost_metric(index):

    def custom_ic_valid_XGB(y_pred, dtrain):
        #epsilon = 1e-7
        y_true=pd.Series(dtrain.get_label()).to_frame(name='y_true')
        y_true.index=index
        y_pred=pd.Series(y_pred).to_frame(name='y_pred')
        y_pred.index=index

        grouped = pd.concat([y_pred, y_true], axis=1)
        #r = grouped.groupby('Date').corr().iloc[0::2, 1].mean()
        r = grouped.groupby(level = 0).corr(method = 'spearman').groupby(level = 1).mean().iloc[0,1]
        return "custom_ic_valid", -1.0 * r
    
    return custom_ic_valid_XGB
# optuna 优化函数
def objective(trial): #先定义很多超参数的范围，然后利用这些参数训练模型，得到test error最小的，假设我们是不知道test data的吧，只能输出cv的test error
    n_splits = 10
    unique_data = train_data.index.get_level_values('Date').unique()
    split_size = len(unique_data) // n_splits
    eta=trial.suggest_float("eta", 0, 0.2)
    max_depth=trial.suggest_int("max_depth", 1, 6)
    # lambda_=trial.suggest_float("lambda", 0, 800)
    # alpha = trial.suggest_float("alpha", 400, 800)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0, 1)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0, 1)
    # gamma = trial.suggest_float("gamma", 0, 2)
    subsample = trial.suggest_float("subsample", 0, 1)
    params = {'booster': 'gbtree',
              'seed': 777,
              'eta' :eta ,
              "max_depth": max_depth,
            # 'lambda': lambda_,
            #   'gamma': gamma,
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
                'colsample_bylevel': colsample_bylevel,
                'colsample_bynode': 0,
                # 'alpha': alpha,
               'device':"cuda",
               'disable_default_eval_metric': 1,
              }

    # try:

    tunning_outsample_score_ic=[]
    tunning_outsample_score_mse = []
    tunning_insample_score_ic = []
    tunning_insample_score_mse = []

    for i in range(n_splits-2):
        test_indices = range((i+2) * split_size, (i + 3) * split_size)  # 测试集
        val_indices = range((i + 1) * split_size, (i + 2) * split_size) 
        train_indices = range(0, (i+1) * split_size)  # 训练集
        #print('train:', train_indices[0], train_indices[-1], 'val:', val_indices[0], val_indices[-1], 'test:', test_indices[0], test_indices[-1])
        train_date = unique_data[train_indices]
        val_date = unique_data[val_indices]
        test_date = unique_data[test_indices]
        train_odata = train_data.loc[train_data.index.get_level_values('Date').isin(train_date)]
        val_odata = train_data.loc[train_data.index.get_level_values('Date').isin(val_date)]
        test_odata = train_data.loc[train_data.index.get_level_values('Date').isin(test_date)]
        # train_odata = train_data[train_data.index.get_level_values('Date').isin(train_dates)]
        # valid_odata = train_data[train_data.index.get_level_values('Date').isin(valid_dates)]
        # test_odata = train_data[train_data.index.get_level_values('Date').isin(test_dates)]
        Dtrain_op = xgb.DMatrix(train_odata.drop('AdjVWAP',axis=1),label=train_odata['AdjVWAP'])
        Dvalid_op = xgb.DMatrix(val_odata.drop('AdjVWAP',axis=1),label=val_odata['AdjVWAP'])
        Dtest_op = xgb.DMatrix(test_odata.drop('AdjVWAP',axis=1),label=test_odata['AdjVWAP'])
        watchlist = [(Dvalid_op, 'eval')]
        #num_round = 200

        model=xgb.train(params=params, dtrain=Dtrain_op, num_boost_round=1000, evals=watchlist,
                        obj=custom_ic_train, custom_metric=XGBoost_metric(val_odata.index),
                        early_stopping_rounds=10,verbose_eval=False
                        )

        # out of sample
        tunning_y_test_outsample = model.predict(Dtest_op)
        tunning_y_test_outsample = pd.DataFrame(tunning_y_test_outsample, index = test_odata.index, columns=['pred'])
        tunning_y_test_outsample['true'] = test_odata['AdjVWAP']
        tunning_outsample_ic = tunning_y_test_outsample.groupby(level = 0).corr(method = 'spearman').groupby(level = 1).mean().iloc[0,1]
        #print("train_index",train_odata.index.get_level_values('Date').unique(),"test_index",test_odata.index.get_level_values('Date'), "ic", model_ic)
        tunning_outsample_score_ic.append(tunning_outsample_ic)
        tunning_outsample_mse = root_mean_squared_error(tunning_y_test_outsample['true'], tunning_y_test_outsample['pred'])
        tunning_outsample_score_mse.append(tunning_outsample_mse)

        # in sample

        tunning_y_test_insample = model.predict(Dtrain_op)
        tunning_y_test_insample = pd.DataFrame(tunning_y_test_insample, index = train_odata.index, columns=['in_pred'])
        tunning_y_test_insample['true'] = train_odata['AdjVWAP']
        tunning_insample_ic = tunning_y_test_insample.groupby(level = 0).corr(method = 'spearman').groupby(level = 1).mean().iloc[0,1]
        #print("train_index",train_odata.index.get_level_values('Date').unique(),"test_index",test_odata.index.get_level_values('Date'), "ic", model_ic)
        tunning_insample_score_ic.append(tunning_insample_ic)
        tunning_imsample_mse = root_mean_squared_error(tunning_y_test_insample['true'], tunning_y_test_insample['in_pred'])
        tunning_insample_score_mse.append(tunning_imsample_mse)


    tunning_mean_ic_outsample=pd.Series(tunning_outsample_score_ic).mean()
    tunning_mean_mse_outsample = pd.Series(tunning_outsample_score_mse).mean()
    tunning_mean_ic_insample = pd.Series(tunning_insample_score_ic).mean()
    tunning_mean_mse_insample = pd.Series(tunning_insample_score_mse).mean()
    results_all_tunning.loc[len(results_all_tunning)] = [test_start_date,eta,max_depth,colsample_bytree,
    colsample_bylevel,subsample,tunning_mean_ic_outsample,tunning_mean_ic_insample,tunning_mean_mse_outsample,tunning_mean_mse_insample]
    print("results_all_tunning",results_all_tunning,file=f,flush=True)
    return tunning_mean_ic_outsample
    # except:
    #     return -10000
    
additional_params = {
            'booster': 'gbtree',
              'seed': 777,
              'device':"cuda", # 使用 GPU 进行预测
                'colsample_bynode': 0,
                'disable_default_eval_metric': 1,

}

direction = "maximize"

results=pd.DataFrame()
results_insample=pd.DataFrame()
score_ic_out_of_sample=[]
score_ic_in_sample=[]
score_mse_out_of_sample=[]
score_mse_in_sample=[]
result_all_train = pd.DataFrame(columns=['test_start_date','eta','max_depth','colsample_bytree','col_sample_bynode',
                                         'colsample_bylevel','subsample','out_of_sample_ic','in_sample_ic','out_of_sample_mse','in_sample_mse'])
for i in range(len(year)-5): 
    train_start_date = year[i][0]
    #valid_start_date = year[i+3][0]
    test_start_date = year[i+4][0]
    valid_start_date = pd.to_datetime(test_start_date) - pd.DateOffset(months=3)
    test_end_date = year[i+5][0]
    print('train_start_date',train_start_date, 'valid_start_date',valid_start_date, 'test_start_date',test_start_date,file=f,flush=True)
    train_data = data.loc[(data.index.get_level_values(0) >= train_start_date) & (data.index.get_level_values(0) < valid_start_date)]
    valid_data = data.loc[(data.index.get_level_values(0) >= valid_start_date) & (data.index.get_level_values(0) < test_start_date)]
    test_data = data.loc[(data.index.get_level_values(0) >= test_start_date) & (data.index.get_level_values(0) < test_end_date)]
    if year[i][1] == 1:# 调参
        print('调参',file=f,flush=True)
        print('test_start_date',test_start_date,file=f,flush=True)
        absolute_path = data_dir + "/xgb_ic_rolling.db"
        study = optuna.create_study(
        storage=f"sqlite:///{absolute_path}",
        study_name="xgb_ic_" + train_start_date+'_' + test_start_date,
        direction=direction,
        load_if_exists=True,
        )

        early_stopping = EarlyStoppingCallback(10, direction=direction)
        # 开始优化
        study.optimize(objective, n_jobs=1, callbacks=[early_stopping], timeout=3600)
        best_params = study.best_params
        best_params.update(additional_params)
        print(best_params,file=f,flush=True)
    Dtrain = xgb.DMatrix(train_data.drop('AdjVWAP',axis=1), label=train_data['AdjVWAP'])
    Dvalid = xgb.DMatrix(valid_data.drop('AdjVWAP',axis=1), label=valid_data['AdjVWAP'])
    Dtest = xgb.DMatrix(test_data.drop('AdjVWAP',axis=1), label=test_data['AdjVWAP'])
    watchlist = [(Dvalid, 'eval')]
    model=xgb.train(params=best_params, dtrain=Dtrain, num_boost_round=1000, evals =  watchlist,early_stopping_rounds=10,
                    obj=custom_ic_train, custom_metric=XGBoost_metric(valid_data.index)
                    )
    variable_part = "已调参" if  year[i][1] == 1 else ""
    model_name = data_dir + 'xgb_ic_' + train_start_date+'_'+test_start_date+ variable_part + '.json'
    model.save_model(model_name)

    ## out of sample 
    y_pred = pd.DataFrame(model.predict(Dtest))
    y_pred.index=test_data.index
    y_pred.columns=[f"pred_"+test_start_date]
    cal_y_pred = y_pred.copy()
    cal_y_pred['true'] = test_data['AdjVWAP']
    train_ic_outsample = cal_y_pred.groupby(level = 0).corr(method = 'spearman').groupby(level = 1).mean().iloc[0,1]
    train_mse_outsample = root_mean_squared_error(cal_y_pred['true'], cal_y_pred['pred_'+test_start_date])
    results = pd.concat([y_pred, results], join= 'outer', axis=1)


    ## in sample
    y_pred_in = pd.DataFrame(model.predict(Dtrain))
    y_pred_in.index=train_data.index
    y_pred_in.columns=[f"pred_in_"+test_start_date]
    cal_y_pred_in = y_pred_in.copy()
    cal_y_pred_in['true'] = train_data['AdjVWAP']
    train_ic_insample = cal_y_pred_in.groupby(level = 0).corr(method = 'spearman').groupby(level = 1).mean().iloc[0,1]
    train_mse_insample = root_mean_squared_error(cal_y_pred_in['true'], cal_y_pred_in['pred_in_'+test_start_date])
    results_insample = pd.concat([y_pred_in, results_insample], join= 'outer', axis=1)


    # 保存结果
    result_all_train.loc[len(result_all_train)] = [test_start_date,best_params['eta'],best_params['max_depth'],best_params['colsample_bytree'],best_params['colsample_bynode'],
    best_params['colsample_bylevel'],best_params['subsample'],train_ic_outsample,train_ic_insample,train_mse_outsample,train_mse_insample]
    print("result_all_train",result_all_train,file=f,flush=True)
results['mean']=results.mean(axis=1)
results = results.sort_index(level='Date')
results.to_parquet(data_dir + 'results_xgb.parquet')
results_insample['mean']=results_insample.mean(axis=1)
results_insample = results_insample.sort_index(level='Date')
results_insample.to_parquet(data_dir + 'results_insample_xgb.parquet')
result_all_train.to_parquet(data_dir + 'result_all_train.parquet')
results_all_tunning.to_parquet(data_dir +'results_all_tunning.parquet')