import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import numpy as np
import optuna
import operator
from sklearn.model_selection import train_test_split
#from xgboost import XGBRegressor

from pandas import read_parquet
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

import pickle
import matplotlib.pyplot as plt
data = pd.read_parquet('/home/cheam/IP_alpha/data/factor_all_std.parquet')
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def check_paras(data, min_value, max_value, num_trials, kf_folds, param_name, if_int = False):
    kf = KFold(n_splits=kf_folds)
    values = np.linspace(min_value, max_value, num=num_trials)
    unique_dates = data.index.get_level_values('Date').unique()
    if if_int:
        values = values.astype(int)
    mse = pd.DataFrame(columns=[param_name, 'rankic','mse'])

    for i in range(num_trials):
        # 定义参数
        params = {param_name: values[i], 
                  #'disable_default_eval_metric': 1,
                  }
        print(param_name, f": {values[i]}")

        score_ic = []
        score_mse = []

        for train_index, test_index in kf.split(unique_dates):
            last_tenth_length = len(train_index) // 9
            valid_index = train_index[-last_tenth_length:]
            train_index = np.setdiff1d(train_index, valid_index)
            train_dates = unique_dates[train_index]
            valid_dates = unique_dates[valid_index]
            test_dates = unique_dates[test_index]
            train_odata = data[data.index.get_level_values('Date').isin(train_dates)]
            valid_odata = data[data.index.get_level_values('Date').isin(valid_dates)]
            test_odata = data[data.index.get_level_values('Date').isin(test_dates)]
            Dtrain_op = xgb.DMatrix(train_odata.drop('AdjVWAP',axis=1),label=train_odata['AdjVWAP'])
            Dvalid_op = xgb.DMatrix(valid_odata.drop('AdjVWAP',axis=1),label=valid_odata['AdjVWAP'])
            Dtest_op = xgb.DMatrix(test_odata.drop('AdjVWAP',axis=1),label=test_odata['AdjVWAP'])
            watchlist = [(Dtrain_op,'train'),(Dvalid_op, 'eval')]

            model = xgb.train(params=params, dtrain=Dtrain_op, num_boost_round=1000, evals=watchlist,
                              early_stopping_rounds=10, verbose_eval=False,
                              #obj=custom_ic_train, custom_metric=XGBoost_metric(Dvalid_op.index),
                              )
            #score.append(model.best_score)
            y_test = model.predict(Dtest_op)
            y_test = pd.DataFrame(y_test, index = test_odata.index, columns=['pred'])
            y_test['true'] = test_odata['AdjVWAP']
            model_ic = y_test.groupby(level = 0).corr(method = 'spearman').groupby(level = 1).mean().iloc[0,1]
            #print("train_index",train_odata.index.get_level_values('Date').unique(),"test_index",test_odata.index.get_level_values('Date'), "ic", model_ic)
            score_ic.append(model_ic)
            model_mse = mean_squared_error(y_test['true'], y_test['pred'])
            score_mse.append(model_mse)

        mean_ic = np.mean(score_ic)
        mean_mse = np.mean(score_mse)
        print("ic",score_ic, "mean_ic",mean_ic)
        print("mse",score_mse, "mean_mse",mean_mse)
        mse.loc[i] = [values[i], mean_ic, mean_mse]  # 只存储 gamma 和 mse
    fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制 IC 的折线图
    ax1.plot(mse[param_name], mse['rankic'], label='Rank IC', marker='o', color='b')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('IC', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 创建第二条 y 轴
    ax2 = ax1.twinx()

    # 绘制 MSE 的折线图
    ax2.plot(mse[param_name], mse['mse'], label='MSE', marker='x', color='r')
    ax2.set_ylabel('MSE', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 添加标题
    plt.title(param_name + ' vs Rank IC and MSE')

    # 添加图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 显示网格
    ax1.grid()

    # 显示图形
    plt.show()
    plt.savefig('/home/cheam/ip_chou/model_xgb/'+ param_name + '_' + str(min_value) + '_' + str(max_value) + '.png')

check_paras(data, 0, 100, num_trials=10, kf_folds=10, param_name='gamma')
check_paras(data, 0.01, 1, 10, kf_folds=10, param_name='eta')
#check_paras(data, 0.01, 0.2, 10, kf_folds=10, param_name='eta')
# 2 < max_depth< 6
check_paras(data, 1, 10, 10, kf_folds=10, param_name='max_depth', if_int=True)
# subsample == 1
check_paras(data, 0, 1, 10-1, kf_folds=10, param_name='subsample' ,if_int=False)
# colsample_bytree == 1
check_paras(data, 0, 1, 10-1, kf_folds=10, param_name='colsample_bytree' ,if_int=False)
# colsample_bylevel == 1
check_paras(data, 0, 1, 10-1, kf_folds=10, param_name='colsample_bylevel' ,if_int=False)
# colsample_bynode == 1
check_paras(data, 0, 1, 10-1, 5, param_name='colsample_bynode' ,if_int=False)
# 150 < lambda < 275

check_paras(data, 0, 1000, 10, kf_folds=10, param_name='lambda' ,if_int=False)
# alpha == 0 
check_paras(data, 0, 1000, 10, kf_folds=10, param_name='alpha' ,if_int=False)