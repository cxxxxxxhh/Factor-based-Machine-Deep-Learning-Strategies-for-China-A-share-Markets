# 数据分析
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from tqdm import *
import os
import datetime
import file_config as fc
import matplotlib.pyplot as plt

# 关闭通知
import warnings
warnings.filterwarnings("ignore")

def get_turnover(buy_list, change_n):
    turnover_dict = {}

    for i, day in enumerate(buy_list.index,1):
        if day == buy_list.index[0]:
            temp_list = buy_list.loc[day]
        else:
            if i % change_n == 0:
                turnover = len(set(buy_list.loc[day].dropna().index).difference(set(temp_list.dropna().index)))/len(temp_list.dropna().index)
                turnover_dict[day] = turnover
                temp_list = buy_list.loc[day]
            
    return pd.DataFrame.from_dict(turnover_dict,orient='index',columns=['turnover'])

def change_stock_name(stock_name:str):
    if stock_name.endswith('.XSHE'):
        return stock_name.replace('.XSHE', '.SZ')
    elif stock_name.endswith('.XSHG'):
        return stock_name.replace('.XSHG', '.SH')
    else:
        return stock_name


def get_factor(year_list:list):
    for year in year_list:
        if not os.path.exists(os.path.join(fc.FACTOR_DIR, f'factor_{year}.parquet')):
            raise FileNotFoundError(f'factor_{year}.parquet not found')
    return pd.concat([pd.read_parquet(os.path.join(fc.FACTOR_DIR, f'factor_{year}.parquet')) for year in year_list])


def get_price(stock_list, start_date, end_date):
    year_list = pd.date_range(start_date, end_date).year.unique().tolist()

    # load factors
    factor = get_factor(year_list)
    factor['Date'] = pd.to_datetime(factor['Date'], format='%Y%m%d')

    # keep desired stocks and dates
    if stock_list == []:
        stock_list = factor.Symbol.unique().tolist()

    price = factor.loc[(factor.Symbol.isin(stock_list)) & (factor.Date >= start_date) & (factor.Date <= end_date), ['Symbol', 'Date', 'AdjVWAP']]
    return price.pivot(index='Date', columns='Symbol', values='AdjVWAP')

def get_index_price(index:str, start_date:str, end_date:str, field:str='close'):
    index_list = ['000300.SH', '000905.SH', '000852.SH', '000985.SH']
    if index not in index_list:
        print(f'pleace select from {index_list}')
    
    index = pd.read_csv(f'{index}.csv')[['ts_code', 'trade_date', field]]
    index['trade_date'] = pd.to_datetime(index['trade_date'], format='%Y%m%d')
    index = index.loc[(index.trade_date >= start_date) & (index.trade_date <= end_date), ['ts_code','trade_date', 'close']]
    return index.pivot(index='trade_date', columns='ts_code', values=field)


def get_previous_trading_date(date, days_ago:int):
    trade_cal = pd.read_csv(os.path.join(fc.ROOT_DIR, 'trade_cal.csv'))
    trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
    trade_cal = trade_cal.loc[trade_cal.is_open == 1, 'cal_date']
    return trade_cal[trade_cal <= date].iloc[days_ago]

# 获取买入对队列
def get_buy_list(df,top_type = 'rank',rank_n = 100,quantile_q = 0.8):
    """
    :param df: 因子值 -> dataframe/unstack
    :param top_tpye: 选择买入队列方式，从['rank','quantile']选择一种方式 -> str
    :param rank_n: 值最大的前n只的股票 -> int
    :param quantile_q: 值最大的前n分位数的股票 -> float
    :return df: 买入队列 -> dataframe/unstack
    """
    if top_type == 'rank':
        df = df.rank(axis  = 1,ascending=False, method='first') <= rank_n
    elif top_type == 'quantile':
        df = df.sub(df.quantile(quantile_q,axis = 1),axis = 0) > 0
    else:
        print("select one from ['rank','quantile']")

    df = df.astype(int)
    df = df.replace(0,np.nan).dropna(how = 'all',axis = 1)
    
    return df


def get_bar(df):
    """
    :param df: 买入队列 -> dataframe/unstack
    :param benchmark: 基准指数 -> str
    :return ret: 基准的逐日收益 -> dataframe
    """
    start_date = get_previous_trading_date(df.index.min(),1)
    end_date = df.index.max()
    stock_list = df.columns.tolist()
    price_open = get_price(stock_list,start_date,end_date)
    
    return price_open


def get_benchmark(df,benchmark):
    """
    :param df: 买入队列 -> dataframe/unstack
    :param benchmark: 基准指数 -> str
    :return ret: 基准的逐日收益 -> dataframe
    """
    start_date = get_previous_trading_date(df.index.min(),1)
    end_date = df.index.max()
    price_open = get_index_price(benchmark,start_date,end_date) 
    return price_open


#最小化方差
def min_variance(weights,temp_cov):
    p_var = np.dot(weights.T,np.dot(temp_cov,weights))
    return p_var


# 主动最小化方差
def min_active_variance(weight,temp_ret,temp_benchamrk_ret):
    active_returns = temp_ret.sub(temp_benchamrk_ret,axis = 0)
    r_act_cov = active_returns.cov()
    p_act_var = np.dot(weight.T,np.dot(r_act_cov,weight))
    p_act_std = np.sqrt(p_act_var)

    return p_act_std

# 风险平价
def RiskParity(weights,temp_cov):
    sigma = np.sqrt(np.dot(weights, np.dot(temp_cov, weights)))  
    MRC = np.dot(temp_cov,weights)/sigma
    TRC = weights * MRC
    delta_TRC = [sum((i - TRC)**2) for i in TRC]
    return sum(delta_TRC)


# 效用理论
def utility_theory(weights,temp_ret):
    p_r = (1 + np.dot(weights,temp_ret.mean())) ** 252 - 1
    temp_cov = temp_ret.cov()
    lam = 10 
    p_sigma = np.sqrt(np.dot(weights.T,np.dot(temp_cov,weights)) * 252) 
    p_utility = p_r - (lam / 2  * p_sigma)
    return -1 * p_utility


# 最大夏普 （均值方差）
def max_sharpe(weights,temp_ret):
    r_f = 0.03 
    p_r = (1 + np.dot(weights,temp_ret.mean())) ** 252 - 1
    temp_cov = temp_ret.cov()
    p_sigma = np.sqrt(np.dot(weights.T,np.dot(temp_cov,weights)) * 252)
    p_sharpe = (p_r - r_f)/p_sigma
    return -1 * p_sharpe


# 最大化信息比例
def max_ir(weights,temp_ret,temp_benchamrk_ret):
    r_b = 0.03
    p_mean = np.sum(temp_ret*weights,axis = 1)
    p_final_mean = (p_mean + 1).cumprod().iloc[-1] ** (len(p_mean)/252) - 1
    std_tr = (p_mean - temp_benchamrk_ret).std() * np.sqrt(252)
    p_ir = (p_final_mean-r_b)/std_tr

    return -1 * p_ir


# 最小化跟踪误差
def min_tr(weights,temp_ret,temp_benchamrk_ret):
    p_mean = np.sum(temp_ret*weights,axis = 1)
    std_tr = (p_mean - temp_benchamrk_ret).std() * np.sqrt(252)
    return std_tr


def optimizer(df,weight_down_limit = 0,weight_up_limit = 0.03,option = 'equal_vol',freq = 1,benchmark = '000985.SH'):
    option_list = ['equal_weight','equal_vol','min_variance','min_active_variance',
                   'RiskParity','utility_theory','max_sharpe','max_ir','min_tr']
    if option not in option_list:
        print(f'pleace select from {option_list}')
    else:
        start_date = get_previous_trading_date(df.index.min(),252)
        end_date = df.index.max()
        stock_list = df.columns.tolist()
        ret = get_price(stock_list,
                          start_date,
                          end_date).pct_change().dropna(how = 'all')

        benchamrk_ret = get_index_price(benchmark,
                                  start_date,
                                  end_date,
                                  ).pct_change().dropna(how = 'all')

        print('base data gen')
        weights_df = pd.DataFrame(index = df.index,columns=df.columns)
        index_date_list = df.index.tolist()
        for i in tqdm(range(0,len(index_date_list),freq)):
            temp_date = index_date_list[i]
            v = df.loc[temp_date]
            temp_stock_list = v.dropna().index.tolist()
            
            temp_benchamrk_ret = benchamrk_ret.loc[get_previous_trading_date(temp_date,252):temp_date].iloc[:,0]
            temp_ret = ret.loc[get_previous_trading_date(temp_date,252):temp_date,temp_stock_list]
            temp_cov = temp_ret.cov()

            martix_value = np.matrix(temp_cov.values)
            #sicpy优化方法参数设置
            x0 = np.ones(martix_value.shape[0]) / martix_value.shape[0]  
            bnds = tuple((weight_down_limit,weight_up_limit) for _ in x0)                                    # 生成权重
            cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})                                              # 所有权重加总为 1
            options = { 'maxiter':100, 'ftol':1e-20}

            #print(temp_ret.shape[1])
            #平均分配权重
            if option == 'equal_weight':
                weights_temp = np.array([1.0/temp_ret.shape[1]]*temp_ret.shape[1])
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp

            elif option == 'equal_vol':
                #等波动率
                wts = 1/temp_ret.std()
                weights_temp = (wts/wts.sum())
                weights_df.loc[temp_date] = weights_temp

            elif option == 'min_variance':
                #方差最小
                weights_temp = minimize(min_variance,x0,args = (temp_cov),
                                    bounds=bnds,constraints=cons,method='SLSQP',options=options)['x']
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp

            elif option == 'min_active_variance':
                #主动方差最小
                weights_temp = minimize(min_active_variance,x0,args = (temp_ret,temp_benchamrk_ret),
                                    bounds=bnds,constraints=cons,method='SLSQP',options=options)['x']
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp

            elif option == 'RiskParity':
                #风险平价
                weights_temp = minimize(RiskParity,x0,args = (temp_cov),
                                    bounds=bnds,constraints=cons,method='SLSQP',options=options)['x']
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp
            
            elif option == 'utility_theory':
                # 效用理论
                weights_temp = minimize(utility_theory,x0,args = (temp_ret),
                                    bounds=bnds,constraints=cons,method='SLSQP',options=options)['x']
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp

            elif option == 'max_sharpe':
                #最大夏普
                weights_temp = minimize(max_sharpe,x0,args = (temp_ret),
                                    bounds=bnds,constraints=cons,method='SLSQP',options=options)['x']
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp

            elif option == 'max_ir':
                #最大信息系数
                weights_temp = minimize(max_ir,x0,args = (temp_ret,temp_benchamrk_ret),
                                    bounds=bnds,constraints=cons,method='SLSQP',options=options)['x']
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp
            elif option == 'min_tr':
                # 最小化跟踪误差
                weights_temp = minimize(min_tr,x0,args = (temp_ret,temp_benchamrk_ret),
                                    bounds=bnds,constraints=cons,method='SLSQP',options=options)['x']
                weights_temp = pd.Series(weights_temp,index = temp_stock_list)
                weights_df.loc[temp_date] = weights_temp
        

        weights_df[weights_df < 1/2000] = np.nan
        weights_df = weights_df.ffill(limit = int(freq - 1))

        return weights_df
    
def backtest(df_weight, change_n = 20, cash = 10000 * 1000, tax = 0.0005, other_tax = 0.0001, commission = 0.0002, min_fee = 5, cash_interest_yield = 0.02):

    # 基础参数
    inital_cash = cash                                                                                                            # 起始资金
    stock_holding_num_hist = 0                                                                                                    # 初始化持仓       
    buy_cost = other_tax + commission                                                                                             # 买入交易成本
    sell_cost = tax + other_tax + commission                                                                                      # 卖出交易成本
    cash_interest_daily = (1 + cash_interest_yield) ** (1/252) - 1                                                                # 现金账户利息(日)
    account = pd.DataFrame(index = df_weight.index,columns=['total_account_asset','holding_market_cap','cash_account'])           # 账户信息存储
    price_open = get_bar(df_weight)
    price_open.index = pd.to_datetime(price_open.index)                                                                                               # 获取开盘价格数据
    stock_round_lot = pd.Series(dict([(i,100) for i in df_weight.columns.tolist()]))                         # 标的最小买入数量
    change_day = sorted(set(df_weight.index.tolist()[::change_n] + [df_weight.index[-1]]))                                        # 调仓日期

    # 滚动计算
    for i in tqdm(range(0,len(change_day)-1)):
        start_date = change_day[i]
        end_date = change_day[i+1]

        # 获取给定权重
        df_weight_temp = df_weight.loc[start_date].dropna()
        stock_list_temp = df_weight_temp.index.tolist()
        # 计算个股持股数量 = 向下取整(给定权重 * 可用资金 // 最小买入股数) * 最小买入股数
        stock_holding_num = ((df_weight_temp 
                            * cash 
                            / (price_open.loc[start_date,stock_list_temp] * (1 + sell_cost))        # 预留交易费用
                            // stock_round_lot.loc[stock_list_temp]) 
                            * stock_round_lot.loc[stock_list_temp])

        # 仓位变动
        stock_holding_num_change = stock_holding_num - stock_holding_num_hist
        # 获取期间价格
        price_open_temp = price_open.loc[start_date:end_date,stock_list_temp]
        # 计算交易成本 (可设置万一免五)
        def calc_fee(x,min_fee):
            if x < 0:
                fee_temp = x * sell_cost                                                                                            # 印花税 + 过户费等 + 佣金
            else:
                fee_temp = x * buy_cost                                                                                             # 过户费等 + 佣金
            # 最低交易成本限制
            if fee_temp > min_fee:
                return fee_temp
            else:
                return min_fee

        transaction_costs = ((price_open_temp.loc[start_date] 
                            * stock_holding_num_change)).apply(lambda x: calc_fee(x,min_fee)).sum()
        # 计算期间市值 （交易手续费在现金账户计提）
        holding_market_cap = (price_open_temp * stock_holding_num).sum(axis =1)
        cash_account = cash - transaction_costs - holding_market_cap.loc[start_date]
        cash_account = pd.Series([cash_account*((1 + cash_interest_daily)**(i+1)) for i in range(0,len(holding_market_cap))],
                                index = holding_market_cap.index)
        total_account_asset = holding_market_cap + cash_account
        
        # 将当前持仓存入 
        stock_holding_num_hist = stock_holding_num
        # 下一期期初可用资金
        cash = total_account_asset.loc[end_date]

        account.loc[start_date:end_date,'total_account_asset'] = round(total_account_asset,2)
        account.loc[start_date:end_date,'holding_market_cap'] = round(holding_market_cap,2)
        account.loc[start_date:end_date,'cash_account'] = round(cash_account,2)

    account.loc[pd.to_datetime(get_previous_trading_date(account.index.min(),1))] = [inital_cash,0,inital_cash]
    account = account.sort_index()
    
    return account

def get_performance_analysis(account_result, option, fig_save_path, name = ' ',rf = 0.02,benchmark_index = '000985.SH'):
    
    # 加入基准    
    benchmark = get_benchmark(account_result,benchmark_index)
    benchmark.index = pd.to_datetime(benchmark.index)

    performance = pd.concat([account_result['total_account_asset'].to_frame('strategy'),
                             benchmark],axis = 1)
    performance_net = performance.pct_change().dropna(how = 'all')                                # 清算至当日开盘
    performance_cumnet = (1 + performance_net).cumprod()
    performance_cumnet['alpha'] = performance_cumnet['strategy']/performance_cumnet[benchmark_index]
    performance_cumnet = performance_cumnet.fillna(1)

    # 指标计算
    performance_pct = performance_cumnet.pct_change().dropna()

    # 策略收益
    strategy_name,benchmark_name,alpha_name = performance_cumnet.columns.tolist() 
    Strategy_Final_Return = performance_cumnet[strategy_name].iloc[-1] - 1

    # 策略年化收益
    Strategy_Annualized_Return_EAR = (1 + Strategy_Final_Return) ** (252/len(performance_cumnet)) - 1

    # 基准收益
    Benchmark_Final_Return = performance_cumnet[benchmark_name].iloc[-1] - 1

    # 基准年化收益
    Benchmark_Annualized_Return_EAR = (1 + Benchmark_Final_Return) ** (252/len(performance_cumnet)) - 1

    # alpha 
    ols_result = sm.OLS(performance_pct[strategy_name] * 252 - rf, sm.add_constant(performance_pct[benchmark_name] * 252 - rf)).fit()
    Alpha = ols_result.params[0]

    # beta
    Beta = ols_result.params[1]

    # beta_2 = np.cov(performance_pct[strategy_name],performance_pct[benchmark_name])[0,1]/performance_pct[benchmark_name].var()
    # 波动率
    Strategy_Volatility = performance_pct[strategy_name].std() * np.sqrt(252)

    # 夏普
    Strategy_Sharpe = (Strategy_Annualized_Return_EAR - rf)/Strategy_Volatility

    # 下行波动率
    strategy_ret = performance_pct[strategy_name]
    Strategy_Down_Volatility = strategy_ret[strategy_ret < 0].std() * np.sqrt(252)

    # sortino
    Sortino = (Strategy_Annualized_Return_EAR - rf)/Strategy_Down_Volatility
    
    # 跟踪误差
    Tracking_Error = (performance_pct[strategy_name] - performance_pct[benchmark_name]).std() * np.sqrt(252)

    # 信息比率
    Information_Ratio = (Strategy_Annualized_Return_EAR - Benchmark_Annualized_Return_EAR)/Tracking_Error

    # 最大回测
    i = np.argmax((np.maximum.accumulate(performance_cumnet[strategy_name]) 
                    - performance_cumnet[strategy_name])
                    /np.maximum.accumulate(performance_cumnet[strategy_name]))
    j = np.argmax(performance_cumnet[strategy_name][:i])
    
    Max_Drawdown = (1-performance_cumnet[strategy_name][i]/performance_cumnet[strategy_name][j])

    # 卡玛比率
    Calmar = (Strategy_Annualized_Return_EAR)/Max_Drawdown

    # 超额收益
    Alpha_Final_Return = performance_cumnet[alpha_name].iloc[-1] - 1

    # 超额年化收益
    Alpha_Annualized_Return_EAR = (1 + Alpha_Final_Return) ** (252/len(performance_cumnet)) - 1

    # 超额波动率
    Alpha_Volatility = performance_pct[alpha_name].std() * np.sqrt(252)

    # 超额夏普
    Alpha_Sharpe = (Alpha_Annualized_Return_EAR - rf)/Alpha_Volatility

    # 超额最大回测
    i = np.argmax((np.maximum.accumulate(performance_cumnet[alpha_name]) 
                    - performance_cumnet[alpha_name])
                    /np.maximum.accumulate(performance_cumnet[alpha_name]))
    j = np.argmax(performance_cumnet[alpha_name][:i])
    Alpha_Max_Drawdown = (1-performance_cumnet[alpha_name][i]/performance_cumnet[alpha_name][j])

    # 胜率
    performance_pct['win'] = performance_pct[alpha_name] > 0
    Win_Ratio = performance_pct['win'].value_counts().loc[True] / len(performance_pct)

    # 盈亏比
    profit_lose = performance_pct.groupby('win')[alpha_name].mean()
    Profit_Lose_Ratio = abs(profit_lose[True]/profit_lose[False])

    # 换手率
    turnover = performance_pct[benchmark_name].abs().mean()
    

    result = {
        'Strategy_Final_Return':round(Strategy_Final_Return,4),
        'Strategy_Annualized_Return_EAR': round(Strategy_Annualized_Return_EAR,4),
        'Benchmark_Final_Return':round(Benchmark_Final_Return,4),
        'Benchmark_Annualized_Return_EAR': round(Benchmark_Annualized_Return_EAR,4),
        'Alpha':round(Alpha,4),
        'Beta':round(Beta,4),
        'Volatility':round(Strategy_Volatility,4),
        'Sharpe':round(Strategy_Sharpe,4),
        'Down_Volatility':round(Strategy_Down_Volatility,4),
        'Sortino':round(Sortino,4),
        'Tracking_Error':round(Tracking_Error,4),
        'Information_Ratio':round(Information_Ratio,4),
        'Max_Drawdown':round(Max_Drawdown,4),
        'Calmar': round(Calmar,4),
        'Alpha_Final_Return':round(Alpha_Final_Return,4),
        'Alpha_Annualized_Return_EAR': round(Alpha_Annualized_Return_EAR,4),
        'Alpha_Volatility':round(Alpha_Volatility,4),
        'Alpha_Sharpe':round(Alpha_Sharpe,4),
        'Alpha_Max_Drawdown':round(Alpha_Max_Drawdown,4),
        'Win_Ratio':round(Win_Ratio,4),
        'Profit_Lose_Ratio':round(Profit_Lose_Ratio,4)

    }


    # 回测图绘制
    import matplotlib.pyplot as plt
    
    x = performance_cumnet.index
    y1 = performance_cumnet['strategy']
    y2 = performance_cumnet[benchmark_index]
    y3 = performance_cumnet['alpha']


    fig, ax = plt.subplots()

    ax.plot(x, y1, label='strategy',color = 'darkred')
    ax.plot(x, y2, label=benchmark_index)
    ax.plot(x, y3, label='alpha')
    plt.title(name)

    # 调整子图的布局，留出空间给表格
    plt.subplots_adjust(right=1.4, top=1.1)
    # 创建一个额外的空白子图
    # 添加表格
    cell_text =  [['index','value']] + [list(result.items())][0]
    table = ax.table(cellText=cell_text, loc='right')

    # 调整表格的大小
    #table.scale(0.7, 0.7)

    # 设置单元格的属性
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 对第一行进行处理
            cell.set_text_props(fontsize=10, ha='center', va='center')  # 居中对齐
        elif j == 0:  # 对第一列进行处理
            cell.set_text_props(fontsize=10, ha='left', va='center')  # 左对齐
        else:
            cell.set_text_props(fontsize=10, ha='right', va='center')  # 右对齐

    # 设置行高
    for i in range(len(cell_text)):
        table._cells[(i, 0)].set_height(0.0454)
        table._cells[(i, 1)].set_height(0.0453)
    table.auto_set_column_width([0, 1])
    table.auto_set_font_size(False)

    # 显示图例
    ax.legend()
    fig.savefig(os.path.join(fig_save_path,f'backtest_result_{option}.png'), bbox_inches='tight')

    return result

def main_A_all():
    option_list = ['equal_weight','equal_vol','min_variance','min_active_variance',
                   'RiskParity','utility_theory','max_sharpe','max_ir','min_tr'] 
    
    fig_save_path = os.path.join(fc.ROOT_DIR,'csi500')


    for option in option_list:
        # if equal_weight_weight.pkl not exist
        if not os.path.exists(os.path.join(fig_save_path,f'{option}.pkl')):
            print(f'generate {option}.pkl')
            # 获取因子
            df = pd.read_pickle(os.path.join(fig_save_path,'icirw_combo_neu.pkl'))
            df.columns = [change_stock_name(i) for i in df.columns]

            buy_list = get_buy_list(df,rank_n=20) #top_type='quantile',quantile_q=0.0
            # 券池数量
            buy_list.count(axis = 1).plot()

            df_weight = optimizer(buy_list,option = option,weight_up_limit = 0.15,freq = 20)
            df_weight.to_pickle(os.path.join(fig_save_path,f'{option}.pkl'))
        

        df_weight = pd.read_pickle(os.path.join(fig_save_path,f'{option}.pkl'))
        print('backtest start')
        df_weight.index = pd.to_datetime(df_weight.index)

        account_result = backtest(df_weight, change_n = 20, cash = 10000 * 1000, tax = 0.0, other_tax = 0.0, commission = 0.0, min_fee = 0, cash_interest_yield = 0.02)
        performance_result = get_performance_analysis(account_result, option)


if __name__=='__main__':
    option_list = ['equal_weight','equal_vol','min_variance','min_active_variance',
                   'RiskParity','utility_theory','max_sharpe','max_ir','min_tr'] 
    
    fig_save_path = os.path.join(fc.ROOT_DIR,'csi500_50_fee')

    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)


    for option in option_list:
        # if equal_weight_weight.pkl not exist
        if True:# os.path.exists(os.path.join(fig_save_path,f'{option}.pkl')):
            print(f'generate {option}.pkl')
            # 获取因子
            df = pd.read_parquet('prediction_xgb_outsample.parquet')
            df = df.pivot_table(index='Date',columns='Symbol',values='pred')
            df = df[df.index > '2019-01-01']

            # choose only csi500 stocks
            index = pd.read_parquet(os.path.join(fc.BARR_DIR,'idx__csi500_weight.parquet'))
            index = index.pivot_table(index='Date',columns='Symbol',values='Weight')
            index.index = pd.to_datetime(index.index, format='%Y%m%d')
            index = index[index.index > '2019-01-01']

            df = df[list(set(df.columns) & set(index.columns))]
            
            buy_list = get_buy_list(df,rank_n=50) #top_type='quantile',quantile_q=0.0
            # 券池数量
            buy_list.count(axis = 1).plot()

            fig = plt.figure(figsize=(10,6))
            turnover_df = get_turnover(buy_list, change_n=20)
            plt.plot(turnover_df.index,turnover_df['turnover'])
            plt.title(f'turnover_{option}')
            fig.savefig(os.path.join(fig_save_path,f'turnover_{option}.png'))


            df_weight = optimizer(buy_list,option = option,weight_up_limit = 0.15,freq = 20,benchmark = '000905.SH')
            df_weight.to_pickle(os.path.join(fig_save_path,f'{option}.pkl'))
            
    

        df_weight = pd.read_pickle(os.path.join(fig_save_path,f'{option}.pkl'))
        print('backtest start')
        df_weight.index = pd.to_datetime(df_weight.index)

        account_result = backtest(df_weight, change_n = 20, cash = 10000 * 1000, cash_interest_yield = 0.02)
        performance_result = get_performance_analysis(account_result, option, fig_save_path, benchmark_index='000905.SH')