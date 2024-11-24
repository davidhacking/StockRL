import codecs
import os
import sys
import json
import time
import pandas as pd

from futu import *
from futu.common import *
from argparse import ArgumentParser
from utils import config
from utils import stock_info_mgr
from utils.env import StockLearningEnv
from utils.models import DRL_Agent, UserStockAccountFactory
from datetime import datetime
import tushare as ts
from utils.preprocessors import FeatureEngineer, split_data
import multiprocessing
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

sys.path.append("..")
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', 'learn', 'exps')

def get_max_date(stock_info):
    max_date = stock_info['date'].max()
    return datetime.strptime(max_date, '%Y-%m-%d').strftime('%Y%m%d')

def download_from_tushare(stocks, start_date, end_date, index_data=False, need_pe=False):
    data_df = pd.DataFrame()
    for ticker in stocks:
        try:
            if index_data:
                func = ts.pro_api().index_daily
            else:
                func = ts.pro_bar
            data_tmp = func(ts_code=ticker, adj='qfq', 
                                    start_date=start_date, end_date=end_date)
            data_df = pd.concat([data_df, data_tmp], ignore_index=True)
        except Exception as e:
            print("tushare error check scores, retry after 3s, except:", e)
            time.sleep(3)
    data_df = data_df.reset_index()
    data_df = data_df.drop(["index", "pre_close", "change", "pct_chg", "amount"], axis = 1)
    data_df.columns = ["tic", "date", "open", "high", "low", "close", "volume"]
    data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
    data_df["day"] = data_df["date"].dt.dayofweek
    data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    if need_pe:
        data_df["pe_ratio"] = 0
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
    return data_df

def download_from_futu(stocks, start_date, end_date, need_pe=False):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    def get_stocks(stock_code):
        ret, data, page_req_key = quote_ctx.request_history_kline(stock_code, start=start_date, end=end_date, ktype=KLType.K_DAY, fields=[constant.KL_FIELD.ALL], max_count=50)
        page_num = 0
        res = None
        if ret == RET_OK:
            res = data
        else:
            print('get_stock_from_futu error:', data)
            return None
        while page_req_key != None:
            ret, data, page_req_key = quote_ctx.request_history_kline(stock_code, start=start_date, end=end_date, max_count=50, page_req_key=page_req_key) # 请求翻页后的数据
            if ret == RET_OK:
                res = pd.concat([res, data], ignore_index=True)
            else:
                print('get_stock_from_futu error:', data)
        return res
    res = pd.DataFrame()
    for tic in stocks:
        parts = tic.split('.')
        download_data = get_stocks(f"{parts[1]}.{parts[0]}")
        if download_data is None:
            continue
        res = pd.concat([res, download_data], ignore_index=True)
    quote_ctx.close()
    if res.empty:
        return res
    res.rename(columns={'code': 'tic'}, inplace=True)
    res["date"] = res.time_key.apply(lambda x: datetime.strptime(x[:4] + '-' + x[5:7] + '-' + x[8:10], "%Y-%m-%d"))
    res["day"] = res["date"].dt.dayofweek
    res["date"] = res.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    res['tic'] = res['tic'].apply(lambda x: '.'.join(x.split('.')[::-1]))
    if need_pe:
        res = res[['tic', 'date', 'open', 'high', 'low', 'close', 'volume', 'day', 'pe_ratio']]
    else:
        res = res[['tic', 'date', 'open', 'high', 'low', 'close', 'volume', 'day']]
    res = res.sort_values(by=['date', 'tic']).reset_index(drop=True)
    return res

def process_data_for_train(origin_stock_info):
    origin_stock_info.sort_values(['date', 'tic'], ignore_index=True)
    stock_info = FeatureEngineer(use_technical_indicator=True).preprocess_data(origin_stock_info)
    stock_info['amount'] = stock_info.volume * stock_info.close
    stock_info['change'] = (stock_info.close - stock_info.open) / stock_info.close
    stock_info['daily_variance'] = (stock_info.high - stock_info.low) / stock_info.close
    stock_info = stock_info.fillna(0)
    return stock_info

def sum_ascii_chars(string):
    total = 0
    for char in string:
        total += ord(char)
    return total

def get_daily_return(
    df: pd.DataFrame,
    value_col_name: str = "account_value"
):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")

    return pd.Series(df["daily_return"], index = df.index)


def financial_metrics(returns):
    # 将NaN值替换为0 
    returns = returns.fillna(0)

    # 计算累计收益率
    cumulative_return = (1 + returns).prod() - 1 

    # 计算最大回撤率
    cumulative_wealth_index = (1 + returns).cumprod()
    previous_peaks = cumulative_wealth_index.cummax()
    drawdowns = (cumulative_wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdowns.min()

    # 计算年化收益率和年化波动率
    annualized_return = np.power(1 + cumulative_return, 252 / len(returns)) - 1 
    annualized_vol = returns.std() * np.sqrt(252)

    # 计算Sharpe比率 (假设无风险利率为0)
    sharpe_ratio = annualized_return / annualized_vol

    # 计算Omega比率
    threshold_return = 0  # 设定阈值收益率为0 
    omega_numerator = returns[returns > threshold_return].sum()
    omega_denominator = -returns[returns < threshold_return].sum()
    omega_ratio = omega_numerator / omega_denominator if omega_denominator != 0 else np.nan

    result_dict = {
        "累计收益率": cumulative_return,
        "最大回撤率": max_drawdown,
        "年化收益率": annualized_return,
        "年化波动率": annualized_vol,
        "Sharpe比率": sharpe_ratio,
        "Omega比率": omega_ratio
    }
    for key, value in result_dict.items():
        if isinstance(value, float):
            result_dict[key] = "{:.2%}".format(value)
    return result_dict

def backtest_plot(
    baseline_df,
    account_value_dict: dict,
    value_col_name: str = "account_value"
):
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    baseline_fdata = financial_metrics(baseline_returns)
    res = {}
    res['baseline'] = baseline_fdata
    def handle_account_value(account_value):
        df = deepcopy(account_value)
        test_returns = get_daily_return(df, value_col_name=value_col_name)
        return financial_metrics(test_returns)
    for k, v in account_value_dict.items():
        res[k] = handle_account_value(v)
    return res

def fill_na_stock(stock_info):
    all_dates = stock_info['date'].unique()
    stock_list = stock_info['tic'].unique()
    all_combinations = pd.MultiIndex.from_product([all_dates, stock_list], names=['date', 'tic'])
    all_df = pd.DataFrame(index=all_combinations).reset_index()
    merged_df = pd.merge(all_df, stock_info, on=['date', 'tic'], how='left')
    merged_df.fillna(0, inplace=True)
    return merged_df


class RunExpStrategy(object):
    
    def __init__(self):
        self.model_names = deepcopy(config.MODEL_LIST)
        self.env_params = deepcopy(config.ENV_PARAMS)
        self.trade_env_params = deepcopy(config.ENV_PARAMS)
        self.rerun_test = False
        self.state_init_func = "AllCashStateIntiator"
        self.user_stock_account = "LocalUserStockAccount"
        self.user_stock_account_ins_dict = {}

    def get_stock_profit(self):
        return {
            k: v.statisic() for k, v in self.user_stock_account_ins_dict.items() if v is not None
        }

    def create_exp_dir(self):
        self.exp_dir = os.path.join(root_dir, f"exp_id_{self.exp_id}")
        self.train_file = os.path.join(self.exp_dir, "train.csv")
        self.test_file = os.path.join(self.exp_dir, "test.csv")
        self.all_stocks_data = os.path.join(self.exp_dir, "all_stocks_data.csv")
        self.baseline_stocks_data = os.path.join(self.exp_dir, "baseline_stocks_data.csv")
        os.makedirs(self.exp_dir, exist_ok=True)

    def dump_exp_params(self):
        args_dict = vars(self.options)
        with open(os.path.join(self.exp_dir, "options.json"), 'w') as f:
            json.dump(args_dict, f, indent=4)

    def download_from_tushare(self):
        self.origin_stock_info = download_from_tushare(self.stocks, self.start_date, self.end_date,
                                                       need_pe=self.need_pe)

    def download_from_futu(self):
        start_date = self.futu_start_download_date
        start_date = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y%m%d').strftime('%Y-%m-%d')
        self.origin_stock_info = download_from_futu(self.stocks, start_date, end_date, self.need_pe)

    def split_train_and_test_data(self):
        start_train_date = self.stock_info['date'].min()
        end_train_date = datetime.strptime(self.split_date, '%Y%m%d').strftime('%Y-%m-%d')
        end_test_date = (datetime.strptime(self.stock_info['date'].max(), '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        self.train_data = split_data(self.stock_info, start_train_date, end_train_date)
        self.test_data = split_data(self.stock_info, end_train_date, end_test_date)
        self.train_data.to_csv(self.train_file, index=False)
        self.test_data.to_csv(self.test_file, index=False)
        self.test_codes = self.test_data['tic'].unique()
        self.test_code2index = {
            code: index for index, code in enumerate(self.test_codes)
        }

    def save_model(self, model, model_path):
        model.save(model_path)

    def get_env(self, random_seed=0):
        e_train_gym = StockLearningEnv(df=self.train_data, random_start=True,
                                        **self.env_params)
        env_train, _ = e_train_gym.get_sb_env()
        e_trade_gym = StockLearningEnv(df=self.test_data, random_start=False,
                                       random_seed=random_seed,
                                        **self.trade_env_params)
        env_trade, _ = e_trade_gym.get_sb_env()
        return env_train, env_trade

    def download_stock_data(self):
        if os.path.exists(self.all_stocks_data):
            self.stock_info = pd.read_csv(self.all_stocks_data)
            return
        self.futu_start_download_date = self.start_date
        if not self.no_tushare:
            self.download_from_tushare()
            max_date = get_max_date(self.origin_stock_info)
            if max_date >= self.end_date:
                self.stock_info = process_data_for_train(self.origin_stock_info)
                self.stock_info.to_csv(self.all_stocks_data, index=False)
                return
            self.futu_start_download_date = max_date
        self.download_from_futu()
        if self.need_pe:
            self.pe_data = self.origin_stock_info[['date', 'tic', 'pe_ratio']]
            self.origin_stock_info = self.origin_stock_info.drop(columns=['pe_ratio'])
        self.stock_info = process_data_for_train(self.origin_stock_info)
        if self.need_pe:
            self.stock_info = pd.merge(self.stock_info, self.pe_data, on=['date', 'tic'])
        self.stock_info = fill_na_stock(self.stock_info)
        self.stock_info = self.stock_info.sort_values(by=['date', 'tic']).reset_index(drop=True)
        self.stock_info.to_csv(self.all_stocks_data, index=False)

    def download_baseline_stocks(self):
        if os.path.exists(self.baseline_stocks_data):
            self.baseline_stocks = pd.read_csv(self.baseline_stocks_data)
            return
        self.baseline_stocks = download_from_tushare(self.baseline_stocks_list, self.split_date, self.end_date, True)
        self.baseline_stocks.to_csv(self.baseline_stocks_data, index=False)

    def train_by_model_name(self, model_name):
        model_path = os.path.join(self.exp_dir, "{}.model".format(model_name))
        if os.path.exists(model_path):
            return
        env_train, env_trade = self.get_env(sum_ascii_chars(model_name))
        agent = DRL_Agent(env=env_train)
        model = agent.get_model(model_name,
                                model_kwargs=config.__dict__["{}_PARAMS".format(model_name.upper())], 
                                verbose=0)
        model.learn(total_timesteps=self.total_timesteps, 
                    eval_env=env_trade, 
                    eval_freq=500,
                    log_interval=1, 
                    tb_log_name='env_cashpenalty_highlr',
                    n_eval_episodes=1)
        self.save_model(model, model_path)

    def train_model(self):
        with multiprocessing.Pool() as pool:
            pool.map(self.train_by_model_name, self.model_names)

    def test_model_by_model_name(self, model_name):
        account_value_path = os.path.join(self.exp_dir, "account_value_{}.csv".format(model_name))
        if not self.rerun_test and os.path.exists(account_value_path):
            return
        e_trade_gym = StockLearningEnv(df=self.test_data, random_start=False,
                                       random_seed=sum_ascii_chars(model_name),
                                        **self.trade_env_params)
        agent = DRL_Agent(env=e_trade_gym)
        model = agent.get_model(model_name,  
                                model_kwargs=config.__dict__["{}_PARAMS".format(model_name.upper())], 
                                verbose=0)
        model_path = os.path.join(self.exp_dir, "{}.model".format(model_name))
        model.load(model_path)
        if model is not None:
            df_account_value, df_actions = DRL_Agent.DRL_prediction(model=model, 
                                                                    environment=e_trade_gym,
                                                                    user_stock_account=self.user_stock_account_ins_dict[model_name])
            df_account_value.to_csv(account_value_path, index=False)

            actions_path = os.path.join(self.exp_dir, "actions_{}.csv".format(model_name))
            df_actions.to_csv(actions_path, index=False)

    def test_model(self):
        for model_name in self.model_names:
            self.user_stock_account_ins_dict[model_name] = None
            if isinstance(self.user_stock_account, str) and self.user_stock_account != "":
                self.user_stock_account_ins_dict[model_name] = UserStockAccountFactory[self.user_stock_account](
                    self.test_code2index
                )
            self.test_model_by_model_name(model_name)

    def plot_test_result(self):
        start_close_value = self.baseline_stocks.iloc[0]['close']
        self.baseline_stocks['processed_close'] = ((self.baseline_stocks['close'] - start_close_value)/start_close_value + 1) * 1e+6
        account_value_dict = {}
        for m in self.model_names:
            account_value_dict[m] = pd.read_csv(os.path.join(self.exp_dir, "account_value_{}.csv".format(m)))
        data = {
            m: account_value_dict[m]['total_assets'] for m in self.model_names
        }
        data['baseline'] = self.baseline_stocks['processed_close']
        backtest_curve = pd.DataFrame(data=data)
        backtest_curve = backtest_curve.iloc[:-1].apply(lambda x : (x - 1e+6)/1e+6)
        
        backtest_table = backtest_plot(self.baseline_stocks, account_value_dict,
                        value_col_name = 'total_assets')
        return backtest_table, backtest_curve
    
    def check_and_build_params(self):
        try:
            self.start_date = datetime.strptime(self.options.start_date, "%Y%m%d")
            self.end_date = datetime.strptime(self.options.end_date, "%Y%m%d")
            self.split_date = datetime.strptime(self.options.split_date, "%Y%m%d")
        except ValueError:
            raise ValueError("date format error YYYYMMDD")
        self.start_date = self.options.start_date
        self.end_date = self.options.end_date
        self.split_date = self.options.split_date

        if not (self.start_date < self.split_date < self.end_date):
            raise ValueError("need match condition end_date > split_date > start_date")
        if self.options.a_stock_list not in config.__dict__:
            raise ValueError("a_stock_list not in config")
        self.stocks = config.__dict__[self.options.a_stock_list]
        if self.options.baseline_stocks not in config.__dict__:
            raise ValueError("baseline_stocks not in config")
        self.baseline_stocks_list = config.__dict__[self.options.baseline_stocks]
        self.total_timesteps = self.options.total_timesteps
        self.exp_id = self.options.exp_id
        self.no_tushare = self.options.no_tushare
        self.need_pe = self.options.need_pe
        self.state_init_func = self.options.state_init_func
        if self.need_pe:
            self.env_params["daily_information_cols"].append("pe_ratio")
            self.trade_env_params["daily_information_cols"].append("pe_ratio")
        if self.state_init_func != "":
            self.env_params["state_init_func"] = self.state_init_func
        self.user_stock_account = self.options.user_stock_account

    def add_params(self):
        parser = ArgumentParser(description="set parameters for run a stock exp strategy")
        parser.add_argument(
            '--total_timesteps', '-tts',
            dest='total_timesteps',
            default=200000,
            help='set the total_timesteps when you train the model',
            metavar="TOTAL_TIMESTEPS",
            type=int
        )
        parser.add_argument(
            '--exp_id', '-eid',
            dest='exp_id',
            default=1,
            help='set exp_id',
            metavar="EXP_ID",
            type=int
        )
        parser.add_argument(
            '--start_date', '-startd',
            dest='start_date',
            default="20090101",
            help='set start_date',
            metavar="START_DATE",
            type=str
        )
        parser.add_argument(
            '--end_date', '-ed',
            dest='end_date',
            default="20241122",
            help='set end_date',
            metavar="END_DATE",
            type=str
        )
        parser.add_argument(
            '--split_date', '-splitd',
            dest='split_date',
            default="20230101",
            help='set split_date',
            metavar="SPLIT_DATE",
            type=str
        )
        parser.add_argument(
            '--a_stock_list', '-asl',
            dest='a_stock_list',
            default="SSE_50",
            help='a_stock_list need define in config.py',
            metavar="A_STOCK_LIST",
            type=str
        )
        parser.add_argument(
            '--h_stock_list', '-hsl',
            dest='h_stock_list',
            default="",
            help='h_stock_list need define in config.py',
            metavar="H_STOCK_LIST",
            type=str
        )
        parser.add_argument(
            '--usa_stock_list', '-usasl',
            dest='usa_stock_list',
            default="",
            help='usa_stock_list need define in config.py',
            metavar="USA_STOCK_LIST",
            type=str
        )
        parser.add_argument(
            '--state_init_func', '-sif',
            dest='state_init_func',
            default="AllCashStateIntiator",
            help='state_init_func need define in config.py',
            metavar="STATE_INIT_FUNC",
            type=str
        )
        parser.add_argument(
            '--baseline_stocks', '-bs',
            dest='baseline_stocks',
            default="SSE_50_INDEX",
            help='baseline_stocks need define in config.py',
            metavar="BASELINE_STOCKS",
            type=str
        )
        parser.add_argument(
            '--user_stock_account', '-usa',
            dest='user_stock_account',
            default="",
            help='user_stock_account defined in models.py',
            metavar="USER_STOCK_ACCOUNT",
            type=str
        )
        parser.add_argument(
            '--no_tushare', '-nt',
            action='store_true',
            default=True,
            help='no download_from_tushare'
        )
        parser.add_argument(
            '--need_pe', '-pe',
            action='store_true',
            default=True,
            help='no download_from_tushare'
        )
        self.options = parser.parse_args()

    def main(self):
        self.add_params()
        self.check_and_build_params()
        self.create_exp_dir()
        self.dump_exp_params()
        self.download_stock_data()
        self.download_baseline_stocks()
        self.split_train_and_test_data()
        self.train_model()
        self.test_model()
        self.plot_test_result()
    
    
if __name__ == "__main__":
    RunExpStrategy().main()