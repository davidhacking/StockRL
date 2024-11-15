from typing import List
import pandas as pd
import numpy as np
from pyfolio import timeseries
import pyfolio
from copy import deepcopy

from utils.pull_data import Pull_data
from utils import config

def get_daily_return(
    df: pd.DataFrame,
    value_col_name: str = "account_value"
) -> pd.Series:
    """获取每天的涨跌值"""
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")

    return pd.Series(df["daily_return"], index = df.index)

def backtest_stats(
    account_value: pd.DataFrame, 
    value_col_name: str = "account_value"
) -> pd.Series:
    """对回测数据进行分析"""
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB"
    )
    print(perf_stats_all)

    return perf_stats_all

def backtest_plot(
    account_value: pd.DataFrame,
    baseline_start: str = config.End_Trade_Date,
    baseline_end: str = config.End_Test_Date,
    baseline_ticker: List = config.SSE_50_INDEX,
    value_col_name: str = "account_value"
) -> None:
    """对回测数据进行分析并画图"""
    df = deepcopy(account_value)
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker,
        start=baseline_start,
        end=baseline_end
    )

    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns,
            benchmark_rets=baseline_returns,
            set_context=False
        )

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

def backtest_plot_from_file(
    filepath, get_baseline_func,
    account_value_dict: dict,
    value_col_name: str = "account_value"
) -> dict:
    """对回测数据进行分析并画图"""
    baseline_df = get_baseline_from_file(filepath, get_baseline_func)
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

def get_baseline(
    ticker: List, start: str, end: str
    ) -> pd.DataFrame:
    """获取指数的行情数据"""
    baselines = Pull_data(
        ticker_list=ticker,
        start_date=start,
        end_date=end,
        pull_index=True
    ).pull_data()

    return baselines

def get_baseline_from_file(filepath, get_baseline_func):
    import os
    if not os.path.exists(filepath):
        data = get_baseline_func()
        data.to_csv(filepath, index=False)
        return data
    else:
        return pd.read_csv(filepath)