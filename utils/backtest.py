import pandas as pd
import numpy as np
from pyfolio import timeseries
import pyfolio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy

from utils.pull_data import Pull_data
from utils import config

# TODO add Readme and descriptions

def get_daily_return(
    df,
    value_col_name="account_value"
):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index = df.index)

def backtest_stats(account_value, value_col_name="account_value"):
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
    account_value,
    baseline_start = config.End_Trade_Date,
    baseline_end = config.End_Test_Date,
    baseline_ticker = config.SSE_50_INDEX,
    value_col_name = "account_value"
):
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

def get_baseline(ticker, start, end):
    baselines = Pull_data(
        ticker_list=ticker,
        start_date=start,
        end_date=end,
        pull_index=True
    ).pull_data()
    return baselines

def trx_plot():
    pass