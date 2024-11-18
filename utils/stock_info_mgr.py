from typing import List
import tushare as ts
import pandas as pd
from utils import config
import time
from datetime import datetime
import os
from futu import *
from futu.common import *
from utils.preprocessors import FeatureEngineer, split_data

class StockInfoMgr(object):
    @staticmethod
    def read_pd(file_path):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return pd.DataFrame()
            df.dropna(how='all', inplace=True)
            if df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            print("read_pd fail e=", e)
            return pd.DataFrame()

    def __init__(self) -> None:
        self.stock_info = self.read_pd(config.stock_info_path)
        self.stock_info_for_train = self.read_pd(config.stock_info_for_train_path)
        ts.set_token(config.Tushare_Tocken)
        self.ticker_list = config.SSE_50

    def check_date(self):
        if self.stock_info.empty:
            return config.stock_info_start_date
        max_date = self.stock_info['date'].max()
        current_date = datetime.now().strftime('%Y-%m-%d')
        if max_date == current_date:
            result = None
        else:
            max_date_formatted = datetime.strptime(max_date, '%Y-%m-%d').strftime('%Y%m%d')
            result = max_date_formatted
        return result
    
    def merge_stock_info(self, stock_info):
        self.stock_info = pd.concat([self.stock_info, stock_info], ignore_index=True)
        self.stock_info = self.stock_info.drop_duplicates(subset=['tic', 'date'], keep='last')
        self.stock_info = self.stock_info.sort_values(by=['date','tic']).reset_index(drop=True)
        self.stock_info.sort_values(['date', 'tic'], ignore_index=True)
    
    def get_stock_from_futu(self):
        print("sync data from futu")
        start_date = datetime.now().strftime('%Y-%m-%d')
        def get_stocks(stock_code):
            quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
            ret, data, page_req_key = quote_ctx.request_history_kline(stock_code, start=start_date, end=None, ktype=KLType.K_DAY, fields=[constant.KL_FIELD.ALL], max_count=50)
            page_num = 0
            res = None
            if ret == RET_OK:
                res = data
            else:
                print('get_stock_from_futu error:', data)
                return None
            while page_req_key != None:
                ret, data, page_req_key = quote_ctx.request_history_kline(stock_code, start=start_date, end=None, max_count=50, page_req_key=page_req_key) # 请求翻页后的数据
                if ret == RET_OK:
                    res = pd.concat([res, data], ignore_index=True)
                else:
                    print('get_stock_from_futu error:', data)
            quote_ctx.close()
            return res
        res = pd.DataFrame()
        for tic in config.SSE_50:
            parts = tic.split('.')
            download_data = get_stocks(f"{parts[1]}.{parts[0]}")
            if download_data is None:
                continue
            res = pd.concat([res, download_data], ignore_index=True)
        if res.empty:
            return res
        res.rename(columns={'code': 'tic'}, inplace=True)
        res["date"] = res.time_key.apply(lambda x: datetime.strptime(x[:4] + '-' + x[5:7] + '-' + x[8:10], "%Y-%m-%d"))
        res["day"] = res["date"].dt.dayofweek
        res["date"] = res.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        res = res[['tic', 'date', 'open', 'high', 'low', 'close', 'volume', 'day']]
        return res
    
    def save_stock_info(self):
        self.stock_info.to_csv(config.stock_info_path, index=False)
    
    def save_stock_info_for_train(self):
        self.stock_info_for_train.to_csv(config.stock_info_for_train_path, index=False)
    
    def process_data_for_train(self):
        self.stock_info_for_train = FeatureEngineer(use_technical_indicator=True).preprocess_data(self.stock_info)
        self.stock_info_for_train['amount'] = self.stock_info_for_train.volume * self.stock_info_for_train.close
        self.stock_info_for_train['change'] = (self.stock_info_for_train.close - self.stock_info_for_train.open) / self.stock_info_for_train.close
        self.stock_info_for_train['daily_variance'] = (self.stock_info_for_train.high - self.stock_info_for_train.low) / self.stock_info_for_train.close
        self.stock_info_for_train = self.stock_info_for_train.fillna(0)
    
    def merge_stock_info_and_process(self, stock_info):
        self.merge_stock_info(stock_info)
        self.save_stock_info()
        self.process_data_for_train()
        self.save_stock_info_for_train()
    
    def sync_data(self):
        start_download_date = self.check_date()
        if start_download_date is None:
            print("no need to sync data")
            return self.stock_info_for_train
        
        current_date = datetime.now().strftime('%Y%m%d')
        print("download from tushare: ", start_download_date, current_date)
        download_data = self.pull_data(start_download_date, current_date)
        self.merge_stock_info_and_process(download_data)
        start_download_date = self.check_date()
        if start_download_date is None:
            print("sync data from tushare success")
            return self.stock_info_for_train
        futu_stock_info = self.get_stock_from_futu()
        self.merge_stock_info_and_process(futu_stock_info)
        return self.stock_info_for_train

    def get_stock_info(self):
        return self.stock_info
    
    def get_stock_info_for_train(self):
        return self.stock_info_for_train
    
    def pull_data(self, start_date, end_date) -> pd.DataFrame:
        """从 Tushare API 拉取数据"""
        data_df = pd.DataFrame()
        stock_num = 0

        print("   --- 开始下载 ----")
        for ticker in self.ticker_list:
            stock_num += 1
            if stock_num % 10 == 0:
                print("   下载进度 : {}%".format(stock_num / len(self.ticker_list) * 100))
            
            try:
                data_tmp = ts.pro_bar(ts_code=ticker, adj='qfq', 
                                        start_date=start_date, end_date=end_date)
                data_df = pd.concat([data_df, data_tmp], ignore_index=True)
            except:
                print("tushare 积分不足或其他异常情况, 请自行检查, 3s 后重试")
                time.sleep(3)
        print("   --- 下载完成 ----")

        # 删除一些列并更改列名
        data_df = data_df.reset_index()
        data_df = data_df.drop(["index", "pre_close", "change", "pct_chg", "amount"], axis = 1)
        data_df.columns = ["tic", "date", "open", "high", "low", "close", "volume"]

        # 更改 date 列数据格式, 添加 day 列(星期一为 0), 再将格式改回成 str
        data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # 删除为空的数据行
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

        print("DataFrame 的大小: ", data_df.shape)
        return data_df