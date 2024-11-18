import sys
sys.path.append("..")

from utils import stock_info_mgr

if __name__ == "__main__":
    mgr = stock_info_mgr.StockInfoMgr()
    mgr.sync_data()
    print(mgr.get_stock_info_for_train().tail())