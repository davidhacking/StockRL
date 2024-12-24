import requests
import json
import argparse
from futu import *
from futu.common import *

base_url = "http://127.0.0.1:5555/"

def balance_info():
    url = base_url + "balance_info"
    try:
        response = requests.get(url)
        response.raise_for_status()
        rt = response.text
        response_text = rt.replace("'", '"')
        data = json.loads(response_text)
        return data['资金余额']
    except Exception as e:
        print(f"balance_info failed: {e}")

def today_trades():
    url = base_url + "today_trades"
    try:
        response = requests.get(url)
        response.raise_for_status()
        rt = response.text
        response_text = rt.replace("'", '"')
        data = json.loads(response_text)
        buys = dict()
        sells = dict()
        for item in data:
            code = item['证券代码']
            num = item['成交数量']
            market = 'SH' if item['交易市场'] == '上海' else 'SZ'
            if item['买卖'] == '证券卖出':
                sells[market + "." + code] = num
            else:
                buys[market + "." + code] = num
        return {'buy': buys, 'sell': sells}
    except Exception as e:
        print(f"today_trades failed: {e}")

def today_entrusts():
    url = base_url + "today_entrusts"
    try:
        response = requests.get(url)
        response.raise_for_status()
        rt = response.text
        print(rt)
        return rt
    except Exception as e:
        print(f"today_entrusts failed: {e}")

def position_info():
    url = base_url + "position_info"
    try:
        response = requests.get(url)
        response.raise_for_status()
        rt = response.text
        response_text = rt.replace("'", '"')
        data = json.loads(response_text)
        res = {}
        for item in data:
            market = 'SH' if item['交易市场'] == '上海' else 'SZ'
            res[item['证券代码'] + "." + market] = item['实际数量']
        return res
    except Exception as e:
        print(f"position_info failed: {e}")

def buy_stock(code, price, qty):
    buy_info = f"buy?code={code}&qty={qty}&price={price}"
    url = base_url + buy_info
    print("buy_stock=", buy_info)
    try:
        response = requests.get(url)
        response.raise_for_status()
        rt = response.text
        print('buy_stock return=', rt)
        return
    except Exception as e:
        print(f"buy_stock failed: {e}")

def sell_stock(code, price, qty):
    sell_info = f"sell?code={code}&qty={qty}&price={price}"
    url = base_url + f"sell?code={code}&qty={qty}&price={price}"
    print("sell_stock=", sell_info)
    try:
        response = requests.get(url)
        response.raise_for_status()
        rt = response.text
        print('sell_stock return=', rt)
        return
    except Exception as e:
        print(f"sell_stock failed: {e}")

def test():
    url = base_url + f"test"
    try:
        response = requests.get(url)
        response.raise_for_status()
        rt = response.text
        return rt
    except Exception as e:
        print(f"sell_stock failed: {e}")


def cur_price(code):
    print(f"cur_price {code}")
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret_sub, err_message = quote_ctx.subscribe([code], [SubType.RT_DATA], subscribe_push=False)
    if ret_sub == RET_OK:
        ret, data = quote_ctx.get_rt_data(code)
        if ret == RET_OK:
            data['time'] = pd.to_datetime(data['time'])
            filtered_df = data[data['code'] == code]
            latest_record = filtered_df.loc[filtered_df['time'].idxmax()]
            return latest_record['cur_price']
        else:
            return 0
    else:
        print('subscription failed', err_message)
    quote_ctx.close()
    return 0

def remove_market(code):
    parts = code.split('.')
    return f"{parts[1]}"

def complete_trade():
    """before call this func, you should revert order in ths"""
    res = today_trades()
    with open("/home/david/MF/github/StockRL/utils/selllist.json", 'r') as file:
        need_sell = json.load(file)
    with open("/home/david/MF/github/StockRL/utils/buylist.json", 'r') as file:
        need_buy = json.load(file)
    incomplete_buys = {}
    incomplete_sells = {}
    print(f"today_trades={res}")
    if len(res['buy'].keys()) == 0 and len(res['sell'].keys()):
        return
    for stock_code, amount in need_buy.items():
        if stock_code not in res['buy']:
            incomplete_buys[stock_code] = need_buy[stock_code]
        elif stock_code in res['buy'] and res['buy'][stock_code] != amount:
            incomplete_buys[stock_code] = amount - res['buy'][stock_code]

    for stock_code, amount in need_sell.items():
        if stock_code not in res['sell']:
            incomplete_sells[stock_code] = need_sell[stock_code]
        elif stock_code in res['sell'] and res['sell'][stock_code] != amount:
            incomplete_sells[stock_code] = amount - res['sell'][stock_code]
    print(f"incomplete_sells={incomplete_sells}")
    print(f"incomplete_buys={incomplete_buys}")
    for code, qty in incomplete_sells.items():
        price = cur_price(code)
        sell_stock(remove_market(code), price, int(qty))
    time.sleep(60)
    for code, qty in incomplete_buys.items():
        price = cur_price(code)
        buy_stock(remove_market(code), price, int(qty))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mode parameter.")
    parser.add_argument('--mode', type=int, default=1, help='Mode(default: 1)')
    args = parser.parse_args()
    mode = args.mode
    # print(test())
    # print(balance_info())
    if mode == 1:
        print(position_info())
    elif mode == 2:
        complete_trade()
    # print(today_entrusts())
    # buy_stock("SZ.000100", 4.97, 200)
    # sell_stock("SZ.000100", 4.97, 100)