import requests
import json

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

if __name__ == "__main__":
    # print(test())
    # print(balance_info())
    print(position_info())
    # buy_stock("SZ.000100", 4.97, 200)
    # sell_stock("SZ.000100", 4.97, 100)