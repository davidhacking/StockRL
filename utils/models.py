from typing import Any
import pandas as pd
import numpy as np
import time

from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from utils import config
from utils import ths_trader
from utils.preprocessors import split_data
from utils.env import StockLearningEnv
from abc import ABC, abstractmethod
from copy import deepcopy
from futu import *
from futu.common import *

def create_dataframes(daily_profits, index2date):
    # 创建一个空列表，用于存储每只股票的三列数据（stock_code, date, profits）
    data_list_1 = []
    # 创建一个空字典，用于存储每只股票的总收益（stock_code, total_profit）
    total_profits_dict = {}

    for stock_code, daily_profit_list in daily_profits.items():
        for i, profit in enumerate(daily_profit_list):
            date = index2date[i]
            data_list_1.append([stock_code, date, profit])

            if stock_code not in total_profits_dict:
                total_profits_dict[stock_code] = profit
            else:
                total_profits_dict[stock_code] += profit

    # 创建第一个DataFrame，包含stock_code, date, profits三列
    df_1 = pd.DataFrame(data_list_1, columns=["stock_code", "date", "profits"])

    # 创建第二个DataFrame，包含stock_code, total_profit两列
    df_2 = pd.DataFrame(list(total_profits_dict.items()), columns=["stock_code", "total_profit"])
    df_2 = df_2.sort_values(by="total_profit", ascending=False)
    return df_1, df_2

class UserStockAccount(ABC):
    def __init__(self, code2index):
        self.code2index = code2index # 股票代码到index的映射
        self.b = 0 # 当前剩余现金
        self.h = np.zeros(len(code2index), dtype=float) # 股票持有数量
        self.closings = np.zeros(len(code2index), dtype=float) # 收盘价
        self.action_history = [] # 交易记录
        self.h_history = [] # 股票持有数量历史
        self.b_history = [] # 剩余现金历史
        self.closings_history = [] # 收盘价历史
        self.total_assets_history = [] # 总资产历史
        self.profits_history = [] # 每日收益
    @abstractmethod
    def curr_holds(self):
        raise ValueError("not implemented")
    
    @abstractmethod
    def take_action(self, action, **kwargs):
        raise ValueError("not implemented")
    
    def statisic(self):
        """
        一只股票的收益情况
        前一条价格p1，前一天持仓h1
        当天价格p2，当天持仓h2，当日买卖数量n（n为负数表示卖出，并且买卖价格为p2）
        则当日该股票的收益为：p2 * h1 - p1 * h1
        """
        num_days = len(self.closings_history)
        daily_profits = {}
        index2date = {}
        for code, index in self.code2index.items():
            daily_profits[code] = []
            for i in range(1, num_days):
                p1 = self.closings_history[i - 1][index]
                h1 = self.h_history[i - 1][index]
                p2 = self.closings_history[i][index]
                h2 = self.h_history[i][index]
                n = self.action_history[i][1][index]
                index2date[i-1] = self.action_history[i][0]
                daily_profits[code].append(p2 * h1 - p1 * h1)
        return create_dataframes(daily_profits, index2date)

class LocalUserStockAccount(UserStockAccount):
    def __init__(self, code2index):
        super().__init__(code2index)
        self.b = 1e6

    def curr_holds(self):
        return self.b, self.h
    
    def take_action(self, date, action, **kwargs):
        if len(self.action_history) == 0:
            self.action_history.append(None)
            self.h_history.append(self.h)
            self.b_history.append(self.b)
            self.total_assets_history.append(self.b)
            self.closings_history.append(self.closings)
            self.profits_history.append(0)
        spend, costs, coh = kwargs["spend"], kwargs["costs"], kwargs["cash_and_sell_stock_money"]
        self.closeings = kwargs["closings"]
        flag = (spend + costs) > coh
        self.action_history.append((date, action, flag, spend, costs, coh))
        if not flag:
            self.h = self.h + action
            self.b = coh - spend - costs
        self.h_history.append(self.h)
        self.b_history.append(self.b)
        self.closings_history.append(self.closeings)
        self.total_assets_history.append(self.b + np.dot(self.h, self.closeings))
        self.profits_history.append(self.total_assets_history[-1] - self.total_assets_history[-2])

def revert_code(code):
    parts = code.split('.')
    return f"{parts[1]}.{parts[0]}"

def remove_market(code):
    parts = code.split('.')
    return f"{parts[0]}"

def rebuild_h(h, code2index, qty_info):
    new_h = np.zeros_like(h)
    for code, qty in qty_info.items():
        if code in code2index:
            index = code2index[code]
            new_h[index] = qty

    return new_h

def get_buylist_and_selllist(action, code2index):
    buylist = {}
    selllist = {}

    for code, index in code2index.items():
        qty = action[index]
        reverted_code = revert_code(code)

        if qty > 0:
            buylist[reverted_code] = qty
        elif qty < 0:
            selllist[reverted_code] = abs(qty)
    print('selllist=', selllist)
    print('buylist=', buylist)
    return buylist, selllist

def buy_cn_stock(code, price, qty, flag):
    print(f"buy_cn_stock code={code} qty={qty} price={price}, flag={flag}")
    trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.CN, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
    ret, data = trd_ctx.unlock_trade('')  # 若使用真实账户下单，需先对账户进行解锁。此处示例为模拟账户下单，也可省略解锁。
    if ret != RET_OK:
        print("unlock_trade ret=", ret)
        return
    trd_side = TrdSide.BUY if flag else TrdSide.SELL
    ret, data = trd_ctx.place_order(price=price, qty=qty, code=code, trd_side=trd_side, trd_env=TrdEnv.SIMULATE)
    if ret != RET_OK:
        print(f"buy place_order code={code} ret={ret}")
    trd_ctx.close()
    time.sleep(5)

class FutuUserStockAccount(UserStockAccount):
    def __init__(self, code2index):
        super().__init__(code2index)

    def curr_holds(self):
        trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.CN, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
        ret, data = trd_ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
        if ret != RET_OK:
            print("accinfo_query ret=", ret)
            return self.b, self.h
        self.b = data["cash"][0]
        ret, data = trd_ctx.position_list_query(trd_env=TrdEnv.SIMULATE)
        if ret != RET_OK:
            print("position_list_query ret=", ret)
            return self.b, self.h
        data['code'] = data['code'].apply(revert_code)
        qty_info = pd.Series(data.qty.values, index=data.code).to_dict()
        self.h = rebuild_h(self.h, self.code2index, qty_info)
        print(f"FutuUserStockAccount cash={self.b} qty_info={qty_info}")
        return self.b, self.h
    
    def take_action(self, date, action, **kwargs):
        print("close_dict=", self.close_dict)
        buys, sells = get_buylist_and_selllist(action, self.code2index)
        for code, qty in sells.items():
            price = self.close_dict.get(revert_code(code), 0)
            buy_cn_stock(code, price, qty, False)
        for code, qty in buys.items():
            price = self.close_dict.get(revert_code(code), 0)
            buy_cn_stock(code, price, qty, True)

class ThsUserStockAccount(UserStockAccount):
    def __init__(self, code2index):
        super().__init__(code2index)

    def curr_holds(self):
        self.b = ths_trader.balance_info()
        qty_info = ths_trader.position_info()
        self.h = rebuild_h(self.h, self.code2index, qty_info)
        print(f"ThsUserStockAccount cash={self.b} qty_info={qty_info}")
        return self.b, self.h
    
    def take_action(self, date, action, **kwargs):
        print("close_dict=", self.close_dict)
        print("take_action=", action)
        buys, sells = get_buylist_and_selllist(action, self.code2index)
        for code, qty in sells.items():
            price = self.close_dict.get(revert_code(code), 0)
            ths_trader.sell_stock(remove_market(code), price, qty)
        for code, qty in buys.items():
            price = self.close_dict.get(revert_code(code), 0)
            ths_trader.buy_stock(remove_market(code), price, qty)

UserStockAccountFactory = {
    "LocalUserStockAccount": LocalUserStockAccount,
    "FutuUserStockAccount": FutuUserStockAccount,
    "ThsUserStockAccount": ThsUserStockAccount,
}

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
MODEL_KWARGS = {x: config.__dict__["{}_PARAMS".format(x.upper())] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise
}

def get_transactions(actions, closings, holdings) -> np.ndarray:
    """获取实际交易的股数"""
    actions = actions * config.ENV_PARAMS['hmax']

    # 收盘价为 0 的不进行交易
    actions = np.where(closings > 0, actions, 0)

    # 去除被除数为 0 的警告
    out = np.zeros_like(actions)
    zero_or_not = closings != 0
    actions = np.divide(actions, closings, out=out, where = zero_or_not)
    if config.ENV_PARAMS['normalize_buy_sell']:
        actions = np.sign(actions) * (np.abs(actions) // 100) * 100
    # 不能卖的比持仓的多
    actions = np.maximum(actions, -np.array(holdings))
    # 将 -0 的值全部置为 0
    actions[actions == -0] = 0
    return actions

def get_spend_and_rest_money(transactions, closings, cash_on_hand):
    sells = -np.clip(transactions, -np.inf, 0)
    proceeds = np.dot(sells, closings)
    buy_cost_pct = 3e-3
    sell_cost_pct = 3e-3
    fixed_fee = 0
    costs = proceeds * sell_cost_pct + fixed_fee
    coh = cash_on_hand + proceeds # 计算现金的数量
    buys = np.clip(transactions, 0, np.inf)
    spend = np.dot(buys, closings)
    costs += spend * buy_cost_pct + fixed_fee
    return spend, costs, coh

class DRL_Agent():
    """强化学习交易智能体

    Attributes:
        env: 强化学习环境
    """

    @staticmethod
    def DRL_prediction(
        model: Any, environment: Any,
        user_stock_account: UserStockAccount = None,
        ) -> pd.DataFrame:
        """回测函数"""

        test_env, test_obs = environment.get_sb_env()

        account_memory = []
        actions_memory = []
        test_env.reset()
        holdings = None
        cash_on_hand = None
        if user_stock_account is not None:
            b, h = user_stock_account.curr_holds()
            holdings = h
            cash_on_hand = b
            start_index = 1 + len(h)
            stock_info = test_obs[0][start_index:]
            arr = [b] + list(h) + list(stock_info)
            test_obs[0] = np.array(arr)
            print(f"after obs={test_obs[0]}")
        len_environment = len(environment.df.index.unique())
        for i in range(len_environment):
            predict_action, _states = model.predict(test_obs)
            if user_stock_account is not None:
                action = deepcopy(predict_action)
                closings = test_env.env_method(method_name="get_closings")[0]
                print("action:", action)
                print("closings:", closings)
                action = get_transactions(action, closings, holdings)
                print("action2", action)
                res = get_spend_and_rest_money(action, closings, cash_on_hand)
                spend, costs, coh = res
                user_stock_account.take_action(
                    date=test_env.env_method(method_name="get_dates")[0],
                    action=action[0],
                    spend=spend[0],
                    costs=costs[0],
                    cash_and_sell_stock_money=coh[0],
                    closings=closings
                )
            test_obs, _, dones, _ = test_env.step(predict_action)
            if i == (len_environment - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("回测完成!")
                break
        return account_memory[0], actions_memory[0]
    
    @staticmethod
    def predict_once(
        model: Any, environment: Any, n = 3
        ):
        flag = False
        while not flag and n > 0:
            test_env, test_obs = environment.get_sb_env()
            test_env.reset()
            action, _ = model.predict(test_obs)
            action = environment.get_transactions(action)
            spend, costs, coh = environment.get_spend_and_rest_money()
            flag = (spend + costs) > coh
            n -= 1
        return action, flag

    def __init__(self, env: Any) -> None:
        self.env = env

    def get_model(
        self,
        model_name: str,
        policy: str = "MlpPolicy",
        policy_kwargs: dict = None,
        model_kwargs: dict = None,
        verbose: int = 1
    ) -> Any:
        """根据超参数生成模型"""
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        
        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]
        
        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)

        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log="{}/{}".format(config.TENSORBOARD_LOG_DIR, model_name),
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs
        )
        
        return model

    def train_model(
        self, model: Any, tb_log_name: str, total_timesteps: int = 5000
        ) -> Any:
        """训练模型"""
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        return model

if __name__ == "__main__":
    print(ths_trader.balance_info())
    print(ths_trader.position_info())