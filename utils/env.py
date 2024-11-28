from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import gym
import math
import time
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import Logger, configure
from abc import ABC, abstractmethod
import threading
from futu import *
from futu.common import *

def revert_code(code):
    parts = code.split('.')
    return f"{parts[1]}.{parts[0]}"

class StockInfo:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        ret, data = quote_ctx.get_user_security("stock_rl")
        if ret == RET_OK:
            data['code'] = data['code'].apply(revert_code)
            self.alot_info = pd.Series(data.lot_size.values, index=data.code).to_dict()
            self.name2code = pd.Series(data.name.values, index=data.code).to_dict()
        else:
            print('error:', data)
        quote_ctx.close()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def set(self, code, alot):
        self.alot_info[code] = alot
    
    def get(self, code):
        if code not in self.alot_info:
            print(f"error: can not find code {code} in stock_rl")
            return 100
        return self.alot_info[code]
    
    def batch_get(self, codes):
        values = [self.alot_info.get(code, 100) for code in codes]
        return np.array(values, dtype=int)
    
    def set_name_code(self, name, code):
        self.name2code[name] = code
    
    def get_code_by_name(self, name):
        if name not in self.name2code:
            print(f"error: can not find name {name} in stock_rl")
            return 0
        return self.name2code[name]

class StateIntiator(ABC):
    def __init__(self, code2index):
        self.code2index = code2index
    @abstractmethod
    def init_state(self, init_amount, stock_close_info):
        raise ValueError("not implemented")

class AllCashStateIntiator(StateIntiator):
    def __init__(self, code2index):
        super().__init__(code2index)
    def init_state(self, init_amount, stock_close_info):
        b = [init_amount]
        h = [0] * len(stock_close_info)
        return b, h 

class RandomCashAndStateIntiator(StateIntiator):
    def __init__(self, code2index):
        super().__init__(code2index)
    def init_state(self, init_amount, stock_close_info):
        # 随机调整init_amount的值
        new_init_amount = random.uniform(0.5, 2.0) * init_amount

        # 随机划分新的init_amount为两部分
        stock_assets = random.uniform(0, new_init_amount)
        b = new_init_amount - stock_assets

        # 对每只股票按100整数倍进行随机持仓
        stock_indexes = list(range(len(stock_close_info)))
        h = [0] * len(stock_close_info)
        remaining_stock_assets = stock_assets

        # 先随机打乱股票顺序，避免第一只股票被优先分配过多份额
        random.shuffle(stock_indexes)

        for i in stock_indexes:
            if remaining_stock_assets <= 0:
                break
            stock_price = stock_close_info[i]
            if math.isclose(stock_price, 0, abs_tol=1e-4):
                continue
            max_shares = remaining_stock_assets // (stock_price * 100)
            if max_shares > 0:
                num_shares = random.randint(0, max_shares) * 100
            else:   
                num_shares = 0
            h[i] = num_shares
            remaining_stock_assets -= num_shares * stock_price

        # 重新调整每只股票的持仓数量以满足等式，同时确保是100的整数倍
        total_value = sum([h[i] * stock_close_info[i] for i in stock_indexes])
        b += stock_assets - total_value
        return [b], h

StateIntiatorFactory = {
    "AllCashStateIntiator": AllCashStateIntiator,
    "RandomCashAndStateIntiator": RandomCashAndStateIntiator
}

class StockLearningEnv(gym.Env):
    """构建强化学习交易环境

        Attributes
            df: 构建环境时所需要用到的行情数据
            buy_cost_pct: 买股票时的手续费
            sell_cost_pct: 卖股票时的手续费
            date_col_name: 日期列的名称
            hmax: 最大可交易的数量
            print_verbosity: 打印的频率
            initial_amount: 初始资金量
            daily_information_cols: 构建状态时所考虑的列
            cache_indicator_data: 是否把数据放到内存中
            random_start: 是否随机位置开始交易（训练和回测环境分别为True和False）
            patient: 是否在资金不够时不执行交易操作，等到有足够资金时再执行
            currency: 货币单位
    """

    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        df: pd.DataFrame,
        buy_cost_pct: float = 3e-3,
        sell_cost_pct: float = 3e-3,
        date_col_name: str = "date",
        hmax: int = 10,
        print_verbosity: int = 10,
        initial_amount: int = 1e6,
        daily_information_cols: List = ["open", "close", "high", "low", "volume"],
        cache_indicator_data: bool = True,
        random_start: bool = True,
        patient: bool = False,
        currency: str = "￥",
        alpha: float = 1.0,
        normalize_buy_sell: bool = False,
        state_init_func: str = "AllCashStateIntiator",
        random_seed: int = 0,
        fixed_fee: float = 0,
    ) -> None:
        self.df = df
        self.fixed_fee = fixed_fee
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.assets_alot = StockInfo.get_instance().batch_get(self.assets)
        self.code2index = {
            stock: i for i, stock in enumerate(self.assets)
        }
        self.state_init_func = StateIntiatorFactory[state_init_func](self.code2index)
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.random_seed = random_seed
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.hmax = hmax
        self.config_amount = initial_amount
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.daily_information_cols = daily_information_cols
        self.close_index = self.daily_information_cols.index('close')
        self.normalize_buy_sell = normalize_buy_sell
        D = len(self.assets) # 股票数量
        b = 1 # 余额
        h = D # 每只股票的持仓信息
        p = D * len(self.daily_information_cols) # 股票的价格信息
        self.state_space = (
            b + h + p
        )
        """
        对于某支股票，动作空间的定义为 {−k, . . . , −1, 0, 1, . . . , k}，其中 𝑘 和 −𝑘 表示我
        们可以购买和出售的股份数量 k = hmax，因为 RL 算法 A2C 和 PPO 直接使用高斯分布输出策略的分布，
        需要进行归一化处理，所以动作空间被归一化为 [−1,1]。
        """
        self.action_space = spaces.Box(low=-1, high=1, shape=(D,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.max_total_assets = 0
        self.alpha = alpha
        if self.cache_indicator_data:
            """cashing data 的结构:
               [[date1], [date2], [date3], ...]
               date1 : [stock1 * cols, stock2 * cols, ...]
            """
            print("加载数据缓存")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("数据缓存成功!")
        
    def seed(self, seed: Any = None) -> None:
        """设置随机种子"""
        if seed is None:
            seed = int(round(time.time() % 1000))
        seed += self.random_seed
        random.seed(seed)
    
    @property
    def current_step(self) -> int:
        """当前回合的运行步数"""
        return self.date_index - self.starting_point
    
    @property
    def cash_on_hand(self) -> float:
        """当前拥有的现金"""
        return self.state_memory[-1][0]
    
    @property
    def holdings(self) -> List:
        """当前的持仓数据"""
        return self.state_memory[-1][1: len(self.assets) + 1]

    @property
    def closings(self) -> List:
        """每支股票当前的收盘价"""
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def get_dates(self):
        return self.dates[self.date_index]
    
    def get_closings(self):
        return self.closings

    def get_date_vector(self, date: int, cols: List = None) -> List:
        """获取 date 那天的行情数据"""
        if(cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            res = []
            for asset in self.assets:
                tmp_res = trunc_df[trunc_df[self.stock_col] == asset]
                try:
                    res += tmp_res.loc[date, cols].tolist()
                except Exception as e:
                    print("debug", date, cols, tmp_res, self.stock_col, asset)
                    raise ValueError("exception")
            assert len(res) == len(self.assets) * len(cols)
            return res
    
    def reset(self) -> np.ndarray:
        self.seed()
        self.sum_trades = 0
        if self.random_start:
            self.starting_point = random.choice(range(int(len(self.dates) * 0.5)))
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": []
        }
        stock_info = self.get_date_vector(self.date_index)
        n = int(len(stock_info) / len(self.daily_information_cols))
        stock_info_close = [stock_info[i*len(self.daily_information_cols)+self.close_index] for i in range(n)]
        b, h = self.state_init_func.init_state(
            self.config_amount, stock_info_close
        )
        self.initial_amount = b[0] + sum([h[i] * stock_info_close[i] for i in range(len(stock_info_close))])
        init_state = np.array(
            b 
            + h
            + stock_info
        )
        self.max_total_assets = self.initial_amount
        self.state_memory.append(init_state)
        return init_state

    def log_step(
        self, reason: str, terminal_reward: float=None
        ) -> None:
        """打印"""
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        assets = self.account_information["total_assets"][-1]
        tmp_retreat_ptc = assets / self.max_total_assets - 1
        retreat_pct = tmp_retreat_ptc if assets < self.max_total_assets else 0
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount

        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(assets))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{retreat_pct*100:0.2f}%"
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def return_terminal(
        self, reason: str = "Last Date", reward: int = 0
    ) -> Tuple[List, float, bool, dict]:
        """terminal 的时候执行的操作"""
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        return state, reward, True, {}

    def log_header(self) -> None:
        """Log 的列名"""
        if not self.printed_header:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
            # 0, 1, 2, ... 是序号
            # 4, 4, 15, ... 是占位格的大小
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD",
                    "GAINLOSS_PCT",
                    "RETREAT_PROPORTION"
                )
            )
            self.printed_header = True

    def get_reward(self) -> float:
        """
        获取奖励值=累计收益率 - 当前回撤率作为奖励值
        在股票交易中，我们可以把智能体取得的累计收益率作为奖励值，这非常合理。
        但为了让收益更加稳健，我们还可以在奖励值中加入当前回撤率，作为负的奖励值，也可以说是惩罚值。
        """
        if self.current_step == 0:
            return 0
        else:
            assets = self.account_information["total_assets"][-1]
            retreat = 0
            if assets >= self.max_total_assets:
                self.max_total_assets = assets
            else:
                retreat = assets / self.max_total_assets - 1
            reward = assets / self.initial_amount - 1
            reward += self.alpha * retreat
            return reward

    def get_transactions(self, actions: np.ndarray) -> np.ndarray:
        """获取实际交易的股数"""
        actions = actions * self.hmax

        # 收盘价为 0 的不进行交易
        actions = np.where(self.closings > 0, actions, 0)

        # 去除被除数为 0 的警告
        out = np.zeros_like(actions)
        zero_or_not = self.closings != 0
        actions = np.divide(actions, self.closings, out=out, where = zero_or_not)
        if self.normalize_buy_sell:
            actions = np.sign(actions) * (np.abs(actions) // self.assets_alot) * self.assets_alot
        # 不能卖的比持仓的多
        actions = np.maximum(actions, -np.array(self.holdings))

        # 将 -0 的值全部置为 0
        actions[actions == -0] = 0

        return actions

    def get_spend_and_rest_money(self, transactions):
        sells = -np.clip(transactions, -np.inf, 0)
        proceeds = np.dot(sells, self.closings)
        costs = proceeds * self.sell_cost_pct + self.fixed_fee
        coh = self.cash_on_hand + proceeds # 计算现金的数量

        buys = np.clip(transactions, 0, np.inf)
        spend = np.dot(buys, self.closings)
        costs += spend * self.buy_cost_pct + self.fixed_fee
        return spend, costs, coh

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List, float, bool, dict]:
        self.sum_trades += np.sum(np.abs(actions))
        self.log_header()
        if(self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        if self.date_index == len(self.dates):
            return self.return_terminal(reward=self.get_reward())
        else:
            begin_cash = self.cash_on_hand
            assert min(self.holdings) >= 0
            assert_value = np.dot(self.holdings, self.closings)
            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(assert_value)
            self.account_information["total_assets"].append(begin_cash + assert_value)
            reward = self.get_reward()
            self.account_information["reward"].append(reward)
            self.actions_memory.append(actions)

            transactions = self.get_transactions(actions)
            spend, costs, coh = self.get_spend_and_rest_money(transactions)

            if (spend + costs) > coh: # 如果买不起
                if self.patient:
#                     self.log_step(reason="CASH SHORTAGE")
                    transactions = np.where(transactions > 0, 0, transactions)
                    spend = 0
                    costs = 0
                else:
                    return self.return_terminal(
                        reason="CASH SHORTAGE", reward=self.get_reward()
                    )
            self.transaction_memory.append(transactions)
            assert (spend + costs) <= coh
            coh = coh - spend - costs
            holdings_updated = self.holdings + transactions
            if self.date_index+1 == len(self.dates):
                return self.return_terminal(reward=self.get_reward())
            self.date_index += 1
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            return state, reward, False, {}

    def get_sb_env(self) -> Tuple[Any, Any]:
        def get_self():
            return deepcopy(self)
        
        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(
        self, n: int = 10
    ) -> Tuple[Any, Any]:
        def get_self():
            return deepcopy(self)
        
        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs
    
    def save_asset_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]):
            ]
            return pd.DataFrame(self.account_information)
    
    def save_action_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]):],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory
                }
            )