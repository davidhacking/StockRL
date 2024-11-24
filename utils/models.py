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
from utils.preprocessors import split_data
from utils.env import StockLearningEnv
from abc import ABC, abstractmethod
from copy import deepcopy


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

UserStockAccountFactory = {
    "LocalUserStockAccount": LocalUserStockAccount,
}

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
MODEL_KWARGS = {x: config.__dict__["{}_PARAMS".format(x.upper())] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise
}

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

        len_environment = len(environment.df.index.unique())
        for i in range(len_environment):
            predict_action, _states = model.predict(test_obs)
            if user_stock_account is not None:
                action = deepcopy(predict_action)
                action = test_env.env_method("get_transactions", action)[0]
                res = test_env.env_method("get_spend_and_rest_money", action)[0]
                spend, costs, coh = res
                closings = test_env.env_method(method_name="get_closings")[0]
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
    from pull_data import Pull_data
    from preprocessors import FeatureEngineer, split_data
    from utils import config
    import time

    # 拉取数据
    df = Pull_data(config.SSE_50[:2], save_data=False).pull_data()
    df = FeatureEngineer().preprocess_data(df)
    df = split_data(df, '2009-01-01','2019-01-01')
    print(df.head())

    # 处理超参数
    stock_dimension = len(df.tic.unique()) # 2
    state_space = 1 + 2*stock_dimension + \
        len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension # 23
    print("stock_dimension: {}, state_space: {}".format(stock_dimension, state_space))
    env_kwargs = {
        "stock_dim": stock_dimension, 
        "hmax": 100, 
        "initial_amount": 1e6, 
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "state_space": state_space, 
        "action_space": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST
    }

    # 测试环境
    e_train_gym = StockLearningEnv(df = df, **env_kwargs)

    ### 测试一次
    # observation = e_train_gym.reset()
    # print("reset_observation: ", observation)
    # action = e_train_gym.action_space.sample()
    # print("action: ", action)
    # observation_later, reward, done, _ = e_train_gym.step(action)
    # print("observation_later: ", observation_later)
    # print("reward: {}, done: {}".format(reward, done))

    ### 多次测试
    observation = e_train_gym.reset()       #初始化环境，observation为环境状态
    count = 0
    for t in range(10):
        action = e_train_gym.action_space.sample()  #随机采样动作
        observation, reward, done, info = e_train_gym.step(action)  #与环境交互，获得下一个state的值
        if done:             
            break
        count+=1
        time.sleep(0.2)      #每次等待 0.2s
    print("observation: ", observation)
    print("reward: {}, done: {}".format(reward, done))

    # 测试 model
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    agent = DRL_Agent(env= env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1"
    }
    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    trained_sac = agent.train_model(
        model=model_sac,
        tb_log_name='sac', 
        total_timesteps= 50000
    )