import codecs
import os
import sys
import pandas as pd
from argparse import ArgumentParser
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append("..")
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils import config
from utils.env import StockLearningEnv
from utils.models import DRL_Agent
from data import Data


class Trainer(object):
    """用来训练的类

    Attributes:
        model_name: 强化学习的算法名称，用来调用指定的算法
        total_timesteps: 总的训练步数
    """

    def __init__(self, model_name = 'a2c' , 
                        total_timesteps = 200000,
                        multi_env = 0,
                        mode = 0) -> None:
        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.multi_env = multi_env
        self.mode = mode
        self.train_dir = "train_file"
        self.data_dir = "data_file"
        if mode == 1:
            self.model_path = os.path.join(self.train_dir, "{}_evaluation.model".format(self.model_name))
        else:
            self.model_path = ""
        self.create_train_dir()
    
    def create_train_dir(self) -> None:
        """创建存储训练结果的文件夹"""
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            print("{} 文件夹创建成功!".format(self.train_dir))
        else:
            print("{} 文件夹已存在!".format(self.train_dir))
    
    def train(self) -> None:
        """开始训练"""
        train_data, trade_data = self.get_data()
        env_train, env_trade = self.get_env(train_data, trade_data)

        agent = DRL_Agent(env = env_train)

        model = agent.get_model(self.model_name,  
                                model_kwargs = config.__dict__["{}_PARAMS".format(self.model_name.upper())], 
                                verbose = 0)
        model.learn(total_timesteps = self.total_timesteps, 
                    eval_env = env_trade, 
                    eval_freq = 500,
                    log_interval = 1, 
                    tb_log_name = 'env_cashpenalty_highlr',
                    n_eval_episodes = 1)
        self.save_model(model)
    
    def get_data(self):
        """获取训练数据集和交易数据集"""
        if self.mode == 0:
            train_data_path = os.path.join(self.data_dir, "train.csv")
            trade_data_path = os.path.join(self.data_dir, "trade.csv")
        elif self.mode == 1:
            train_data_path = config.stock_info_train_csv
            trade_data_path = config.stock_info_test_csv
        if not (os.path.exists(train_data_path) or
                os.path.exists(trade_data_path)):
            print("数据不存在，开始下载")
            Data().pull_data()
        
        train_data = pd.read_csv(train_data_path)
        trade_data = pd.read_csv(trade_data_path)
        print("数据读取成功!")
        
        return train_data, trade_data

    def get_env(self, 
                train_data: pd.DataFrame, 
                trade_data: pd.DataFrame) -> DummyVecEnv:
        """分别返回训练环境和交易环境"""
        e_train_gym = StockLearningEnv(df = train_data,
                                                    random_start = True,
                                                    **config.ENV_PARAMS)
        if self.multi_env <= 0:
            env_train, _ = e_train_gym.get_sb_env()
        else:
            env_train, _ = e_train_gym.get_multiproc_env(self.multi_env)

        e_trade_gym = StockLearningEnv(df = trade_data,
                                                    random_start = False,
                                                    **config.ENV_PARAMS)
        env_trade, _ = e_trade_gym.get_sb_env()

        return env_train, env_trade

    def save_model(self, model) -> None:
        model_path = os.path.join(self.train_dir, "{}.model".format(self.model_name))
        if self.model_path != "":
            model_path = self.model_path
        model.save(model_path)


def start_train():
    parser = ArgumentParser(description="set parameters for train mode")
    parser.add_argument(
        '--model', '-m',
        dest='model',
        default='a2c',
        help='choose the model you want to train',
        metavar="MODEL",
        type=str
    )
    
    parser.add_argument(
        '--mode', '-d',
        dest='mode',
        default=0,
        help='choose the mode you want to train',
        metavar="MODE",
        type=int
    )

    parser.add_argument(
        '--total_timesteps', '-tts',
        dest='total_timesteps',
        default=200000,
        help='set the total_timesteps when you train the model',
        metavar="TOTAL_TIMESTEPS",
        type=int
    )
    
    parser.add_argument(
        '--multi_env', '-me',
        dest='multi_env',
        default=0,
        help='parallel train number',
        metavar="MULTI_ENV",
        type=int
    )

    options = parser.parse_args()
    Trainer(model_name = options.model,
            total_timesteps = options.total_timesteps,
            multi_env=options.multi_env,
            mode=options.mode).train()

if __name__ == "__main__":
    import time
    start = time.time()
    start_train()
    end = time.time()
    delta = end - start
    
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))
    delta = end - start
    total_seconds = int(delta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    delta_str = f"{hours}H{minutes}min{seconds}s"

    print("start=", start_time_str)
    print("end=", end_time_str)
    print("train delta=", delta_str)