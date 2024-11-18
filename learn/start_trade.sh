python -u ./trader.py -m 'a2c' >./logs/A2C_trade.log 2>&1 &
python -u ./trader.py -m 'ppo' >./logs/PPO_trade.log 2>&1 &
python -u ./trader.py -m 'td3' >./logs/TD3_trade.log 2>&1 &
python -u ./trader.py -m 'ddpg' >./logs/DDPG_trade.log 2>&1 &
python -u ./trader.py -m 'sac' >./logs/SAC_trade.log 2>&1 &