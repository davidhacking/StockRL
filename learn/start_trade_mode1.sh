python -u ./trader.py -m 'a2c' -d 1 >./logs/A2C_mode1_trade.log 2>&1 &
python -u ./trader.py -m 'ppo' -d 1 >./logs/PPO_mode1_trade.log 2>&1 &
python -u ./trader.py -m 'td3' -d 1 >./logs/TD3_mode1_trade.log 2>&1 &
python -u ./trader.py -m 'ddpg' -d 1 >./logs/DDPG_mode1_trade.log 2>&1 &
python -u ./trader.py -m 'sac' -d 1 >./logs/SAC_mode1_trade.log 2>&1 &