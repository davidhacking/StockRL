python -u ./trainer.py -m 'a2c' -tts 200000 -d 1 >./logs/A2C_mode1.log 2>&1 &
python -u ./trainer.py -m 'ppo' -tts 200000 -d 1 >./logs/PPO_mode1.log 2>&1 &
python -u ./trainer.py -m 'td3' -tts 200000 -d 1 >./logs/TD3_mode1.log 2>&1 &
python -u ./trainer.py -m 'ddpg' -tts 200000 -d 1 >./logs/DDPG_mode1.log 2>&1 &
python -u ./trainer.py -m 'sac' -tts 200000 -d 1 >./logs/SAC_mode1.log 2>&1 &