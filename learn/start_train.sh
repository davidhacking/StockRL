python -u ./trainer.py -m 'a2c' -tts 200000 >./logs/A2C.log 2>&1 &
python -u ./trainer.py -m 'ppo' -tts 200000 >./logs/PPO.log 2>&1 &
python -u ./trainer.py -m 'td3' -tts 200000 >./logs/TD3.log 2>&1 &
python -u ./trainer.py -m 'ddpg' -tts 200000 >./logs/DDPG.log 2>&1 &
python -u ./trainer.py -m 'sac' -tts 200000 >./logs/SAC.log 2>&1 &