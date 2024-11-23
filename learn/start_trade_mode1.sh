#!/bin/bash

function run_traders_mode1 {
    mkdir -p ./logs

    python -u ./trader.py -m 'a2c' -d 1 >./logs/A2C_mode1_trade.log 2>&1 &
    local pid_a2c=$!
    python -u ./trader.py -m 'ppo' -d 1 >./logs/PPO_mode1_trade.log 2>&1 &
    local pid_ppo=$!
    python -u ./trader.py -m 'td3' -d 1 >./logs/TD3_mode1_trade.log 2>&1 &
    local pid_td3=$!
    python -u ./trader.py -m 'ddpg' -d 1 >./logs/DDPG_mode1_trade.log 2>&1 &
    local pid_ddpg=$!
    python -u ./trader.py -m 'sac' -d 1 >./logs/SAC_mode1_trade.log 2>&1 &
    local pid_sac=$!

    wait $pid_a2c $pid_ppo $pid_td3 $pid_ddpg $pid_sac
}

run_traders_mode1