#!/bin/bash

function run_traders {
    mkdir -p ./logs
    python -u ./trader.py -m 'a2c' >./logs/A2C_trade.log 2>&1 &
    local pid_a2c=$!
    python -u ./trader.py -m 'ppo' >./logs/PPO_trade.log 2>&1 &
    local pid_ppo=$!
    python -u ./trader.py -m 'td3' >./logs/TD3_trade.log 2>&1 &
    local pid_td3=$!
    python -u ./trader.py -m 'ddpg' >./logs/DDPG_trade.log 2>&1 &
    local pid_ddpg=$!
    python -u ./trader.py -m 'sac' >./logs/SAC_trade.log 2>&1 &
    local pid_sac=$!
    wait $pid_a2c $pid_ppo $pid_td3 $pid_ddpg $pid_sac
}

run_traders