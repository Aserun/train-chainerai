#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(0)

from environment import MarubatsuEnvironment
from agent import MarubatsuAgent

if __name__ == "__main__":

    # environment, agent
    env = MarubatsuEnvironment()
    agent = MarubatsuAgent() 
    agent.load_model()  # if allradey model file exsists then load the file

    with open('result.txt', 'a') as f:
        f.writelines('START: ' + str(datetime.datetime.now()) + '\n')

    which_episode = 0
    total_win = 0
    total_draw = 0
    total_lose = 0

    ns_epoch = []
    pcts_win = []
    pcts_win_or_draw = []
    pcts_lose = []

    # 初期化
    # REWARDS = 報酬 (-1.0 ~ 1.0)   ex) 勝 1, 引分 -0.5, 負 -1
    r_win = 1.0
    r_draw = -0.5
    r_lose = -1.0

    # 50,000回実行
    for _ in range(0, 5 * 10**4):

        # 強化学習の主処理
        which_episode += 1

        # ゲーム1回 開始
        map = env.env_start()
        total_steps = 1
        free = env.action_target(env.flg_agent)

        # 勝負がつくまでのステップ数と報酬を取得
        action = agent.agent_start(map, free)
        map, reward, terminal = env.env_step(action)
        while not terminal:
            free = env.action_target(env.flg_agent)
            if len(free)> 0:
                action = agent.agent_step(reward, map, free)
                map, reward, terminal = env.env_step(action)
            else:
                # ○を置く場所がない場合
                map, reward, terminal = env.env_step(-1)

            total_steps += 1

        agent.agent_end(reward)

        # 今回の結果を表示
        if reward == r_win:
            total_win += 1
        elif reward == r_draw:
            total_draw += 1
        elif reward == r_lose:
            total_lose += 1

        print("Episode "+str(which_episode)+"\t "+str(total_steps)+ " steps \t" + str(reward) + " total reward\t " + str(terminal) + " natural end")

        # 100回毎に勝敗を集計
        record_interval = 100

        if which_episode % record_interval == 0:
            line = 'Episode: {}, {} wins, {} draws, {} loses'.format(which_episode, total_win, total_draw, total_lose)
            print('---------------------------------------------------------------')
            print(line)
            print('---------------------------------------------------------------')

            # 集計結果をファイルに出力
            with open('result.txt', 'a') as f:
                f.writelines(line + '\n')

            ns_epoch.append(which_episode)
            pcts_win.append(float(total_win) / record_interval * 100)
            pcts_win_or_draw.append(float(total_win + total_draw) / record_interval * 100)
            pcts_lose.append(float(total_win) / record_interval * 100)

            total_win = 0
            total_draw = 0
            total_lose = 0

    # 学習結果をグラフで出力
    plt.plot(np.asarray(ns_epoch), np.asarray(pcts_win_or_draw))
    plt.xlabel('episode')
    plt.ylabel('percentage')
    plt.title('Average win or draw rate')
    plt.grid(True)
    plt.savefig("percentages.png")

    # 終了時刻をファイルに書出し
    with open('result.txt', 'a') as f:
        f.writelines('END: ' + str(datetime.datetime.now()) + '\n')
