from __future__ import print_function
import sys
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(0)

from reversi import ReversiEnvironment
from agent import MarubatsuAgent


if __name__ == "__main__":

    # environment, agent
    # env = MarubatsuEnvironment()
    env = ReversiEnvironment()
    agent = MarubatsuAgent(env.n_actions) 
    agent.load_model()  # if allradey model file exsists then load the file

    r_win = 1.0
    r_draw = -0.5
    r_lose = -1.0

    # 永久に学習を続ける
    while True:

        # パラメータの初期化
        which_episode = 0
        total_win = 0
        total_draw = 0
        total_lose = 0

        WinningPercentage = 0
        pcts_epoch = []
        pcts_win_or_draw = []

        # 開始時刻をファイルに書出し
        with open('result.txt', 'a') as f:
            f.writelines('Agent '   + str(agent.witch_agent) +','
                         +'Episode '+ str(which_episode)     +','
                         +'START: ' + str(datetime.datetime.now()) + '\n')


        # 強化学習の主処理
        # 勝率 が 9割 を超えるまで学習する
        while WinningPercentage < 90:

            which_episode += 1

            steps = 0
            reward = 0

            # ゲーム1回 開始
            state = env.env_start()
            winner = 0
            terminal = False

            # agent1 がトレーニングエージェントの時は agent2 が 初手 を 打つ
            if agent.witch_agent == env.flg_player1:
                target = env.action_target(env.flg_player2)
                action = agent.select_opp_action(state, target)
                state, winner, terminal = env.env_step(action, env.flg_player2)


            # 勝負がつくまでのステップ数と報酬を取得
            while True:

                # トレーニングエージェントの手番
                target = env.action_target(agent.witch_agent)
                if len(target) > 0:
                    action = agent.agent_step(reward, state, target)
                    state, winner, terminal = env.env_step(action, agent.witch_agent)
                    if terminal:
                        if winner == agent.witch_agent:
                            reward = r_win
                        elif winner == agent.enemy_agent:
                            reward = r_lose
                        else:
                            reward = r_draw
                        break
    
                # 対戦相手エージェントの手番
                target = env.action_target(agent.enemy_agent)
                if len(target) > 0:
                    action = agent.select_opp_action(state, target)
                    state, winner, terminal = env.env_step(action, agent.enemy_agent)
                    if terminal:
                        if winner == agent.witch_agent:
                            reward = r_win
                        elif winner == agent.enemy_agent:
                            reward = r_lose
                        else:
                            reward = r_draw
                        break

                steps += 1

            # ゲーム終了
            agent.agent_end(reward)

            # 勝敗に応じて トレーニングエージェントの εを更新する。
            if reward == r_win:
                agent.update_eps(0.8)
                total_win += 1
            elif reward == r_draw:
                total_draw += 1
            elif reward == r_lose:
                agent.update_eps(1.05)
                total_lose += 1

            # 今回の結果を表示
            print("Agent "+str(agent.witch_agent)+"\t "+"Episode "+str(which_episode)+"\t "+str(steps)+ " steps \t" + str(reward) + " total reward\t " + str(agent.eps) + " eps")

            # 100回毎に勝敗を集計
            record_interval = 100

            if which_episode % record_interval == 0:
                line = 'Agent: {}, Episode: {}, {} wins, {} draws, {} loses'.format(agent.witch_agent, which_episode, total_win, total_draw, total_lose)
                print('---------------------------------------------------------------')
                print(line)
                print('---------------------------------------------------------------')

                WinningPercentage = float(total_win + total_draw) / record_interval * 100

                pcts_epoch.append(which_episode)
                pcts_win_or_draw.append(WinningPercentage)

                total_win = 0
                total_draw = 0
                total_lose = 0


                # 学習結果をグラフで出力
                plt.plot(np.asarray(pcts_epoch), np.asarray(pcts_win_or_draw))
                plt.xlabel('episode')
                plt.ylabel('percentage')
                plt.title('Agent '+str(agent.witch_agent)+'\t '+'Average win or draw rate')
                plt.grid(True)
                plt.savefig("percentages.png")


        plt.figure()

        # 終了時刻をファイルに書出し
        with open('result.txt', 'a') as f:
            f.writelines('Agent '   + str(agent.witch_agent) +','
                         +'Episode '+ str(which_episode)     +','
                         +'END: '   + str(datetime.datetime.now()) + '\n')

        # 学習対象エージェントを入れ替える
        agent.agent_switch()

