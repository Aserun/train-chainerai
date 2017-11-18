#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
np.random.seed(0)

class MarubatsuEnvironment():

    # 盤の状態 [空白, ○, ×]
    flg_free = 0
    flg_agent = 1
    flg_env = 2

    # 報酬
    r_win = 1.0
    r_draw = -0.5
    r_lose = -1.0

    # 敵プレイヤーが正常に打つ確率
    opp = 0.75

    def __init__(self):
        self.n_rows = 3
        self.n_cols = self.n_rows

        # 3つ並んだら勝敗が決まるライン
        # 3×3の盤なので、[0,1,2],[3,4,5].....
        lines = []

        for i in range(self.n_rows):
            horizontal = []
            for j in range(self.n_cols):
                horizontal.append(i*self.n_cols + j)
            lines.append(horizontal)

        for i in range(self.n_cols):
            vertical = []
            for j in range(self.n_rows):
                vertical.append(i + j*self.n_cols)
            lines.append(vertical)

        if self.n_rows == self.n_cols:
            n = self.n_rows
            tl_br, tr_bl = [], []
            for i in range(n):
                tl_br.append(i*n + i)
                tr_bl.append(i*n + (n - 1) - i)
            lines.append(tl_br)
            lines.append(tr_bl)

        self.lines = lines

        self.history = []


    # Episodeの開始
    def env_start(self):
        # 盤面を初期化
        self.map = [0] * self.n_rows * self.n_cols

        # 盤の状態を保持し、最後に確認するためのリスト
        self.history = []

        current_map = ''
        for i in range(0, len(self.map), self.n_cols):
            current_map += ' '.join(map(str, self.map[i:i+self.n_cols])) + '\n'
        self.history.append(current_map)

        # 盤の状態をエージェントに渡す
        return self.map

    def env_step(self, action):
        # エージェントから受け取った○を打つ場所
        int_action_agent = action

        # 盤に○を打ち、空白の個所を取得する
        self.map[int_action_agent] = self.flg_agent
        free = [i for i, v in enumerate(self.map) if v == self.flg_free]
        n_free = len(free)

        reward = 0.0
        terminal = False

        # ○を打った後の勝敗を確認する
        for line in self.lines:
            state = np.array(self.map)[line]

            point = sum(state == self.flg_agent)

            if point == self.n_rows:
                reward = self.r_win
                terminal = True
                break

            point = sum(state == self.flg_env)

            if point == self.n_rows:
                reward = self.r_lose
                terminal = True
                break

        # 勝敗がつかなければ、×を打つ位置を決める
        if not terminal:
            # 空白がなければ引き分け
            if n_free == 0:
                reward = self.r_draw
                terminal = True
            else:
                int_action_env = None

                # 空白が1個所ならばそこに×を打つ
                if n_free == 1:
                    int_action_env = free[0]
                    terminal = True
                else:
                    # ×の位置を決定する 75%
                    if np.random.rand() < self.opp:
                        for line in self.lines:
                            state = np.array(self.map)[line]
                            point = sum(state == self.flg_env)

                            if point == self.n_rows - 1:
                                index = np.where(state == self.flg_free)[0]

                                if len(index) != 0:
                                    int_action_env = line[index[0]]
                                    break

                        if int_action_env is None:
                            for line in self.lines:
                                state = np.array(self.map)[line]
                                point = sum(state == self.flg_agent)

                                if point == self.n_rows - 1:
                                    index = np.where(state == self.flg_free)[0]
                                    if len(index) != 0:
                                        int_action_env = line[index[0]]
                                        break

                    # ×の位置をランダムに決定する 25%
                    if int_action_env is None:
                        int_action_env = free[np.random.randint(n_free)]

                # 盤に×を打つ
                self.map[int_action_env] = self.flg_env

                free = [i for i, v in enumerate(self.map) if v == self.flg_free]
                n_free = len(free)

                # ×を打った後の勝敗を確認する
                for line in self.lines:
                    state = np.array(self.map)[line]

                    point = sum(state == self.flg_agent)

                    if point == self.n_rows:
                        reward = self.r_win
                        terminal = True
                        break

                    point = sum(state == self.flg_env)

                    if point == self.n_rows:
                        reward = self.r_lose
                        terminal = True
                        break

                if not terminal and n_free == 0:
                    reward = self.r_draw
                    terminal = True

        # 盤の状態と報酬、決着がついたかどうか をまとめて エージェントにおくる。

        current_map = ''
        for i in range(0, len(self.map), self.n_cols):
            current_map += ' '.join(map(str, self.map[i:i+self.n_cols])) + '\n'
        self.history.append(current_map)

        if reward == -1:
            f = open('history.txt', 'a')
            history = '\n'.join(self.history)
            f.writelines('# START\n' + history + '# END\n\n')
            f.close()

        # 決着がついた場合は agentのagent_end
        # 決着がついていない場合は agentのagent_step に続く
        return self.map, reward, terminal

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
