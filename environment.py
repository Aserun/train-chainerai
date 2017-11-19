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
        self.n_rows = 8
        self.n_cols = self.n_rows

        # 可能な手
        self.enable_actions = np.arange(self.n_rows*self.n_cols)
        self.n_actions = len(self.enable_actions)

        self.history = []


    # Episodeの開始
    def env_start(self):
        # 盤面を初期化
        self.map = [0] * self.n_rows * self.n_cols
        self.map[27] = self.flg_agent
        self.map[28] = self.flg_env
        self.map[35] = self.flg_env
        self.map[36] = self.flg_agent
        
        # 盤の状態を保持し、最後に確認するためのリスト
        self.history = []

        current_map = ''
        for i in range(0, len(self.map), self.n_cols):
            current_map += ' '.join(map(str, self.map[i:i+self.n_cols])) + '\n'
        self.history.append(current_map)

        # 盤の状態をエージェントに渡す
        return self.map



    def env_step(self, action):

        reward = 0.0
        terminal = False

        if action >= 0: 
            # エージェントから受け取った○を打つ場所
            int_action_agent = action

            # 盤に○を打ち、空白の個所を取得する
            self.put_piece(int_action_agent, self.flg_agent)
            terminal = self.isEnd()

            # ○を打った後の勝敗を確認する
            if terminal:
                if self.winner() == self.flg_agent:
                    reward = self.r_win
                elif  self.winner() == self.flg_env:
                    reward = self.r_lose
                else:
                    reward = self.r_draw

        # 勝敗がつかなければ、×を打つ位置を決める
        if not terminal:

            # ×を打つ位置がなければパス
            free = self.action_target(self.flg_env)
            if len(free) == 0:
                pass
            else:
                # ×の位置をランダムに決定する
                int_action_env = np.random.choice(free)

                # 盤に×を打つ
                self.put_piece(int_action_env, self.flg_env)

                # ×を打った後の勝敗を確認する
                terminal = self.isEnd()
                if terminal:
                    if self.winner() == self.flg_agent:
                        reward = self.r_win
                    elif  self.winner() == self.flg_env:
                        reward = self.r_lose
                    else:
                        reward = self.r_draw


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


    # 置ける位置のリストを返す
    def action_target(self, flg_player):
        result = []
        free = [i for i, v in enumerate(self.map) if v == self.flg_free]
        for i in free:
            if self.put_piece(i, flg_player, False) > 0:
                #ここ置ける!!
                result.append(i)

        return result


    def put_piece(self, action, flg_player, puton=True):
        #自駒flg_player(1 or 2)を位置action(0～63)に置く関数 
        int_action = action
        
        if self.map[int_action] != self.flg_free:
            return -1

        # ---------------------------------------------------------
        #   縦横斜めの8通りは、1次元データなので、
        #   現在位置から[-9, -8, -7, -1, 1, 7, 8, 9] 
        #   ずれた方向を見ます。
        #   これは、[-1, 0, 1]と[-8, 0, 8]の組合せで調べます
        #   (0と0のペアは除く)。
        #
        t, x, y, l = 0, int_action%8, int_action//8, []
        for di, fi in zip([-1, 0, 1], [x, 7, 7-x]):
            for dj, fj in zip([-8, 0, 8], [y, 7, 7-y]):                
                if not di == dj == 0:
                    b, j, k, m, n =[], 0, 0, [], 0                    
                    # a:対象位置のid リス
                    a = self.enable_actions[int_action+di+dj::di+dj][:min(fi, fj)]
                    # b:対象位置の駒id リスト
                    for i in a: 
                        if self.map[i] == self.flg_free: #空白
                            break  
                        elif self.map[i] == flg_player: #自駒があればその間の相手の駒を取れる
                            # 取れる数を確定する
                            n = k
                            # ひっくり返す駒を確定する  
                            l += m
                            # その方向の探査終了 
                            break
                        else: #相手の駒
                            k += 1
                            # ひっくり返す位置をストックする 
                            m.append(i) 

                    t += n 
                    
        if t == 0:
            return 0
            
        if puton:
            # ひっくり返す石を登録する
            for i in l:
                self.map[i] = flg_player
            # 今置いた石を追加する 
            self.map[int_action] = flg_player
            
        return t


    def isEnd(self):
        #双方置けなくなったらゲーム終了
        e1 = self.action_target(self.flg_agent)        
        e2 = self.action_target(self.flg_env)  
        if len(e1) == 0 and len(e2) == 0:
            return True

        #空白マスがなくなったらゲーム終了
        free = [i for i, v in enumerate(self.map) if v == self.flg_free]
        if len(free) == 0:
            return True
                  
        return False


    def winner(self):
        # 勝ったほうを返す 
        player1_score = self.map.count(self.flg_agent)
        player2_score = self.map.count(self.flg_env)
            
        if player1_score == player2_score:
            return 0 # 引き分け
        elif player1_score > player2_score:
            return self.flg_agent # agentの勝ち
        elif player1_score < player2_score:
            return self.flg_env   # envの勝ち
       