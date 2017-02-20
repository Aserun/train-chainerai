#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

import numpy as np
np.random.seed(0)


class ReversiEnvironment:

    # ?Ղ̏??? [????, ??, ?~]
    flg_free = 0
    flg_player1 = 1
    flg_player2 = 2

    def __init__(self):
        self.name = os.path.splitext(os.path.basename(__file__))[0]
 
        self.n_rows = 8
        self.n_cols = self.n_rows

        self.enable_actions = np.arange(self.n_rows*self.n_cols)
        self.n_actions = len(self.enable_actions)


    def env_start(self):
        self.map = [0] * self.n_rows * self.n_cols
        self.map[27] = self.flg_player1
        self.map[28] = self.flg_player2
        self.map[35] = self.flg_player2
        self.map[36] = self.flg_player1

        return self.map


    def env_step(self, action, flg_player):

        int_action = action

        self.put_piece(int_action, flg_player)

        terminal = self.isEnd()
        winner = 0.0

        if terminal:
            winner = self.winner()

        return self.map, winner, terminal


    """ 置ける位置のリストを返す """
    def action_target(self, flg_player):
        result = []
        free = [i for i, v in enumerate(self.map) if v == self.flg_free]
        for i in free:
            if self.put_piece(i, flg_player, False) > 0:
                """ ここ置ける!! """
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
        e1 = self.action_target(self.flg_player1)        
        e2 = self.action_target(self.flg_player2)  
        if len(e1) == 0 and len(e2) == 0:
            return True

        #空白マスがなくなったらゲーム終了
        free = [i for i, v in enumerate(self.map) if v == self.flg_free]
        if len(free) == 0:
            return True
                  
        return False

    def winner(self):
        """ 勝ったほうを返す """
        player1_score = self.map.count(self.flg_player1)
        player2_score = self.map.count(self.flg_player2)
            
        if player1_score == player2_score:
            return 0 # 引き分け
        elif player1_score > player2_score:
            return self.flg_player1 # player1の勝ち
        elif player1_score < player2_score:
            return self.flg_player2 # player2の勝ち
       