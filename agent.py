#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

import argparse
import copy

import numpy as np
np.random.seed(0)

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import serializers


class QNet(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(QNet, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def value(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

    def __call__(self, s_data, a_data, y_data):
        self.loss = None

        s = chainer.Variable(self.xp.asarray(s_data))
        Q = self.value(s)

        Q_data = copy.deepcopy(Q.data)

        if type(Q_data).__module__ != np.__name__:
            Q_data = self.xp.asnumpy(Q_data)

        t_data = copy.deepcopy(Q_data)
        for i in range(len(y_data)):
            t_data[i, a_data[i]] = y_data[i]

        t = chainer.Variable(self.xp.asarray(t_data))
        self.loss = F.mean_squared_error(Q, t)

        return self.loss


# エージェントクラス
class MarubatsuAgent: #(Agent):

    # エージェントの初期化
    # 学習の内容を定義する
    def __init__(self, n_actions):

        self.name = os.path.splitext(os.path.basename(__file__))[0]

        # 学習のInputサイズ
        self.dim = n_actions
        self.bdim = self.dim * 2

        # 学習を開始させるステップ数
        self.learn_start = 5 * 10**3

        # 保持するデータ数
        self.capacity = 1 * 10**4

        # eps = ランダムに○を決定する確率
        self.eps_start = 0.5
        self.eps_end = 0.001

        # 学習時にさかのぼるAction数
        self.n_frames = 3

        # 一度の学習で使用するデータサイズ
        self.batch_size = 32

        self.update_freq = 1 * 10**4

        self.r_win = 1.0
        self.r_draw = -0.5
        self.r_lose = -1.0

        # param
        self.agent_reset()
        # model
        self.agent_init()

    def agent_reset(self):

        self.eps = self.eps_start

        self.replay_mem = []
        self.last_state = None
        self.last_action = None
        self.reward = None
        self.state1 = np.zeros((1, self.n_frames, self.bdim)).astype(np.float32)
        self.state2 = np.zeros((1, self.n_frames, self.bdim)).astype(np.float32)

        self.step_counter = 0

    # ゲーム情報の初期化
    def agent_init(self):

        #２体のエージェントを内包している(self.Q, self.oppQ)ので、
        # 現在の学習対象がどちらのエージェントか示すフラグ
        self.witch_agent = 1
        self.enemy_agent = 2

        # 自分
        self.gamma =  0.99 #DISCOUNTFACTOR
        self.Q = QNet(self.bdim*self.n_frames, 200, self.dim)

        self.xp = np 

        # 自分の現状を保持する変数
        self.targetQ = copy.deepcopy(self.Q)
        # 敵
        self.oppQ = copy.deepcopy(self.Q)

        # 損失関数
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95,
                                                  momentum=0.0)
        self.optimizer.setup(self.Q)

    def agent_switch(self):

        self.save_model()

        if self.witch_agent == 1:
            self.witch_agent = 2
            self.enemy_agent = 1
        else:
            self.witch_agent = 1
            self.enemy_agent = 2

        self.load_model()
        self.targetQ = copy.deepcopy(self.Q)

        self.agent_reset()



    # environment.py env_startの次に呼び出される。
    # エージェントの手を決定し、返す
    def agent_step(self, reward, map, target):
        # ステップを1増加
        self.step_counter += 1

        # observationを[0-2]の9ユニットから[0-1]の18ユニットに変換する
        self.update_state(self.witch_agent, map)

        self.update_targetQ()

        # ○の場所を決定
        int_action = self.select_int_action(target)

        if self.step_counter > 1:

            self.reward = reward

            # データを保存 (状態、アクション、報酬、結果)
            self.store_transition(terminal=False)

            # 学習実行
            if self.step_counter > self.learn_start:
                self.replay_experience()

        # state = 盤の状態 と action = ○を打つ場所 を退避する
        self.last_state = copy.deepcopy(self.state1)
        self.last_action = copy.deepcopy(int_action)

        # ○の位置をエージェントへ渡す
        return int_action

    # ゲームが終了した時点で呼ばれる
    def agent_end(self, reward):
        # 環境から受け取った報酬
        self.reward = reward

        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(terminal=True)

        # 学習実行
        if self.step_counter > self.learn_start:
            self.replay_experience()

    def update_state(self, flg_player, map=None):
        if map is None:
            frame = np.zeros(1, 1, self.bdim).astype(np.float32)
        else:
            observation_binArray = []

            for int_observation in map:
                bin_observation = '{0:02b}'.format(int_observation)
                observation_binArray.append(int(bin_observation[0]))
                observation_binArray.append(int(bin_observation[1]))

            frame = (np.asarray(observation_binArray).astype(np.float32)
                                                     .reshape(1, 1, -1))
        if flg_player == self.witch_agent:
            self.state1 = np.hstack((self.state1[:, 1:], frame))
        else:
            self.state2 = np.hstack((self.state2[:, 1:], frame))


    def update_eps(self, a):
        # eps を更新する。epsはランダムに○を打つ確率
        if self.step_counter > self.learn_start:
            t = self.eps * a
            self.eps = min(self.eps_start, t)
            self.eps = max(self.eps_end, self.eps)

    def update_targetQ(self):
        if self.step_counter % self.update_freq == 0:
            self.targetQ = copy.deepcopy(self.Q)

    def select_int_action(self, free):

        s = chainer.Variable(self.xp.asarray(self.state1))
        Q = self.Q.value(s)

        # Follow the epsilon greedy strategy
        if np.random.rand() < self.eps:
            int_action = np.random.choice(free)
        else:
            Qdata = Q.data[0]

            for i in np.argsort(-Qdata):
                if i in free:
                    int_action = i
                    break

        return int_action


    def store_transition(self, terminal=False):
        if len(self.replay_mem) < self.capacity:
            self.replay_mem.append(
                (self.last_state, self.last_action, self.reward,
                 self.state1, terminal))
        else:
            self.replay_mem = (self.replay_mem[1:] +
                [(self.last_state, self.last_action, self.reward, self.state1,
                  terminal)])

    def replay_experience(self):
        indices = np.random.randint(0, len(self.replay_mem), self.batch_size)
        samples = np.asarray(self.replay_mem)[indices]

        s, a, r, s2, t = [], [], [], [], []

        for sample in samples:
            s.append(sample[0])
            a.append(sample[1])
            r.append(sample[2])
            s2.append(sample[3])
            t.append(sample[4])

        s = np.asarray(s).astype(np.float32)
        a = np.asarray(a).astype(np.int32)
        r = np.asarray(r).astype(np.float32)
        s2 = np.asarray(s2).astype(np.float32)
        t = np.asarray(t).astype(np.float32)

        s2 = chainer.Variable(self.xp.asarray(s2))
        Q = self.targetQ.value(s2)
        Q_data = Q.data

        if type(Q_data).__module__ == np.__name__:
            max_Q_data = np.max(Q_data, axis=1)
        else:
            max_Q_data = np.max(self.xp.asnumpy(Q_data).astype(np.float32), axis=1)

        t = np.sign(r) + (1 - t)*self.gamma*max_Q_data

        self.optimizer.update(self.Q, s, a, t)


    def select_opp_action(self, map, free):
        #敵の行動を決定する。

        self.update_state(self.enemy_agent, map)

        s = chainer.Variable(self.xp.asarray(self.state2))
        Q = self.oppQ.value(s)

        # Follow the epsilon greedy strategy
        Qdata = Q.data[0]

        for i in np.argsort(-Qdata):
            if i in free:
                int_action = i
                break

        return int_action

    def load_model(self):

        # 自分
        save_model_name = self.name + str(self.witch_agent) + '.model'
        save_state_name = self.name + str(self.witch_agent) + '.state'
        if os.path.exists(save_model_name):
            serializers.load_hdf5(save_model_name, self.Q)
        if os.path.exists(save_state_name):
            serializers.load_hdf5(save_state_name, self.optimizer)

        # 敵
        save_model_name = self.name + str(self.enemy_agent) + '.model'
        if os.path.exists(save_model_name):
            serializers.load_hdf5(save_model_name, self.oppQ)


    def save_model(self):

        save_model_name = self.name + str(self.witch_agent) + '.model'
        save_state_name = self.name + str(self.witch_agent) + '.state'

        #modelとoptimizerの二つ保存する必要あり
        serializers.save_hdf5(save_model_name, self.Q)
        serializers.save_hdf5(save_state_name, self.optimizer)


