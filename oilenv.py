import torch
import random
import numpy as np
import pandas as pd
import gym
from gym import spaces
from scipy.spatial.distance import cosine  # 修正: cosine関数をインポート

MAX_CAPACITY = 1000  # タンクの最大容量

class OilDeliveryEnv(gym.Env):
    def __init__(self, file_path, inbound_reward_coef, outbound_reward_coef, negative_content_penalty, verbose=True):
        super(OilDeliveryEnv, self).__init__()
        self.file_path = file_path
        self.inbound_reward_coef = inbound_reward_coef
        self.outbound_reward_coef = outbound_reward_coef
        self.negative_content_penalty = negative_content_penalty
        self.verbose = verbose

        # Excelデータから設定を読み込む
        self.load_data()

        # アクションスペースと観測スペースを設定
        num_actions = self.num_tanks * self.num_oil_types + self.num_tanks
        self.action_space = spaces.MultiDiscrete([10] * num_actions)
        
        # 観測空間：タンクの状態 + 計画全体
        observation_shape = (self.num_oil_types * self.num_tanks + len(self.plan_data) * self.num_oil_types,)
        self.observation_space = spaces.Box(low=0, high=MAX_CAPACITY, shape=observation_shape, dtype=np.float32)
        
        self.current_request_index = 0  # current_request_indexの初期化
        self.reset()

    def load_data(self):
        # Excelファイルを読み込む
        data = pd.read_excel(self.file_path, sheet_name=None)
        self.plan_data = data['plan']
        tank_data = data['tank']

        # 油種とタンクの数を取得
        oil_columns = [col for col in tank_data.columns if col.startswith('oil_')]
        self.num_oil_types = len(oil_columns)
        self.num_tanks = tank_data['tank_index'].nunique()

        # タンクの初期状態を読み込み
        self.tank_initial_state = tank_data[oil_columns].fillna(0).values.flatten()

        # 固定された要求を格納するリストを構築
        self.fixed_requests = []
        for idx, row in self.plan_data.iterrows():
            if row['type'] == 'in':
                inbound = tuple(row[oil_columns].fillna(0).astype(int))
                order = [0] * self.num_oil_types  # 搬入の場合、orderはゼロ
                self.fixed_requests.append({'inbound': inbound, 'order': order})
            elif row['type'] == 'out':
                inbound = (0,) * self.num_oil_types  # 搬出の場合、inboundはゼロ
                order = list(row[oil_columns].fillna(0).astype(int))
                self.fixed_requests.append({'inbound': inbound, 'order': order})

    def reset(self):
        # エピソードの開始時にタンクの状態をリセット
        self.state = self.tank_initial_state.copy()  # 初期タンク状態を使用
        self.current_request_index = 0
        current_request = self.fixed_requests[self.current_request_index]
        self.inbound_request = current_request['inbound']
        self.order = np.array(current_request['order'])
        self.num_operations = 0  # 操作回数をリセット
        self.total_reward = 0

        # 計画データをフラットなベクトルに変換して観測に含める
        plan_flat = self.plan_data[[col for col in self.plan_data.columns if col.startswith('oil_')]].fillna(0).values.flatten() / MAX_CAPACITY  # 正規化
        return np.concatenate((self.state, plan_flat))

    def step(self, action):
        # エピソード終了条件
        if self.current_request_index >= len(self.fixed_requests):
            return self.state, 0, True, {"info": "End of plan"}

        current_request = self.fixed_requests[self.current_request_index]

        # 出荷操作の場合はスキップ
        if np.sum(current_request['order']) > 0:
            self.current_request_index += 1
            if self.current_request_index >= len(self.fixed_requests):
                return self.state, 0, True, {"info": "End of plan"}
            # 次の要求を取得
            current_request = self.fixed_requests[self.current_request_index]
            self.inbound_request = current_request['inbound']
            self.order = np.array(current_request['order'])
            return np.concatenate((self.state, self.plan_data[[col for col in self.plan_data.columns if col.startswith('oil_')]].fillna(0).values.flatten() / MAX_CAPACITY)), 0, False, {}

        # 搬入操作の場合
        action_details = []
        inbound_processed = [0] * self.num_oil_types  # 各油種の搬入量を記録
        reward = 0
        deadend = False
        print(current_request["inbound"])

        for tank_index in range(self.num_tanks):
            for oil_index in range(self.num_oil_types):
                action_value = action[tank_index * self.num_oil_types + oil_index]
                quantity_to_add = action_value * (MAX_CAPACITY / 10)
                
                # リクエストに記載のない油種を搬入した場合はエピソードを終了
                if self.inbound_request[oil_index] == 0 and quantity_to_add > 0:
                    reward += -quantity_to_add
                    deadend = True
                    print(f"Attempted to add oil type {oil_index} which is not requested.")
    #                return self.state, reward - 1000, True, {"info": "Invalid oil type"}

                self.state[tank_index * self.num_oil_types + oil_index] += quantity_to_add
                inbound_processed[oil_index] += quantity_to_add
                action_details.append((tank_index, oil_index, quantity_to_add))

                # タンク容量をチェック
                if self.state[tank_index * self.num_oil_types + oil_index] > MAX_CAPACITY:
                    excess_amount = self.state[tank_index * self.num_oil_types + oil_index] - MAX_CAPACITY
                    reward += -excess_amount
                    deadend = True
                    print(f"Tank {tank_index} of oil type {oil_index} overflow: {self.state[tank_index * self.num_oil_types + oil_index]}")
    #                return self.state, reward - 1000, True, {"info": "Tank overflow"}

        # リクエストに記載の油種を記載量以上搬入した場合はエピソードを終了
        for oil_index in range(self.num_oil_types):
            if inbound_processed[oil_index] > self.inbound_request[oil_index]:
                excess_amount = inbound_processed[oil_index] - self.inbound_request[oil_index]
                reward += -excess_amount
                deadend = True
                print(f"Exceeded requested amount for oil type {oil_index}.")
    #            return self.state, reward - 1000, True, {"info": "Exceeded requested amount"}

        self.num_operations += 1
        self.current_request_index += 1

        # コサイン類似度に基づく報酬
        if np.linalg.norm(self.inbound_request) > 0 and np.linalg.norm(inbound_processed) > 0:
            cos_similarity = 1 - cosine(self.inbound_request, inbound_processed)
            reward += cos_similarity * 100

        done = self.current_request_index >= len(self.fixed_requests)

        # エピソード終了時の報酬
        if done:
            reward += 1000
        
        # deadendの場合の報酬
        if deadend:
            reward -= 1000

        self.total_reward += reward

        # 次の状態を返す
        plan_flat = self.plan_data[[col for col in self.plan_data.columns if col.startswith('oil_')]].fillna(0).values.flatten() / MAX_CAPACITY  # 正規化
        next_state = np.concatenate((self.state, plan_flat))
        
        # タンクの状態とアクションの詳細を表示
#        print(f"Tank states after step {self.num_operations}: {self.state}")
#        print(f"Actions taken: {action_details}")
#        print(f"reward: {reward}")
#        print(f"reward total: {self.total_reward}")
        
        return next_state, reward, done, {}