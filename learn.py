from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn
import numpy as np
from gym import spaces
from oilenv import OilDeliveryEnv
import mlflow
import tensorboard
import os

class CustomMLPExtractor(nn.Module):
    def __init__(self, features_dim, net_arch):
        super(CustomMLPExtractor, self).__init__()
        self.latent_dim_pi = net_arch[0]["pi"][-1]
        self.latent_dim_vf = net_arch[0]["vf"][-1]

        self.shared_net = nn.Sequential(
            nn.Linear(features_dim, net_arch[0]["pi"][0]),
            nn.ReLU(),
            nn.Linear(net_arch[0]["pi"][0], net_arch[0]["pi"][1]),
            nn.ReLU(),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(net_arch[0]["pi"][1], self.latent_dim_pi),
            nn.ReLU(),
        )

        self.value_net = nn.Sequential(
            nn.Linear(net_arch[0]["vf"][1], self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features):
        shared_latent = self.shared_net(features)
        latent_pi = self.policy_net(shared_latent)
        latent_vf = self.value_net(shared_latent)
        return latent_pi, latent_vf

    def forward_actor(self, features):
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent)

    def forward_critic(self, features):
        shared_latent = self.shared_net(features)
        return self.value_net(shared_latent)

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.net_arch = [dict(pi=[64, 64], vf=[64, 64])]
        self.mlp_extractor = CustomMLPExtractor(self.features_dim, self.net_arch)

        action_dim = sum(self.action_space.nvec)
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)

# ファイルパスを指定して環境をインスタンス化
file_path = "D:\\OneDrive\\oilopt\\plan.xlsx"
env = OilDeliveryEnv(file_path, inbound_reward_coef=1.0, outbound_reward_coef=1.0, negative_content_penalty=1.0)

# TensorBoardのログディレクトリを設定
log_dir = "./tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)

# カスタムポリシーを使用してPPOをインスタンス化
model = PPO(CustomPolicy, env, verbose=1, tensorboard_log=log_dir)

# 学習
model.learn(total_timesteps=100000)