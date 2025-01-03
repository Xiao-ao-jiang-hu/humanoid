import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import cv2
import humanoid_bench
from stable_baselines3.common.vec_env import SubprocVecEnv


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # input state B x W x S
        self.hidden_size = hidden_dim
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.gru_unit = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_mu = nn.Linear(hidden_dim, action_dim)
        self.action_var = nn.Linear(hidden_dim, action_dim)
        self.state_mu = nn.Linear(hidden_dim, state_dim)
        self.state_var = nn.Linear(hidden_dim, state_dim)

    def forward(self, input_state, input_action):
        B, W, _ = input_action.shape
        action_hidden = self.action_encoder(
            input_action.view(B*W, -1)).view(B, W, -1)
        state_hidden = self.state_encoder(
            input_state.view(B*(W+1), -1)).view(B, W+1, -1)

        cat = torch.empty(
            B, 2 * W + 1, self.hidden_size, dtype=action_hidden.dtype, device=action_hidden.device)
        cat[:, 0::2, :] = state_hidden
        cat[:, 1::2, :] = action_hidden

        hiddens = [torch.zeros(B, self.hidden_size)]
        for i in range(2*W+1):
            hiddens.append(self.gru_unit(cat[:, i, :], hiddens[-1]))

        mus = []
        vars = []
        for i in range(1, 2*W+2):
            if i % 2:
                mus.append(self.action_mu(hiddens[i]))
                vars.append(self.action_var(hiddens[i]))
            else:
                mus.append(self.state_mu(hiddens[i]))
                vars.append(self.state_var(hiddens[i]))

        return torch.stack(mus[0::2], dim=1), torch.stack(mus[1::2], dim=1), torch.stack(vars[0::2], dim=1), torch.stack(vars[1::2], dim=1)


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # input state B x W x S
        self.hidden_size = hidden_dim
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.gru_unit = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_value = nn.Linear(2*hidden_dim, 1)

    def forward(self, input_state, input_action):
        B, W, _ = input_action.shape
        action_hidden = self.action_encoder(
            input_action.view(B*W, -1)).view(B, W, -1)
        state_hidden = self.state_encoder(
            input_state[:, :-1, :].reshape(B*W, -1)).view(B, W, -1)

        cat = torch.empty(
            B, 2 * W, self.hidden_size, dtype=action_hidden.dtype, device=action_hidden.device)
        cat[:, 0::2, :] = state_hidden
        cat[:, 1::2, :] = action_hidden

        hiddens = [torch.zeros(B, self.hidden_size)]
        for i in range(2*W):
            hiddens.append(self.gru_unit(cat[:, i, :], hiddens[-1]))

        values = []
        for i in range(1, 2*W+1, 2):
            hidden_cat = torch.cat([hiddens[i], hiddens[i+1]], dim=-1)
            value = self.action_value(hidden_cat)
            values.append(value)

        return torch.stack(values, dim=1)


# model = PolicyNet(216, 512, 61)
# am, sm, av, sv = model(torch.rand((8, 17, 216)), torch.rand((8, 16, 61)))
# print(am.shape)
# print(av.shape)
# print(sm.shape)
# print(sv.shape)


# q = QNet(216, 512, 61)
# values = q(torch.rand((8, 17, 216)), torch.rand((8, 16, 61)))
# print(values.shape)

# exit(0)


class sac:
    def __init__(self, action_dim, state_dim, hidden_dim, window_len):
        # policy input: B x W x A
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.qnet1 = QNet(state_dim, hidden_dim, action_dim)
        self.qnet2 = QNet(state_dim, hidden_dim, action_dim)
        self.qtar1 = QNet(state_dim, hidden_dim, action_dim)
        self.qtar2 = QNet(state_dim, hidden_dim, action_dim)

        self.policy_op = torch.optim.Adam(
            self.policy_net.parameters(), lr=0.001)
        self.qnet1_op = torch.optim.Adam(self.qnet1.parameters(), lr=0.001)
        self.qnet2_op = torch.optim.Adam(self.qnet2.parameters(), lr=0.001)
        self.qtar1.load_state_dict(self.qnet1.state_dict())
        self.qtar2.load_state_dict(self.qnet2.state_dict())

    def _compute_density(self, mean, variance):
        B, D = mean.shape
        normalization_factor = 1.0 / \
            torch.sqrt((2 * torch.pi * variance).prod(dim=1, keepdim=True))
        exponent = -0.5 * ((mean ** 2) / variance).sum(dim=1, keepdim=True)
        density = normalization_factor * torch.exp(exponent)
        return density

    def gen_episode(self):
        pass

    def init_model(self):
        pass

    def update_q(self, sample, gamma, alpha, pi_a_given_s):
        states, actions, rewards = sample
        y = rewards + gamma * \
            (torch.minimum(self.qtar1(states, actions)) -
             alpha*torch.log(pi_a_given_s))
        B = y.size(0)
        loss1 = 1/B*torch.sum(torch.square(self.qnet1()-y))
        loss2 = 1/B*torch.sum(torch.square(self.qnet2()-y))
        loss1.backward()
        loss2.backward()
        self.qnet1_op.step()
        self.qnet2_op.step()
        self.qnet1_op.zero_grad()
        self.qnet2_op.zero_grad()
        pass

    def update_policy(self):
        pass

    def update_model_param(self, group1, group2, alpha):
        with torch.no_grad():
            for param1, param2 in zip(group1, group2):
                param1.data = (1-alpha)*param1.data + alpha*param2
