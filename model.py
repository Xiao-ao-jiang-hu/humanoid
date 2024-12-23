import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn

env = gym.make("h1hand-truck-v0")
# env.reset()
print(env.reset())
print(env.action_space)
print(env.observation_space)

term = False
while not term:
    obs, reward, term, a, b = env.step(env.action_space.sample())
    print(reward)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.hidden = hidden_dim
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.action_emb = nn.Linear(action_dim, hidden_dim)
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 8, 1024, activation=F.gelu), num_layers=6, norm=nn.LayerNorm)
        # from S hidden to a
        self.action_mu = nn.Linear(hidden_dim, action_dim)
        self.action_sigma = nn.Linear(hidden_dim, action_dim)
        # from a hidden to S'
        self.state_mu = nn.Linear(hidden_dim, state_dim)
        self.state_sigma = nn.Linear(hidden_dim, state_dim)

    def forward(self, states, actions):
        '''
        input:
        - actions: tensor B x D x T
        - states: tesor B x D' x (T+1)
        - combine to SaSaS......Sa
        output:
        - answer is aSaS....aS
        '''
        state_h = self.state_emb(states)
        actions_h = self.action_emb(actions)
        # B x H x (2T+1)
        input_sequence = torch.empty((state_h.size(0), self.hidden, state_h.size(
            -1) + actions_h.size(-1) - 1), dtype=state_h.dtype)
        input_sequence[:, :, 0::2] = state_h[:, :, :-1]
        input_sequence[:, :, 1::2] = actions_h
        hiddens = self.encoder.forward(input_sequence, is_causal=True)
        state_out = hiddens[:, :, 0:2]
        action_out = hiddens[:, :, 1:2]
        

        return None


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
