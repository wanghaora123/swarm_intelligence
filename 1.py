import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# 定义环境
class CaptureEnv:
    def __init__(self, num_hunters=3, width=10, height=10):
        self.num_hunters = num_hunters
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.hunters = np.random.rand(self.num_hunters, 2) * [self.width, self.height]
        self.prey = np.random.rand(1, 2) * [self.width, self.height]
        return self._get_state()

    def step(self, actions):
        self.hunters += actions
        self.hunters = np.clip(self.hunters, 0, [self.width, self.height])
        self.prey += (np.random.rand(1, 2) - 0.5) * 0.1
        self.prey = np.clip(self.prey, 0, [self.width, self.height])
        done = self._is_captured()
        reward = 1 if done else -0.1
        return self._get_state(), reward, done

    def _get_state(self):
        return np.concatenate((self.hunters.flatten(), self.prey.flatten()))

    def _is_captured(self):
        distances = np.linalg.norm(self.hunters - self.prey, axis=1)
        return np.any(distances < 0.5)

    def render(self, episode=None):
        plt.clf()
        plt.scatter(self.hunters[:, 0], self.hunters[:, 1], c='blue', label='Hunters')
        plt.scatter(self.prey[:, 0], self.prey[:, 1], c='red', label='Prey')
        plt.legend()
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.pause(0.01)
        if episode is not None:
            plt.savefig(f'episode_{episode}.png')

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # Ensure action is flattened to match state dimension
        action = action.view(action.size(0), -1)  # Flatten action to [batch_size, num_agents * action_dim]
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义MADDPG算法
class MADDPG:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim, action_dim) for _ in range(num_agents)]

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=0.001) for critic in self.critics]

    def select_action(self, state):
        actions = []
        for i in range(self.num_agents):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            if state_tensor.shape[1] != self.state_dim:
                raise ValueError(f"State shape mismatch: expected {self.state_dim}, got {state_tensor.shape[1]}")
            action = self.actors[i](state_tensor)
            actions.append(action.detach().cpu().numpy().flatten())
        return np.array(actions)

    def train(self, replay_buffer, batch_size=32):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        for i in range(self.num_agents):
            state = states[:, i * self.state_dim:(i + 1) * self.state_dim]
            next_state = next_states[:, i * self.state_dim:(i + 1) * self.state_dim]
            action = actions[:, i * self.action_dim:(i + 1) * self.action_dim]
            reward = rewards[:, i].unsqueeze(1)
            done = dones[:, i].unsqueeze(1)

            # Critic loss
            next_action = self.actors[i](next_state)
            target_value = self.critics[i](next_state, next_action.detach())
            expected_value = reward + 0.99 * target_value * (1 - done)
            value = self.critics[i](state, action)
            critic_loss = torch.mean((expected_value - value) ** 2)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Actor loss
            policy_loss = -self.critics[i](state, self.actors[i](state)).mean()

            self.actor_optimizers[i].zero_grad()
            policy_loss.backward()
            self.actor_optimizers[i].step()

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32).view(batch_size, -1),  # Flatten actions
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# 初始化环境和算法
env = CaptureEnv()
maddpg = MADDPG(num_agents=3, state_dim=8, action_dim=2)
replay_buffer = ReplayBuffer(capacity=10000)

# 训练参数
num_episodes = 100
batch_size = 32

# 训练过程
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        actions = maddpg.select_action(state)
        next_state, reward, done = env.step(actions)
        total_reward += reward

        replay_buffer.push(state, actions, reward, next_state, done)
        state = next_state

        if len(replay_buffer) > batch_size:
            maddpg.train(replay_buffer, batch_size)

    if episode % 10 == 0:
        print(f'Episode: {episode}, Total Reward: {total_reward}')
        env.render(episode)

# 最终可视化
env.render()