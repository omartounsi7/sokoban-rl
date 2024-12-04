import sys
import time
import psutil
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from src.SokobanEnv import SokobanEnv
from src.constants import *

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.index = 0
        self.size = 0

    def push(self, state, action, next_state, reward, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.next_states[self.index] = next_state
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32),
            torch.tensor(self.actions[indices], dtype=torch.long),
            torch.tensor(self.next_states[indices], dtype=torch.float32),
            torch.tensor(self.rewards[indices], dtype=torch.float32),
            torch.tensor(self.dones[indices], dtype=torch.float32),
        )

    def __len__(self):
        return self.size

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class DQNTrainer:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(int(MAX_STEPS_DQN / 10), state_size)
        self.total_timesteps = MAX_STEPS_DQN
        self.gamma = GAMMA
        self.start_epsilon = EPSILON
        self.epsilon_min = MIN_EPSILON
        self.epsilon = self.start_epsilon
        self.exp_fraction = 0.5
        self.batch_size = 256
        self.target_update = 1000  # Frequency of updating target network
        self.learn_start = 1000  # Number of steps before learning starts
        self.Q_net = DQNAgent(state_size, action_size).to(self.device)
        self.target_net = DQNAgent(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.Q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=LEARNING_RATE)
    
    def select_action(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_values = self.Q_net(state)
                return q_values.argmax().item()
    
    def push_transition(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        q_values = self.Q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.Q_net.state_dict())
    
    def train(self, env):
        state = env.reset()
        done = False
        step = 0

        while step < self.total_timesteps:
            action = self.select_action(state, self.epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            self.push_transition(state, action, next_state, reward, done)
            state = next_state
            step += 1

            if step > self.learn_start and step % 4 == 0:
                self.optimize_model()

            if step % self.target_update == 0:
                self.update_target_network()

            self.epsilon = linear_schedule(self.start_epsilon, self.epsilon_min, self.exp_fraction * self.total_timesteps, step)

            if done:
                state = env.reset()

            if step % 10000 == 0:
                print(f"Step {step}/{self.total_timesteps}")
                # avg_reward = self.evaluate(env)
                # print(f"Average Reward: {avg_reward:.2f}")

    def evaluate(self, env):
        total_reward = 0.0
        state = env.reset()
        done = False
        while not done:
            action = self.select_action(state, eps=0.0)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
        return total_reward
    
    def get_policy(self):
        policy = {}
        for state_index in range(self.memory.size):
            state = self.memory.states[state_index]
            serialized_state = tuple(state)
            with torch.no_grad():
                q_values = self.Q_net(torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax().item()
            policy[serialized_state] = action
        return policy

def deep_q_learning(env):
    print("Running Vanilla DQN algorithm...")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNTrainer(state_size, action_size)
    agent.train(env)
    print("Vanilla DQN algorithm completed.")
    return agent.get_policy()
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dqn.py <puzzle_file>")
        sys.exit(1)
    
    level_file = sys.argv[1]
    num_episodes = int(sys.argv[2])
    env = SokobanEnv(level_file)

    start_time = time.time()
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024

    policy = deep_q_learning(env)

    after = process.memory_info().rss / 1024 / 1024
    time_to_train = time.time() - start_time
    
    print(f"Time to train: {time_to_train:.2f}s")
    print(f"Total memory used: {after - before:.2f} MB")

    env.autoplay(policy)
    env.root.mainloop()