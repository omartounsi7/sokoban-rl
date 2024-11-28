import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.SokobanEnv import SokobanEnv
from src.constants import *
import src.util

from ray.rllib.algorithms.sac.sac import SAC


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


def reinforce_policy_gradient(env, num_episodes=1000, gamma=0.99, lr=1e-3):
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = len(ACTIONSPACE)
    policy_net = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    all_rewards = []
    policy = {}
    best_policy = {}
    best_reward = float("-inf")

    for episode in range(num_episodes):
        state = env.reset()
        state_serialized = src.util.serialize_state(state)
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        log_probs = []
        rewards = []
        trajectory = []

        done = False
        truncated = False
        while not done and not truncated:
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            next_state_serialized = src.util.serialize_state(next_state)
            next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.float32)

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            trajectory.append((state_serialized, action.item()))

            state_serialized = next_state_serialized
            state_tensor = next_state_tensor

        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)
        discounted_returns = torch.tensor(discounted_returns)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (
            discounted_returns.std() + 1e-5
        )

        loss = 0
        for log_prob, G in zip(log_probs, discounted_returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for state, action in trajectory:
            policy[state] = action

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_policy = policy.copy()

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}"
        )

    return best_policy, all_rewards


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(x)


def actor_critic_policy_gradient(
    env, num_episodes=1000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3
):
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = len(ACTIONSPACE)

    # Initialize actor and critic networks
    actor_net = PolicyNetwork(input_dim, output_dim)
    critic_net = CriticNetwork(input_dim)

    # Optimizers for actor and critic
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=lr_critic)

    all_rewards = []
    policy = {}
    best_policy = {}
    best_reward = float("-inf")

    for episode in range(num_episodes):
        state = env.reset()
        state_serialized = src.util.serialize_state(state)
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        log_probs = []
        rewards = []
        values = []
        trajectory = []

        done = False
        truncated = False
        while not done and not truncated:
            # Actor chooses action
            action_probs = actor_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            # Critic evaluates the state
            value = critic_net(state_tensor)
            values.append(value)

            # Perform action in the environment
            next_state, reward, done, truncated, _ = env.step(action.item())
            next_state_serialized = src.util.serialize_state(next_state)
            next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.float32)

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            trajectory.append((state_serialized, action.item()))

            state_serialized = next_state_serialized
            state_tensor = next_state_tensor

        # Compute discounted returns
        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)
        discounted_returns = torch.tensor(discounted_returns)

        # Normalize returns for stability
        returns = (discounted_returns - discounted_returns.mean()) / (
            discounted_returns.std() + 1e-5
        )

        # Compute actor and critic losses
        actor_loss = 0
        critic_loss = 0
        for log_prob, value, G in zip(log_probs, values, returns):
            advantage = G - value.item()  # Temporal difference advantage
            actor_loss -= log_prob * advantage
            critic_loss += (G - value) ** 2

        # Update actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update the policy dictionary
        for state, action in trajectory:
            policy[state] = action

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_policy = policy.copy()

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}"
        )

    return best_policy, all_rewards


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.SokobanEnv import SokobanEnv
from src.constants import *
import src.util

from ray.rllib.algorithms.sac.sac import SAC


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(x)


def reinforce_policy_gradient(env, num_episodes=1000, gamma=0.99, lr=1e-3):
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = len(ACTIONSPACE)
    policy_net = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    all_rewards = []
    policy = {}
    best_policy = {}
    best_reward = float("-inf")

    for episode in range(num_episodes):
        state = env.reset()
        state_serialized = src.util.serialize_state(state)
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        log_probs = []
        rewards = []
        trajectory = []

        done = False
        truncated = False
        while not done and not truncated:
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            next_state_serialized = src.util.serialize_state(next_state)
            next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.float32)

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            trajectory.append((state_serialized, action.item()))

            state_serialized = next_state_serialized
            state_tensor = next_state_tensor

        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)
        discounted_returns = torch.tensor(discounted_returns)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (
            discounted_returns.std() + 1e-5
        )

        loss = 0
        for log_prob, G in zip(log_probs, discounted_returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for state, action in trajectory:
            policy[state] = action

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_policy = policy.copy()

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}"
        )

    return best_policy, all_rewards


def actor_critic_policy_gradient(
    env, num_episodes=1000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3
):
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = len(ACTIONSPACE)

    actor_net = PolicyNetwork(input_dim, output_dim)
    critic_net = CriticNetwork(input_dim)

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=lr_critic)

    all_rewards = []
    policy = {}
    best_policy = {}
    best_reward = float("-inf")

    for episode in range(num_episodes):
        state = env.reset()
        state_serialized = src.util.serialize_state(state)
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        log_probs = []
        rewards = []
        values = []
        trajectory = []

        done = False
        truncated = False
        while not done and not truncated:
            action_probs = actor_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            value = critic_net(state_tensor)
            values.append(value)

            next_state, reward, done, truncated, _ = env.step(action.item())
            next_state_serialized = src.util.serialize_state(next_state)
            next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.float32)

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            trajectory.append((state_serialized, action.item()))

            state_serialized = next_state_serialized
            state_tensor = next_state_tensor

        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)
        discounted_returns = torch.tensor(discounted_returns)

        returns = (discounted_returns - discounted_returns.mean()) / (
            discounted_returns.std() + 1e-5
        )

        actor_loss = 0
        critic_loss = 0
        for log_prob, value, G in zip(log_probs, values, returns):
            advantage = G - value.item()
            actor_loss -= log_prob * advantage
            critic_loss += (G - value) ** 2

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        for state, action in trajectory:
            policy[state] = action

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_policy = policy.copy()

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}"
        )

    return best_policy, all_rewards


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python policygradient.py <puzzle_file> <number_of_episodes> <method>"
        )
        print("Methods: reinforce, actor_critic")
        sys.exit(1)

    level_file = sys.argv[1]
    num_episodes = int(sys.argv[2])
    method = sys.argv[3]

    env = SokobanEnv(level_file)

    if method == "reinforce":
        policy, rewards = reinforce_policy_gradient(env, num_episodes=num_episodes)
    elif method == "actor_critic":
        policy, rewards = actor_critic_policy_gradient(env, num_episodes=num_episodes)
    else:
        print(f"Unknown method: {method}. Use 'reinforce' or 'actor_critic'.")
        sys.exit(1)

    env.autoplay(policy)
    env.root.mainloop()

    # torch.save(policy.state_dict(), "sokoban_policy_net.pth")
