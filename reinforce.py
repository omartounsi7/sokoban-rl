import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.SokobanEnv import SokobanEnv
from src.constants import *
import src.util


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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python reinforce.py <puzzle_file> <number_of_episodes>"
        )
        sys.exit(1)

    level_file = sys.argv[1]
    num_episodes = int(sys.argv[2])
    env = SokobanEnv(level_file)
    policy, rewards = reinforce_policy_gradient(env, num_episodes=num_episodes)
    env.autoplay(policy)
    env.root.mainloop()

    # torch.save(policy.state_dict(), "sokoban_policy_net.pth")