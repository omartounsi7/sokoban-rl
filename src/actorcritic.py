import sys
import time
import psutil
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.SokobanEnv import SokobanEnv
from src.constants import *
from src.reinforce import PolicyNetwork


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(x)


def actor_critic_policy_gradient(env, num_episodes=MAX_EPISODES_PG, gamma=GAMMA, lr_actor=LEARNING_RATE, lr_critic=LEARNING_RATE):
    print("Running Actor-Critic algorithm...")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    actor_net = PolicyNetwork(input_dim, output_dim)
    critic_net = CriticNetwork(input_dim)

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=lr_critic)

    all_rewards = []
    policy = {}
    best_policy = {}
    best_reward = float("-inf")
    explored_states = set()

    for episode in range(num_episodes):
        state, info = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        log_probs = []
        rewards = []
        values = []
        trajectory = []
        visited = set()

        steps = 0
        done = False
        while not done and steps < MAX_STEPS_PG:
            steps += 1
            visited.add(tuple(state))
            explored_states.add(tuple(state))
            action_probs = actor_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            value = critic_net(state_tensor)
            values.append(value)

            next_state, reward, done, truncated, _ = env.step(action.item())
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            if tuple(next_state) in visited:
                reward += SUPERMALUS

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            trajectory.append((tuple(state), action.item()))

            state = next_state
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
            f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}"
        )

        # Check for convergence
        if best_reward > BEST_REWARD_THRESHOLD:
            print("Optimal reward reached, algorithm has converged.")
            break

    print(f"Total number of unique states explored: {len(explored_states)}")

    if episode != num_episodes:
        print("Number of episodes to converge: " + str(episode + 1))
    print("Actor-Critic algorithm completed.")
    return best_policy, all_rewards


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python actorcritic.py <puzzle_file> <number_of_episodes>")
        sys.exit(1)

    level_file = sys.argv[1]
    num_episodes = int(sys.argv[2])
    env = SokobanEnv(level_file)

    start_time = time.time()
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024

    policy, rewards = actor_critic_policy_gradient(env, num_episodes=num_episodes)

    after = process.memory_info().rss / 1024 / 1024
    time_to_train = time.time() - start_time
    
    print(f"Time to train: {time_to_train:.2f}s")
    print(f"Total memory used: {after - before:.2f} MB")

    env.autoplay(policy)
    env.root.mainloop()