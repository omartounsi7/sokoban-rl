import sys
import time
import random
import psutil
import os
from src.SokobanEnv import SokobanEnv
from src.constants import *


def mc_policy_evaluation(env, num_episodes=100000, gamma=0.99, epsilon=0.9):
    print("Running Monte Carlo policy optimization algorithm...")
    Q = {}
    returns_sum = {}
    returns_count = {}
    policy = {}
    action_space = list(ACTIONSPACE.keys())

    for episode in range(num_episodes):
        if (episode + 1) % 100 == 0:
            print("Episode " + str(episode + 1) + "/" + str(num_episodes))
        trajectory = []
        terminalState = False
        visited_states = set()
        current_state = tuple(env.reset())

        while not terminalState:
            visited_states.add(current_state)
            epsilon = max(MINEPSILON, epsilon * EPSILONDECAY)

            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                action = policy.get(current_state, random.choice(action_space))
            
            obs, reward, terminated, truncated, info = env.step(action)
            serialized_obs = tuple(obs)
            
            if serialized_obs in visited_states:
                # print("LOOP DETECTED!")
                trajectory.append((current_state, action, SUPERMALUS))
                terminalState = True
            else:
                trajectory.append((current_state, action, reward))

            if terminated:
                terminalState = True
            else:
                current_state = serialized_obs

        episode_length = len(trajectory)

        for t in range(episode_length):
            state, action, reward = trajectory[t]
            if state not in Q:
                Q[state] = {a: 0.0 for a in action_space}
                returns_sum[state] = {a: 0.0 for a in action_space}
                returns_count[state] = {a: 0 for a in action_space}
            G = 0
            for k in range(t, episode_length):
                state_k, action_k, reward_k = trajectory[k]
                G += gamma ** (k - t) * reward_k
            returns_sum[state][action] += G
            returns_count[state][action] += 1
            Q[state][action] = returns_sum[state][action] / returns_count[state][action]
            best_action = max(Q[state], key=Q[state].get)
            policy[state] = best_action

    print("Total number of episodes: " + str(episode + 1))
    print("Monte Carlo policy optimization algorithm completed.")
    return policy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python montecarlo.py <puzzle_file> <number_of_episodes>"
        )
        sys.exit(1)

    env = SokobanEnv(sys.argv[1])

    start_time = time.time()
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024

    policy = mc_policy_evaluation(
        env,
        num_episodes=int(sys.argv[2])
    )

    after = process.memory_info().rss / 1024 / 1024
    time_to_train = time.time() - start_time
    
    print(f"Time to train: {time_to_train:.2f}s")
    print(f"Total memory used: {after - before:.2f} MB")

    env.autoplay(policy)
    env.root.mainloop()