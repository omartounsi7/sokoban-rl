import sys
import time
import psutil
import os
import random
from src.SokobanEnv import SokobanEnv
from src.constants import *


def mc_policy_evaluation(env, num_episodes=MAX_EPISODES_MC, gamma=GAMMA, epsilon=EPSILON):
    print("Running Monte Carlo policy optimization algorithm...")
    Q = {}
    returns_sum = {}
    returns_count = {}
    policy = {}
    action_space = list(ACTION_SPACE.keys())

    episode = 0
    no_policy_change_ctr = 0
    explored_states = set()

    while no_policy_change_ctr < EARLY_STOPPING_PATIENCE and episode < num_episodes:
        if (episode + 1) % 1000 == 0:
            print("Episode " + str(episode + 1))
        trajectory = []
        terminalState = False
        visited_states = set()
        current_state, info = env.reset()
        current_state = tuple(current_state)

        while not terminalState:
            visited_states.add(current_state)
            explored_states.add(current_state)
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

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
        has_policy_changed = False
        
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

            if policy.get(state) is not None and best_action != policy[state]:
                has_policy_changed = True
            policy[state] = best_action

        episode += 1
        if not has_policy_changed:
            no_policy_change_ctr += 1
        else:
            no_policy_change_ctr = 0

    print(f"Total number of unique states explored: {len(explored_states)}")
    if episode != num_episodes:
        print("No policy change for " + str(EARLY_STOPPING_PATIENCE) + " episodes, algorithm has converged.")
        print("Number of episodes to converge: " + str(episode + 1))
    print("Monte Carlo policy optimization algorithm completed.")
    return policy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python montecarlo.py <puzzle_file>"
        )
        sys.exit(1)

    env = SokobanEnv(sys.argv[1])

    start_time = time.time()
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024

    policy = mc_policy_evaluation(env)

    after = process.memory_info().rss / 1024 / 1024
    time_to_train = time.time() - start_time
    
    print(f"Time to train: {time_to_train:.2f}s")
    print(f"Total memory used: {after - before:.2f} MB")

    env.autoplay(policy)
    env.root.mainloop()