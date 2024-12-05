import os
import sys
import time
import random
import psutil
from src.SokobanEnv import SokobanEnv
from src.constants import *


def td_learning(env, num_episodes=MAX_EPISODES_TD, gamma=0.95, epsilon=EPSILON, alpha=0.01):
    print("Running Temporal Difference Learning algorithm...")
    Q = {}
    policy = {}
    action_space = list(ACTION_SPACE.keys())
    episode = 0
    no_policy_change_ctr = 0
    explored_states = set()

    while no_policy_change_ctr < EARLY_STOPPING_PATIENCE and episode < num_episodes:
        if (episode + 1) % 1000 == 0:
            print("Episode " + str(episode + 1))

        current_state = tuple(env.reset())
        terminalState = False
        visited_states = set()
        has_policy_changed = False

        while not terminalState:
            visited_states.add(current_state)
            explored_states.add(current_state)
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                action = policy.get(current_state, random.choice(action_space))

            # take the action
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = tuple(obs)

            # penalize loops
            if next_state in visited_states:
                reward = SUPERMALUS
                terminalState = True

            # initialize Q-values for unseen states
            if current_state not in Q:
                Q[current_state] = {a: 0.0 for a in action_space}
            if next_state not in Q:
                Q[next_state] = {a: 0.0 for a in action_space}

            # TD update
            target = reward + (gamma * max(Q[next_state].values()) if not terminated else 0)
            Q[current_state][action] += alpha * (target - Q[current_state][action])

            # update policy
            best_action = max(Q[current_state], key=Q[current_state].get)
            if policy.get(current_state) is not None and best_action != policy[current_state]:
                has_policy_changed = True
            policy[current_state] = best_action

            # transition to next state
            if terminated:
                terminalState = True
            else:
                current_state = next_state

        episode += 1
        if not has_policy_changed:
            no_policy_change_ctr += 1
        else:
            no_policy_change_ctr = 0

    print(f"Total number of unique states explored: {len(explored_states)}")

    if episode != num_episodes:
        print("No policy change for " + str(EARLY_STOPPING_PATIENCE) + " episodes, algorithm has converged.")
        print("Number of episodes to converge: " + str(episode + 1))
    print("Temporal Difference Learning algorithm completed.")

    return policy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python td.py <puzzle_file>"
        )
        sys.exit(1)

    env = SokobanEnv(sys.argv[1])

    start_time = time.time()
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024

    policy = td_learning(env)

    after = process.memory_info().rss / 1024 / 1024
    time_to_train = time.time() - start_time
    
    print(f"Time to train: {time_to_train:.2f}s")
    print(f"Total memory used: {after - before:.2f} MB")

    env.autoplay(policy)
    env.root.mainloop()