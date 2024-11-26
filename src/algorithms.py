from src.constants import *
import src.util
import time
import random
import copy


def mc_policy_evaluation(env, num_episodes=100000, gamma=0.95, epsilon=0.9, convergence_thres=0, every_visit=True):
    print("Running Monte Carlo policy optimization algorithm...")
    start_time = time.time()
    Q = {}
    returns_sum = {}
    returns_count = {}
    policy = {}
    action_space = list(ACTIONSPACE.keys())
    
    for episode in range(num_episodes):
        # print("Episode " + str(episode + 1))
        Q_old = copy.deepcopy(Q)
        trajectory = []
        terminalState = False
        visited_states = set()
        current_state = src.util.serialize_state(env.reset())

        while not terminalState:
            visited_states.add(current_state)
            epsilon = max(MINEPSILON, epsilon * EPSILONDECAY)
            
            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                action = policy.get(current_state, random.choice(action_space))
            
            obs, reward, done, info = env.step(action)
            serialized_obs = src.util.serialize_state(obs)
            
            if serialized_obs in visited_states:
                # print("LOOP DETECTED!")
                trajectory.append((current_state, action, SUPERMALUS))
                terminalState = True
            else:
                trajectory.append((current_state, action, reward))
            
            if done:
                terminalState = True
            else:
                current_state = serialized_obs
        
        episode_length = len(trajectory)
        visited_state_actions = set()

        for t in range(episode_length):
            state, action, reward = trajectory[t]
            if every_visit or (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
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

        if episode > MINEPISODES and has_converged(Q_old, Q, convergence_thres):
            print("Q has converged.")
            break
    
    time_to_train = time.time() - start_time
    print(f"Time to train: {time_to_train:.2f}s")
    print("Total number of episodes: " + str(episode + 1))
    print("Monte Carlo policy optimization completed.")
    return policy

def has_converged(Q_old, Q_new, threshold):
    max_diff = 0 
    for state in Q_old:
        if state not in Q_new:
            continue 
        action_old = max(Q_old[state], key=Q_old[state].get)
        action_new = max(Q_new[state], key=Q_new[state].get)
        if action_old != action_new:
            return False
        for action in Q_old[state]:
            if action not in Q_new[state]:
                continue
            diff = abs(Q_old[state][action] - Q_new[state][action])
            max_diff = max(max_diff, diff) 
    return max_diff < threshold
