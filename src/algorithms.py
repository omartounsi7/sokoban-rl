from src.constants import *
import time
import random
import copy
import src.util

def mc_policy_evaluation(initial_state, num_episodes=100000, gamma=0.95, epsilon=0.9, every_visit=True, convergence_thres=0.001):        
    print("Running Monte Carlo policy optimization algorithm...")
    start_time = time.time()
    Q = {}
    returns_sum = {}
    returns_count = {}
    policy = {}
    
    for episode in range(num_episodes):
        # print("Episode", episode + 1)
        Q_old = copy.deepcopy(Q)
        current_state = copy.deepcopy(initial_state)
        steps = 0
        trajectory = []
        terminalState = False

        moved_box = False
        prev_pos = None

        while not terminalState and steps < MAXSTEPS:
            steps += 1
            serialized_current_state = src.util.serialize_state(current_state)
            epsilon = max(0.1, epsilon * EPSILONDECAY)

            if random.random() < epsilon:
                action = random.choice(ACTIONSPACE)
            else:
                action = policy.get(serialized_current_state, random.choice(ACTIONSPACE))
            
            action_vector = src.util.get_action(action)

            temp_pos = src.util.find_player_in_state(current_state)
            previously_moved_box = moved_box

            new_pos, reward, moved_box = src.util.execute_action(current_state, action_vector)
            
            if prev_pos == new_pos and not previously_moved_box:
                # print("LOOP DETECTED!")
                terminalState = True
                trajectory.append((serialized_current_state, action, SUPERMALUS))
            else:
                trajectory.append((serialized_current_state, action, reward))
            
            prev_pos = temp_pos

            if reward == SUPERBONUS:
                # print("PUZZLE COMPLETED!")
                terminalState = True
            elif reward == SUPERMALUS:
                # print("BOX STUCK!")
                terminalState = True

        # print("Number of steps: " + str(steps))
        
        episode_length = len(trajectory)
        visited_state_actions = set()

        for t in range(episode_length):
            state, action, reward = trajectory[t]
            if every_visit or (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))

                if state not in Q:
                    Q[state] = {a: 0.0 for a in ACTIONSPACE}
                    returns_sum[state] = {a: 0.0 for a in ACTIONSPACE}
                    returns_count[state] = {a: 0 for a in ACTIONSPACE}
                
                G = 0
                for k in range(t, episode_length):
                    state_k, action_k, reward_k = trajectory[k]
                    G += gamma ** (k - t) * reward_k
                
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                best_action = max(Q[state], key=Q[state].get)
                policy[state] = best_action

        if convergence_thres > 0 and has_converged(Q_old, Q, convergence_thres):
            print("Q has converged.")
            break

    time_to_train = time.time() - start_time
    print(f"Time to train: : {time_to_train:.2f}s")
    print("Total number of episodes: " + str(episode + 1))
    print("Monte Carlo policy optimization completed.")

    return policy

def has_converged(Q_old, Q_new, threshold):
    if len(Q_old) < 5:
        return False
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
