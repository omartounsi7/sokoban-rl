import sys
from src.SokobanGame import SokobanEnv
from src.constants import *
import src.util
import time
import random
import copy




"""def td_learning(env, num_episodes=100000, gamma=0.95, epsilon=0.9, alpha=0.01, convergence_thres=0):
  print("Running Temporal Difference (TD) policy evaluation...")
  start_time = time.time()
  Q = {}  # State-action value function
  policy = {}  # Mapping from state to best action
  action_space = list(ACTIONSPACE.keys())




  for episode in range(num_episodes):
      Q_old = copy.deepcopy(Q)
      current_state = src.util.serialize_state(env.reset())
      done = False
      step_count = 0
      visited_states = set()
    
      while not done:
          # Epsilon-greedy action selection
          epsilon = max(MINEPSILON, epsilon * EPSILONDECAY)
          if random.random() < epsilon:
              action = random.choice(action_space)
          else:
              action = policy.get(current_state, random.choice(action_space))




          # Take the action
          obs, reward, done, _ = env.step(action)
          next_state = src.util.serialize_state(obs)




          # Penalize loops
          if current_state in visited_states:
              reward = SUPERMALUS
              done = True
          else:
              visited_states.add(current_state)




          # Initialize Q-values for unseen states
          if current_state not in Q:
              Q[current_state] = {a: 0.0 for a in action_space}
          if next_state not in Q:
              Q[next_state] = {a: 0.0 for a in action_space}




          # TD Update (this is the difference between MC and TD)
          target = reward + (gamma * max(Q[next_state].values()) if not done else 0)
          Q[current_state][action] += alpha * (target - Q[current_state][action])




          # Update policy
          policy[current_state] = max(Q[current_state], key=Q[current_state].get)




          # Move to the next state
          current_state = next_state
          step_count += 1




      # Check convergence
      if episode > MINEPISODES and has_converged(Q_old, Q, convergence_thres):
          print("Q has converged.")
          break




      # Log progress
      if episode % 100 == 0:
          print(f"Episode {episode + 1}, Steps: {step_count}, Epsilon: {epsilon:.2f}")




  time_to_train = time.time() - start_time
  print(f"Time to train: {time_to_train:.2f}s")
  print(f"Total number of episodes: {episode + 1}")
  print("TD policy evaluation completed.")
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




if __name__ == "__main__":
  if len(sys.argv) != 5:
      print("Usage: python td_fa_learning.py <puzzle_file> <number_of_episodes> <discount_factor> <exploration_rate>")
      sys.exit(1)
   level_file = sys.argv[1]
  env = SokobanEnv(level_file)
  policy = td_learning(env, num_episodes=int(sys.argv[2]), gamma=float(sys.argv[3]), epsilon=float(sys.argv[4]))
  print(policy)
  env.autoplay(policy)
  env.root.mainloop()"""


import sys
from src.SokobanEnv import SokobanEnv
from src.constants import *
import src.util
import time
import random
import copy
import psutil
import os


# set MAX_EPISODES_TD in constants.py
def td_learning(env, num_episodes=MAX_EPISODES_TD, gamma=GAMMA, epsilon=EPSILON, alpha=ALPHA):
   print("Running Temporal Difference (TD) policy optimization algorithm...")
   Q = {}
   policy = {}
   action_space = list(ACTION_SPACE.keys())


   episode = 0
   no_policy_change_ctr = 0


   while no_policy_change_ctr < EARLY_STOPPING_PATIENCE and episode < num_episodes:
       if (episode + 1) % 100 == 0:
           print(f"Episode {episode + 1}/{num_episodes}")


       current_state = tuple(env.reset())
       terminalState = False
       visited_states = set()
       has_policy_changed = False


       while not terminalState:
           visited_states.add(current_state)
           epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)


           # Epsilon-greedy action selection
           if random.random() < epsilon:
               action = random.choice(action_space)
           else:
               action = policy.get(current_state, random.choice(action_space))


           # Take the action
           obs, reward, terminated, truncated, info = env.step(action)
           next_state = tuple(obs)


           # Penalize loops
           if next_state in visited_states:
               reward = SUPERMALUS
               terminalState = True


           # Initialize Q-values for unseen states
           if current_state not in Q:
               Q[current_state] = {a: 0.0 for a in action_space}
           if next_state not in Q:
               Q[next_state] = {a: 0.0 for a in action_space}


           # TD Update
           target = reward + (gamma * max(Q[next_state].values()) if not terminated else 0)
           Q[current_state][action] += alpha * (target - Q[current_state][action])


           # Update policy
           best_action = max(Q[current_state], key=Q[current_state].get)
           if policy.get(current_state) is not None and best_action != policy[current_state]:
               has_policy_changed = True
           policy[current_state] = best_action


           # Transition to next state
           if terminated:
               terminalState = True
           else:
               current_state = next_state


       episode += 1
       if not has_policy_changed:
           no_policy_change_ctr += 1
       else:
           no_policy_change_ctr = 0


   if episode != num_episodes:
       print(f"Number of episodes to converge: {episode}")
   print("TD policy optimization algorithm completed.")
   return policy




if __name__ == "__main__":
   if len(sys.argv) != 2:
       print("Usage: python td_learning.py <puzzle_file>")
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
