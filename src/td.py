from src.SokobanEnv import SokobanEnv
from src.constants import *
import random


def td_learning(env, num_episodes=MAX_EPISODES_TD, gamma=0.95, epsilon=EPSILON, alpha=0.01):
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


  if episode != num_episodes:
      print(f"Number of episodes to converge: {episode}")
  print("TD policy optimization algorithm completed.")


  return policy