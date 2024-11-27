import src.util
from src.constants import *


def compute_v_and_q_from_policy(env, policy_actions, gamma=0.99):
    V = {}
    Q = {}

    state = env.reset()
    trajectory = []

    for action in policy_actions:
        serialized_state = src.util.serialize_state(state)
        next_state, reward, done, truncated, _ = env.step(ACTIONSPACE[action])
        trajectory.append((serialized_state, action, reward))
        state = next_state
        if done or truncated:
            break

    G = 0
    for serialized_state, action, reward in reversed(trajectory):
        G = reward + gamma * G
        V[serialized_state] = G
        if serialized_state not in Q:
            Q[serialized_state] = {}
        Q[serialized_state][action] = G

    return V, Q
