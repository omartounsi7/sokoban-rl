import src.util
from src.constants import *


def compute_value_functions(env, optimal_policy, gamma=0.99):
    V = {}
    Q = {}

    for state in optimal_policy.keys():
        V[state] = 0
        Q[state] = {}

        env.reset()
        env.set_serialized_state(state)

        for action in ACTIONSPACE.keys():
            next_obs, reward, done, _, _ = env.step(action)
            next_state = src.util.serialize_state(next_obs)

            if done:
                Q[state][action] = reward
            else:
                Q[state][action] = reward + gamma * V.get(next_state, 0)

        V[state] = max(Q[state].values())

    return V, Q
