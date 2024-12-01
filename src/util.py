import itertools
from src.constants import *


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

def compute_v_and_q_from_policy(env, policy_actions, gamma):
    # policy_actions should be numeric
    # e.g. [0, 1, 2, 3] not ['up', 'left', 'down', 'right']
    # Note that this function only returns the V and Q values for the states visited in the inputted policy.
    V = {}
    Q = {}

    state = env.reset()
    trajectory = []

    for action in policy_actions:
        serialized_state = tuple(state)
        next_state, reward, done, truncated, _ = env.step(action)
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

def find_player_in_state(state):
    for y, row in enumerate(state):
        for x, cell in enumerate(row):
            if cell in ('@', '+'):
                return (x, y)
    return None

def is_wall(state, x, y):
    return state[y][x] == "#"

def is_box(state, x, y):
    return state[y][x] in ["$", "*"]

def is_box_placed(state, x, y):
    return state[y][x] == "*"

def is_obstacle(state, x, y):
    return state[y][x] in ["#", "$", "*"]

def is_box_stuck(state, x, y):
    if is_box_placed(state, x, y):
        return False
    
    dirs = list(ACTION_MAP.values())

    for i in range(4):
        dx1, dy1 = dirs[i]
        dx2, dy2 = dirs[(i + 1) % 4]
        if is_obstacle(state, x + dx1, y + dy1) and is_obstacle(state, x + dx2, y + dy2):
            return True

    return False

def move_agent(state, x, y, nx, ny):
    if state[y][x] == "@":
        state[y][x] = " "
    elif state[y][x] == "+":
        state[y][x] = "."
    if state[ny][nx] == ".":
        state[ny][nx] = "+"
    else:
        state[ny][nx] = "@"

def is_position_free(state, x, y):
    return state[y][x] in [' ', '.']

def move_box(state, x, y, dx, dy):
    nx, ny = x + dx, y + dy
    if not is_position_free(state, nx, ny):
        return False
    if state[y][x] == "$":
        state[y][x] = " "
    elif state[y][x] == "*":
        state[y][x] = "."
    if state[ny][nx] == ".":
        state[ny][nx] = "*"
    else:
        state[ny][nx] = "$"
    return True

def check_win(state):
    for row in state:
        if '$' in row:
            return False
    return True

def generate_state_space(initial_level):
    print("Generating state space...")
    state_space = set()
    wall_positions = set()
    goal_positions = set()
    box_positions = set()

    for y, row in enumerate(initial_level):
        for x, cell in enumerate(row):
            if cell == "#":
                wall_positions.add((y, x))
            elif cell == ".":
                goal_positions.add((y, x))
            elif cell == "$":
                box_positions.add((y, x))
            elif cell == "@":
                player_position = (y, x)

    non_wall_positions = [
        (y, x)
        for y, row in enumerate(initial_level)
        for x, cell in enumerate(row)
        if cell != "#"
    ]
    box_combinations = itertools.combinations(
        non_wall_positions, len(box_positions)
    )

    for box_comb in box_combinations:
        for player_pos in non_wall_positions:
            if player_pos not in box_comb:
                level_state = []
                for y, row in enumerate(initial_level):
                    new_row = []
                    for x, cell in enumerate(row):
                        pos = (y, x)
                        if pos in wall_positions:
                            new_row.append("#")
                        elif pos in goal_positions and pos in box_comb:
                            new_row.append("*")
                        elif pos in goal_positions and pos == player_pos:
                            new_row.append("+")
                        elif pos in goal_positions:
                            new_row.append(".")
                        elif pos == player_pos:
                            new_row.append("@")
                        elif pos in box_comb:
                            new_row.append("$")
                        else:
                            new_row.append(" ")
                    level_state.append(tuple(new_row))

                state_space.add(tuple(level_state))
    print("Finished generating state space.")
    return state_space