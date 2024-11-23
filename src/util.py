import itertools
from src.constants import *

def count_placed_boxes(state):
    return sum(row.count("x") for row in state)

def get_action(key):
    return KEYMAPPING.get(key.lower(), None)

def serialize_state(state):
    return tuple(tuple(row) for row in state)

def print_state(state):
    for row in state:
        print(row)

def find_player_in_state(state):
    for y, row in enumerate(state):
        for x, cell in enumerate(row):
            if cell in ('@', '+'):
                return (x, y)
    return None

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
                            new_row.append("x")
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

def execute_action(state, action):
    dx, dy = action
    x, y = find_player_in_state(state)
    nx, ny = x + dx, y + dy
    
    reward = MALUS
    new_position = (x, y)
    moved_box = False

    if is_wall(state, nx, ny):
        return new_position, reward, moved_box

    if is_box(state, nx, ny):
        if not move_box(state, nx, ny, dx, dy):
            return new_position, reward, moved_box
        moved_box = True
    
    move_agent(state, x, y, nx, ny)
    new_position = (nx, ny)
    reward = STEPREWARD

    if moved_box:
        bx, by = nx + dx, ny + dy
        if is_box_stuck(state, bx, by):
            reward = SUPERMALUS
        elif check_win(state):
            reward = SUPERBONUS
        elif is_box_placed(state, bx, by):
            reward = BONUS

    return new_position, reward, moved_box

def is_wall(state, x, y):
    return state[y][x] == "#"

def is_box(state, x, y):
    return state[y][x] in ["$", "x"]

def is_goal(state, x, y):
    return state[y][x] == "."

def is_box_placed(state, x, y):
    return state[y][x] in ["x"]

def is_position_free(state, x, y):
    return state[y][x] in [' ', '.']

def is_obstacle(state, x, y):
    return state[y][x] in ["#", "$", "x"]

def is_box_stuck(state, x, y):
    if is_box_placed(state, x, y):
        return False

    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for i in range(4):
        dx1, dy1 = dirs[i]
        dx2, dy2 = dirs[(i + 1) % 4]
        if is_obstacle(state, x + dx1, y + dy1) and is_obstacle(state, x + dx2, y + dy2):
            return True

    return False

def check_win(state):
    for row in state:
        if '$' in row:
            return False
    return True

def move_agent(state, x, y, nx, ny):
    if state[y][x] == "@":
        state[y][x] = " "
    elif state[y][x] == "+":
        state[y][x] = "."

    if state[ny][nx] == ".":
        state[ny][nx] = "+"
    else:
        state[ny][nx] = "@"

def move_box(state, x, y, dx, dy):
    nx, ny = x + dx, y + dy
    if state[ny][nx] in [" ", "."]:
        if state[y][x] == "$":
            state[y][x] = " "
        elif state[y][x] == "x":
            state[y][x] = "."

        if state[ny][nx] == ".":
            state[ny][nx] = "x"
        else:
            state[ny][nx] = "$"
        return True
    return False