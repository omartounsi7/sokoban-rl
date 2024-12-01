TILE_SIZE = 40
X_OFFSET = 50
Y_OFFSET = 25
SUPERMALUS = -10
MALUS = -1
STEP_REWARD = 0
BONUS = 10
SUPERBONUS = 100
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995
MIN_EPISODES = 1000
ACTION_SPACE = {0: "up", 1: "left", 2: "down", 3: "right"}
NUMERIC_ACTION_SPACE= {v: k for k, v in ACTION_SPACE.items()} # don't remove this
ACTION_MAP = {
    0: (0, -1),  # Up
    1: (-1, 0),  # Left
    2: (0, 1),  # Down
    3: (1, 0),  # Right
}
WAIT_TIME = 500  # in milliseconds
EARLY_STOPPING_PATIENCE = 500