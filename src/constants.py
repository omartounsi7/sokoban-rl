# For GUI
TILE_SIZE = 40
X_OFFSET = 50
Y_OFFSET = 25
WAIT_TIME = 500  # in milliseconds

# Rewards
SUPERMALUS = -10
MALUS = -1
STEP_REWARD = -0.01
BONUS = 10
SUPERBONUS = 100

# Hyperparameters
MAX_EPISODES_MC = 100000
MAX_EPISODES_PG = 1000
MAX_STEP = 1000
GAMMA = 0.99
EPSILON = 0.9
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 500 # for monte carlo convergence
BEST_REWARD_THRESHOLD = SUPERBONUS * 0.8 # for policy gradient convergence

# For MDP
ACTION_SPACE = {0: "up", 1: "left", 2: "down", 3: "right"}
NUMERIC_ACTION_SPACE= {v: k for k, v in ACTION_SPACE.items()} # don't remove this
ACTION_MAP = {
    0: (0, -1),  # Up
    1: (-1, 0),  # Left
    2: (0, 1),  # Down
    3: (1, 0),  # Right
}