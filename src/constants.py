TILESIZE = 40
XOFFSET = 50
YOFFSET = 25
SUPERMALUS = -10
MALUS = -1
STEPREWARD = 0
BONUS = 10
SUPERBONUS = 100
MINEPSILON = 0.1
EPSILONDECAY = 0.995
MINEPISODES = 1000
ACTIONSPACE = {0: "up", 1: "left", 2: "down", 3: "right"}
NUMERIC_ACTION_SPACE= {v: k for k, v in ACTIONSPACE.items()} # don't remove this
ACTIONMAP = {
    0: (0, -1),  # Up
    1: (-1, 0),  # Left
    2: (0, 1),  # Down
    3: (1, 0),  # Right
}
WAITTIME = 500  # in milliseconds
