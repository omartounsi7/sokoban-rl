import tkinter as tk

class Sokoban:
    def __init__(self, master, level_file):
        self.master = master
        self.master.title("Sokoban")
        self.level_file = level_file
        self.load_level()
        self.create_widgets()
        self.bind_keys()
        self.game_over = False

    def load_level(self):
        with open(self.level_file, 'r') as file:
            lines = file.readlines()
        self.level = [list(line.rstrip('\n')) for line in lines]
        self.initial_level = [row.copy() for row in self.level]
        self.height = len(self.level)
        self.width = max(len(line) for line in self.level)
        self.find_player()

    def reset_level(self):
        self.level = [row.copy() for row in self.initial_level]
        self.find_player()
        self.game_over = False
        self.message_label.config(text="")
        self.draw_game()

    def find_player(self):
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                if char == '@' or char == '+':
                    self.player_pos = (x, y)
                    return

    def create_widgets(self):
        self.title_label = tk.Label(self.master, text="Sokoban", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        self.canvas = tk.Canvas(self.master, width=450, height=450)
        self.canvas.pack()

        self.reset_button = tk.Button(self.master, text="Reset", command=self.reset_level)
        self.reset_button.pack(pady=5)

        self.message_label = tk.Label(self.master, text="", font=("Helvetica", 14))
        self.message_label.pack(pady=5)

        self.draw_game()

    def bind_keys(self):
        self.master.bind("<Key>", self.key_pressed)

    def key_pressed(self, event):
        if self.game_over:
            return
        action = self.get_action(event.keysym)
        if action:
            self.execute_action(action)
            self.draw_game()
            self.check_win()

    def get_action(self, key):
        key_mapping = {
            'w': (0, -1),
            'a': (-1, 0),
            's': (0, 1),
            'd': (1, 0),
            'Up': (0, -1),
            'Left': (-1, 0),
            'Down': (0, 1),
            'Right': (1, 0)
        }
        return key_mapping.get(key, None)

    def execute_action(self, action):
        dx, dy = action
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        if self.is_wall(nx, ny):
            return

        if self.is_box(nx, ny):
            if not self.move_box(nx, ny, dx, dy):
                return

        # Update player's current position
        if self.level[y][x] == '@':
            self.level[y][x] = ' '
        elif self.level[y][x] == '+':  # Player was on a goal
            self.level[y][x] = '.'

        # Update new player's position
        if self.level[ny][nx] == '.':
            self.level[ny][nx] = '+'
        else:
            self.level[ny][nx] = '@'

        self.player_pos = (nx, ny)

    def is_wall(self, x, y):
        return self.level[y][x] == '#'

    def is_box(self, x, y):
        return self.level[y][x] in ['$','x']

    def move_box(self, x, y, dx, dy):
        nx, ny = x + dx, y + dy
        if self.level[ny][nx] in [' ', '.']:
            # Update box's current position
            if self.level[y][x] == '$':
                self.level[y][x] = ' '
            elif self.level[y][x] == 'x':
                self.level[y][x] = '.'

            # Update box's new position
            if self.level[ny][nx] == '.':
                self.level[ny][nx] = 'x'
            else:
                self.level[ny][nx] = '$'
            return True
        return False

    def draw_game(self):
        self.canvas.delete("all")
        size = 40  # Size of each tile
        offset_x = 25  # Offset to center the game
        offset_y = 25
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                x1 = offset_x + x * size
                y1 = offset_y + y * size
                x2, y2 = x1 + size, y1 + size
                if char == '#':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray")
                elif char == '.':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                elif char == '$':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_rectangle(x1+10, y1+10, x2-10, y2-10, fill="brown")
                elif char == 'x':  # Box on goal
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_rectangle(x1+10, y1+10, x2-10, y2-10, fill="green")
                elif char == '@':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill="blue")
                elif char == '+':  # Player on goal
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill="blue")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

    def check_win(self):
        for row in self.level:
            if '$' in row:
                return False
        self.game_over = True
        self.message_label.config(text="You won!")
        return True