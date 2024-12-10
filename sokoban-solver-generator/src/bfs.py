import time
from collections import deque

import numpy as np
import pygame
import psutil
import os
import gc

from .utils import can_move, get_state, is_deadlock, is_solved, print_state

def bfs(matrix, player_pos, widget=None, visualizer=False):
	print('Breadth-First Search')
	initial_state = get_state(matrix)
	shape = matrix.shape
	print_state(initial_state, shape)
	seen = {None}
	q = deque([(initial_state, player_pos, 0, '')])
	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	curr_depth = 0
	direction = {
		(1, 0): 'D',
		(-1, 0): 'U', 
		(0, -1): 'L',
		(0, 1): 'R',
	}
	while q:
		if widget:
			pygame.event.pump()
		state, pos, depth, path = q.popleft()
		# if depth != curr_depth:
		# 	print(f'Depth: {depth}')
		# 	curr_depth = depth
		seen.add(state)
		for move in moves:
			new_state, _ = can_move(state, shape, pos, move)
			deadlock = is_deadlock(new_state, shape)
			if new_state in seen or deadlock:
				continue
			q.append((
				new_state, 
				(pos[0] + move[0], pos[1] + move[1]),
				depth + 1,
				path + direction[move],
			))
			if is_solved(new_state):
				print(f'[BFS] Solution found!\n\n{path + direction[move]}\nDepth {depth + 1}\n')
				if widget and visualizer:
					widget.solved = True
					widget.set_text(f'[BFS] Solution Found!\n{path + direction[move]}', 20)
					pygame.display.update()
				return (path + direction[move], depth + 1)
			if widget and visualizer:
				widget.set_text(f'[BFS] Solution Depth: {depth + 1}\n{path + direction[move]}', 20)
				pygame.display.update()
	print(f'[BFS] Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'[BFS] Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not q else depth + 1)


def solve_bfs(puzzle, widget=None, visualizer=False):
	matrix = puzzle
	where = np.where((matrix == '*') | (matrix == '%'))
	player_pos = where[0][0], where[1][0]

	gc.collect()
	process = psutil.Process(os.getpid())
	before = process.memory_info().rss / 1024 / 1024

	result = bfs(matrix, player_pos, widget, visualizer)

	after = process.memory_info().rss / 1024 / 1024
	memory_usage = after - before
    
	print(f"Memory Usage: {memory_usage} MB")
    
	if widget and visualizer:
		widget.set_text(
			f'[BFS] Solution Found!\nMemory Usage: {memory_usage} MB',
			20
		)

	return result

	
if __name__ == '__main__':
	start = time.time()
	process = psutil.Process(os.getpid())
	before = process.memory_info().rss / 1024 / 1024

	root = solve_bfs(np.loadtxt('levels/lvl1.dat', dtype='<U1'))

	after = process.memory_info().rss / 1024 / 1024

	print(f'Runtime: {time.time() - start} seconds')
	print(f"Total memory used: {after - before} MB")