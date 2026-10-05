"""
gridworld_env.py — the "Gladiator" arena: a custom 2-D grid environment built with NumPy.

Tiles
  S = start      G = goal      # = wall (impassable)      T = trap      . = floor

Rules (see README for the full table)
  * The agent moves up / right / down / left, one tile per step.
  * Every step costs -1, so shorter paths score higher.
  * Walking into a wall or the arena edge costs -5 and the agent stays put.
  * Stepping on a trap costs -100 and ends the episode (a loss).
  * Reaching the goal gives +100 and ends the episode (a win).
  * After MAX_STEPS moves the episode times out (a loss).
"""
from collections import deque

import numpy as np

ARENA = [
    "S..#......",
    ".#.#.###T.",
    ".#.....#..",
    ".###T#.#.#",
    "...#.#....",
    "##.#.###T.",
    "....T...#.",
    ".######.#.",
    "..T....#..",
    "###.#T...G",
]

ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

REWARD_STEP = -1
REWARD_WALL = -5
REWARD_TRAP = -100
REWARD_GOAL = 100
MAX_STEPS = 100


class GladiatorGridWorld:
    def __init__(self, layout=ARENA, max_steps=MAX_STEPS):
        self.grid = np.array([list(row) for row in layout])
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(ACTIONS)
        self.max_steps = max_steps
        self.start = tuple(np.argwhere(self.grid == "S")[0])
        self.goal = tuple(np.argwhere(self.grid == "G")[0])
        self.pos = self.start
        self.steps = 0

    # state <-> (row, col) helpers
    def to_state(self, pos):
        return pos[0] * self.n_cols + pos[1]

    def to_pos(self, state):
        return divmod(state, self.n_cols)

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.to_state(self.pos)

    def step(self, action):
        """Returns (next_state, reward, done, event)."""
        self.steps += 1
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc

        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or self.grid[r, c] == "#":
            reward, event = REWARD_WALL, "wall"          # bumped: stay in place
        else:
            self.pos = (r, c)
            tile = self.grid[r, c]
            if tile == "T":
                return self.to_state(self.pos), REWARD_TRAP, True, "trap"
            if tile == "G":
                return self.to_state(self.pos), REWARD_GOAL, True, "goal"
            reward, event = REWARD_STEP, "step"

        if self.steps >= self.max_steps:
            return self.to_state(self.pos), reward, True, "timeout"
        return self.to_state(self.pos), reward, False, event

    def shortest_path_length(self):
        """Breadth-first search: the true minimum number of moves from S to G
        avoiding walls and traps. Used to measure the agent's path efficiency."""
        queue, seen = deque([(self.start, 0)]), {self.start}
        while queue:
            (r, c), d = queue.popleft()
            if (r, c) == self.goal:
                return d
            for dr, dc in ACTIONS.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols
                        and self.grid[nr, nc] not in "#T" and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    queue.append(((nr, nc), d + 1))
        return None

    def render_ascii(self, path=None):
        """Text picture of the arena, with an optional path drawn as '*'."""
        g = self.grid.copy()
        for (r, c) in (path or []):
            if g[r, c] == ".":
                g[r, c] = "*"
        if self.pos not in (self.start, self.goal):
            g[self.pos] = "A"
        return "\n".join(" ".join(row) for row in g)
