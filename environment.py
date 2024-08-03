import numpy as np

class RoboticArmEnv:
    def __init__(self, grid_size=(5, 5), target_position=(4, 4)):
        self.grid_size = grid_size
        self.state_size = grid_size
        self.action_size = 4  # Up, Down, Left, Right
        self.target_position = target_position
        self.reset()

    def reset(self):
        self.position = (0, 0)
        return self.position

    def step(self, action):
        row, col = self.position
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.grid_size[0] - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Right
            col = min(col + 1, self.grid_size[1] - 1)

        self.position = (row, col)
        reward = 1 if self.position == self.target_position else -0.1
        done = self.position == self.target_position
        return self.position, reward, done
