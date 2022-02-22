import random
import numpy as np

class Simulator:
    def __init__(self, pattern: tuple = (1, 1), rand: bool = False):
        self._pattern = [0]*pattern[0] + [1]*pattern[1]     # [0]=empty, [1]=using
        self._period = len(self._pattern)
        self._rand = rand
        self._pointer = 0

    def step(self, action: int):    # action 0 or 1
        if self._rand:              # random
            ch_state=random.choice([0, 1])
            obs = (ch_state or action)

        else:                       # pattern
            ch_state = self._pattern[self._pointer]
            obs = (ch_state or action)
            self._pointer += 1
            if self._pointer >= self._period:
                self._pointer = 0
        return np.array([obs])

    def reset(self):
        self._pointer = 0