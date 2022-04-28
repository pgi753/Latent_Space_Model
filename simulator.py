import random
import numpy as np
from visualization import Visualization
from pathlib import Path


class Simulator:
    def __init__(self, pattern_fixed: bool = False, max_period: int = 20, cycle_range: tuple = (500, 1000), seed: int = 0):
        self._fixed = pattern_fixed
        self._max_period = max_period
        self._cycle_range = cycle_range
        path = Path(__file__).parent.resolve() / 'video' / 'output.mp4'
        self._visualization = Visualization(video_file_path=path, freq_channel_list=[0],
                                            network_operator_list=['agent', 'user1'])
        random.seed(seed)
        self._set_pattern()
        self._ch_state = 0
        self._pointer = 0

    def _set_pattern(self):
        while True:
            using = random.randint(1, self._max_period)
            empty = random.randint(0, self._max_period)
            if using + empty <= self._max_period:
                break
        self._pattern = [0] * empty + [1] * using       # [0]=empty, [1]=using
        self._period = len(self._pattern)
        self._cycles = random.randint(self._cycle_range[0], self._cycle_range[1])
        if self._fixed:
            print(f"pattern: {empty, using}, period: {self._period}, pattern fixed")
        else:
            print(f"pattern: {empty, using}, period: {self._period}, cycles: {self._cycles}, total: {self._period*self._cycles}")

    def step(self, action):
        # observation
        self._ch_state = self._pattern[self._pointer]
        obs = np.array([1, 0]) if self._ch_state == 0 else np.array([0, 1])
        self._pointer += 1
        if self._pointer >= self._period:
            self._pointer = 0
            self._cycles -= 1
        # reward
        if action[0]:  # packet not sent    [1, 0]
            reward = -2.0 if self._ch_state == 0 else 1.0
        else:              # packet sent
            reward = 3.0 if self._ch_state == 0 else -4.0

        log = {'channel info': {'agent':{'freq channel': [0], 'packet': np.argmax(action)},
                                'user1':{'freq channel': [0], 'packet': self._ch_state}},
               'reward': reward}
        self._visualization(log)

        # pattern change
        if not self._fixed and self._cycles <= 0:
            self._set_pattern()
        return obs, reward

    def sample_action(self):
        action = np.array([1, 0]) if np.random.randint(2) == 0 else np.array([0, 1])
        return action

    def reset(self):
        # self._set_pattern()
        self._pointer = 0
