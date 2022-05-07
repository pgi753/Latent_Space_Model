import random
import numpy as np
from visualization import Visualization


class Simulator:
    def __init__(self, path: str, pattern_fixed: bool = False, pomdp: bool = False, max_period: int = 20,
                 cycle_range: tuple = (500, 1000), seed: int = 0):
        self._pattern_fixed = pattern_fixed
        self._pomdp_mode = pomdp
        self._max_period = max_period
        self._cycle_range = cycle_range
        self._visualization = Visualization(video_file_path=path, freq_channel_list=[0],
                                            network_operator_list=['agent', 'user1'])
        random.seed(seed)
        self._set_pattern()
        self._ch_state = 0
        self._pointer = 0
        self._log ={}

    def _set_pattern(self):
        while True:
            self._using = random.randint(1, self._max_period)
            self._empty = random.randint(0, self._max_period)
            if self._using + self._empty <= self._max_period:
                break
        self._pattern = [0] * self._empty + [1] * self._using       # [0]=empty, [1]=using
        self._period = len(self._pattern)
        self._cycles = random.randint(self._cycle_range[0], self._cycle_range[1])
        if self._pattern_fixed:
            print(f"pattern: {self._empty, self._using}, period: {self._period}, pattern fixed")
        else:
            print(f"pattern: {self._empty, self._using}, period: {self._period}, cycles: {self._cycles}, total: {self._period*self._cycles}")

    def step(self, action):
        # observation
        self._ch_state = self._pattern[self._pointer]
        if self._pomdp_mode:
            obs_prob = random.random()
            if obs_prob <= 0.9:
                obs = np.array([1, 0]) if self._ch_state == 0 else np.array([0, 1])
            else:
                obs = np.array([0, 1]) if self._ch_state == 0 else np.array([1, 0])
        else:
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

        self._log = {'channel info': {'collision': {'freq channel': [0], 'packet': 0},
                                      'agent':{'freq channel': [0], 'packet': np.argmax(action)},
                                      'user1':{'freq channel': [0], 'packet': self._ch_state}},
                     'reward': reward,
                     'pattern': (self._empty, self._using)}
        self._visualization(self._log)

        # pattern change
        if not self._pattern_fixed and self._cycles <= 0:
            self._set_pattern()
        return obs, reward

    def sample_action(self):
        action = np.array([1, 0]) if np.random.randint(2) == 0 else np.array([0, 1])
        return action

    def reset(self):
        # self._set_pattern()
        self._pointer = 0
