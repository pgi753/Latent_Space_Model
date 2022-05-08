import random
import numpy as np
from visualization import Visualization


class Simulator:
    def __init__(self, rendering: bool = False, path: str = '', pattern_fixed: bool = False, pomdp: bool = False,
                 max_period: int = 20, cycle_range: tuple = (500, 1000), seed: int = 0):
        self._pattern_fixed = pattern_fixed
        self._pomdp_mode = pomdp
        self._max_period = max_period
        self._cycle_range = cycle_range
        self._rendering = rendering
        if self._rendering:
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
            reward = -1.0 if self._ch_state == 0 else 1.0
        else:              # packet sent
            reward = 3.0 if self._ch_state == 0 else -3.0

        # visualization
        if self._rendering:
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


import torch
import torch.distributions as td
class Simulator_Markovian:
    def __init__(self, rendering: bool = False, path: str = '', p: float = 0.7, q : float = 0.8):
        self._p = p
        self._q = q
        self.transition_matrix = np.array([[p, 1-p], [1-q, q]])
        self._ch_state = 0
        self._pointer = 0
        self._packet_length = 0
        self._discount_arr = self._q * torch.ones(31)
        self._rendering = rendering
        if self._rendering:
            self._visualization = Visualization(video_file_path=path, freq_channel_list=[0],
                                                network_operator_list=['agent', 'user1'])
            self._log = {}

    def step(self, action):
        # observation

        logit = torch.from_numpy(self.transition_matrix[self._ch_state])
        sampling_dist = td.categorical.Categorical(logit)
        self._ch_state = sampling_dist.sample()

        if self._ch_state == 1:
            self._packet_length += 1
            if self._packet_length > 30:
                self._packet_length = 30
        else:
            self._packet_length = 0

        q = self._discount_arr[self._packet_length]
        self.transition_matrix = np.array([[self._p, 1 - self._p], [1 - q, q]])


        obs = np.array([1, 0]) if self._ch_state == 0 else np.array([0, 1])
        self._pointer += 1

        # reward
        if action[0]:  # packet not sent    [1, 0]
            reward = -1.0 if self._ch_state == 0 else 1.0
        else:              # packet sent
            reward = 3.0 if self._ch_state == 0 else -3.0

        # visualization
        if self._rendering:
            self._log = {'channel info': {'collision': {'freq channel': [0], 'packet': 0},
                                          'agent':{'freq channel': [0], 'packet': np.argmax(action)},
                                          'user1':{'freq channel': [0], 'packet': self._ch_state}},
                         'reward': reward,
                         'pattern': "markov"}
            self._visualization(self._log)


        return obs, reward

    def sample_action(self):
        action = np.array([1, 0]) if np.random.randint(2) == 0 else np.array([0, 1])
        return action

    def reset(self):
        self._ch_state = np.random.randint(2)
        self._packet_length = 0
        discount_arr = torch.cat([torch.ones_like(self._discount_arr[:1]), self._discount_arr[1:]])
        self._discount_arr = torch.cumprod(discount_arr[:-1], 0)
        print('initial state', self._ch_state)
        print(self._discount_arr)