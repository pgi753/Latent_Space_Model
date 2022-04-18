import numpy as np
from typing import List, Tuple
import torch


class Buffer:
    def __init__(self, capacity: int, observ_shape: Tuple[int], action_shape: Tuple[int], seq_len: int, batch_size: int):
        self.capacity = capacity
        self.observ_shape = observ_shape
        self.action_shape = action_shape
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.observ = np.zeros((capacity, *observ_shape), dtype=np.int32)
        self.action = np.zeros((capacity, *action_shape), dtype=np.int32)
        self.reward = np.zeros((capacity,), dtype=np.float32)

    def add(self, observ: List[np.ndarray], action: List[np.ndarray], reward: List[float]):
        for o, a, r in zip(observ, action, reward):
            self.observ[self.idx] = o
            self.action[self.idx] = a
            self.reward[self.idx] = np.array(r, dtype=np.float32)
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.capacity
            valid_idx = not self.idx in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        observ = self.observ[vec_idxs]
        action = self.action[vec_idxs]
        return observ.reshape(l, n, *self.observ_shape), action.reshape(l, n, *self.action_shape), self.reward[vec_idxs].reshape(l, n)

    def sample(self):
        n = self.batch_size
        l = self.seq_len + 1
        obs, act, rew = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs, act, rew = self._shift_sequences(obs, act, rew)
        return obs, act, rew

    def _shift_sequences(self, obs, actions, rewards):
        obs = obs[1:]
        actions = actions[:-1]
        rewards = rewards[1:]
        return obs, actions, rewards
