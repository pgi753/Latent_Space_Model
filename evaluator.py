import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from simulator import Simulator
from models import POMDPModel
from buffer import Buffer
from pathlib import Path


class Evaluator:
    def __init__(self, args):
        self._state_cls_size = args.state_cls_size
        self._state_cat_size = args.state_cat_size
        self._action_shape = (args.action_size,)
        self._observ_shape = (args.observ_size,)
        self._rnn_input_size = args.rnn_input_size
        self._rnn_hidden_size = args.rnn_hidden_size
        self._horizon = args.horizon
        self._wm_lr = args.wm_lr
        self._actor_lr = args.actor_lr
        self._value_lr = args.value_lr
        self._lambda = args.lambda_
        self._actor_entropy_scale = args.actor_entropy_scale
        self._discount = args.discount
        if torch.cuda.is_available() and args.device:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self._simulator = Simulator(pattern_fixed=False, max_period = 20, cycle_range = (50, 150), seed = 123)
        self._pomdp = POMDPModel(self._state_cls_size, self._state_cat_size, self._action_shape, self._observ_shape,
                                 self._rnn_input_size, self._rnn_hidden_size, self._device,
                                 self._wm_lr, self._actor_lr, self._value_lr, self._lambda, self._actor_entropy_scale,
                                 self._discount)
        self._pomdp.load_model()

    def step(self, num_steps = 1):
        action = self._pomdp.prev_action
        observ_list = []
        reward_list = []
        action_list = []
        for i in range(num_steps):
            observ, reward = self._simulator.step(action)
            action = self._pomdp.step(observ)
            observ_list.append(observ)
            reward_list.append(reward)
            action_list.append(action)
        return observ_list, reward_list, action_list

    def test(self, test_steps:int = 1000):
        print(f"observ, action, reward")
        for _ in range(test_steps):
            obs, rew, act = evaluator.step()
            print(f"{np.argmax(obs, axis=-1)}, {np.argmax(act, axis=-1)}, {np.round(rew)}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--state_cls_size', type=int, default=1, help='State class size')
    parser.add_argument('--state_cat_size', type=int, default=64, help='State category size')
    parser.add_argument('--action_size', type=int, default=2, help='Action size')
    parser.add_argument('--observ_size', type=int, default=2, help='Observation size')
    parser.add_argument('--rnn_input_size', type=int, default=8, help='RNN input size')
    parser.add_argument('--rnn_hidden_size', type=int, default=64, help='RNN hidden size')
    parser.add_argument('--wm_lr', type=float, default=0.003, help='World model Learning rate')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='Action model Learning rate')
    parser.add_argument('--value_lr', type=float, default=0.0005, help='Value model Learning rate')
    parser.add_argument('--horizon', type=int, default=100, help='Horizon length')
    parser.add_argument('--lambda_', type=float, default=0.9, help='TD lambda')
    parser.add_argument('--actor_entropy_scale', type=float, default=0.001, help='Actor entropy scale')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.test(test_steps = 100000)





