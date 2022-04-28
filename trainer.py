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


class Trainer:
    def __init__(self, args):
        self._state_cls_size = args.state_cls_size
        self._state_cat_size = args.state_cat_size
        self._action_shape = (args.action_size,)
        self._observ_shape = (args.observ_size,)
        self._rnn_input_size = args.rnn_input_size
        self._rnn_hidden_size = args.rnn_hidden_size
        self._seq_len = args.seq_len
        self._batch_size = args.batch_size
        self._horizon = args.horizon
        self._wm_lr = args.wm_lr
        self._actor_lr = args.actor_lr
        self._value_lr = args.value_lr
        self._lambda = args.lambda_
        self._actor_entropy_scale = args.actor_entropy_scale
        self._discount = args.discount
        self._value_mix_rate = args.value_mix_rate
        self._update_interval = args.update_interval
        if torch.cuda.is_available() and args.device:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self._simulator = Simulator(pattern_fixed=True, max_period = 20, cycle_range = (500, 1000), seed = 0)
        self._pomdp = POMDPModel(self._state_cls_size, self._state_cat_size, self._action_shape, self._observ_shape,
                                 self._rnn_input_size, self._rnn_hidden_size, self._device,
                                 self._wm_lr, self._actor_lr, self._value_lr, self._lambda, self._actor_entropy_scale,
                                 self._discount)
        self._buffer = Buffer(1000000, self._observ_shape, self._action_shape, self._seq_len, self._batch_size)
        self._set_model()

    def _set_model(self):
        path = str(Path(__file__).parent.resolve() / 'models') + '\model.pt'
        if Path(path).exists():
            self._pomdp.load_model()
        else:
            self._pomdp.save_model()

    def add_to_buffer(self, observ, action, reward):
        self._buffer.add(observ, action, reward)

    def sample_buffer(self):
        return self._buffer.sample()

    def reset(self):
        self._simulator.reset()
        self._pomdp.reset(initial_action=self._simulator.sample_action())

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

    def train(self, observ, action, reward, num_step):
        observ = torch.tensor(observ, dtype=torch.float32).to(self._device)
        action = torch.tensor(action, dtype=torch.float32).to(self._device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self._device)

        # Dynamics learning
        for step in range(1, num_step + 1):
            rnn_hidden, kld_loss, observ_loss, model_loss, reward_loss = self._pomdp.world_model_loss(observ, action, reward)

            self._pomdp.world_model_optimizer.zero_grad()
            model_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self._pomdp.world_model_parameters, max_norm=0.1)
            self._pomdp.world_model_optimizer.step()

            if step % 10 == 0:
                print(f"{step}/{num_step}  ", f"Model loss: {model_loss.item():.6f} ",
                      f"KLD loss: {kld_loss.item():.6f} ", f"Observation loss: {observ_loss.item():.6f} ",
                      f"Reward loss: {reward_loss.item():.6f} ")
        # imagine test
        o, a, r = self._pomdp.imagine_test(rnn_hidden, self._horizon)
        print(f"imagine test \nobserv:{o[0]} \naction:{a[0]} \nreward:{r[0]}")
        # self._pomdp.save_model()


        # Behavior learning
        for step in range(1, (num_step + 1)):
            actor_loss, value_loss = self._pomdp.actor_critic_loss(rnn_hidden, self._horizon)

            self._pomdp.actor_optimizer.zero_grad()
            self._pomdp.value_model_optimizer.zero_grad()
            actor_loss.backward()
            value_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self._pomdp.actor_parameters, 1)
            # torch.nn.utils.clip_grad_norm_(self._pomdp.value_model_parameters, 0.1)
            self._pomdp.actor_optimizer.step()
            self._pomdp.value_model_optimizer.step()

            if step % self._update_interval == 0:
                self._pomdp.update_target_value(self._value_mix_rate)

            if step % 10 == 0:
                print(f"{step}/{num_step}  ", f"Actor loss: {actor_loss.item():.6f} ", f"Value loss: {value_loss.item():.6f} ")

            if step % 100 == 0:
                # imagine test
                o, a, r = self._pomdp.imagine_test(rnn_hidden, self._horizon)
                print(f"imagine test \nobserv:{o[:1]} \naction:{a[:1]} \nreward:{r[:1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=400, help='Sequence length')
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
    parser.add_argument('--lambda_', type=float, default=0.8, help='TD lambda')
    parser.add_argument('--actor_entropy_scale', type=float, default=0.001, help='Actor entropy scale')
    parser.add_argument('--value_mix_rate', type=float, default=1, help='Update value model mix rate, range: 0 ~ 1')
    parser.add_argument('--update_interval', type=int, default=10, help='Value model slow update')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    args = parser.parse_args()
    np.set_printoptions(threshold=100000, linewidth=100000)

    trainer = Trainer(args)
    trainer.reset()

    # collect episode
    obs, rew, act = trainer.step(1000)
    trainer.add_to_buffer(obs, act, rew)

    train_steps = 10000
    for iter in range(train_steps):
        print(f"-------------------train_step: {iter}-------------------")
        # train Dynamics and Behavior
        observ, action, reward = trainer.sample_buffer()
        trainer.train(observ, action, reward, 500)

        # environment interaction
        obs, rew, act = trainer.step(40)
        print(f"environment interaction")
        print(f"observ : {np.argmax(obs, axis= -1)}")
        print(f"action : {np.argmax(act, axis= -1)}")
        trainer.add_to_buffer(obs, act, rew)






