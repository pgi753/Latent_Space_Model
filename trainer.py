import numpy as np
import torch
import pandas as pd
from simulator import Simulator, Simulator_Markovian
from models import POMDPModel
from buffer import Buffer
from pathlib import Path


class Trainer:
    def __init__(self, config):
        self._state_cls_size = config['pomdp_model']['state_cls_size']
        self._state_cat_size = config['pomdp_model']['state_cat_size']
        self._action_shape = (config['pomdp_model']['action_size'],)
        self._observ_shape = (config['pomdp_model']['observ_size'],)
        self._rnn_input_size = config['pomdp_model']['rnn_input_size']
        self._rnn_hidden_size = config['pomdp_model']['rnn_hidden_size']
        self._seq_len = config['pomdp_model']['seq_len']
        self._batch_size = config['pomdp_model']['batch_size']
        self._horizon = config['pomdp_model']['horizon']
        self._wm_lr = config['pomdp_model']['wm_lr']
        self._actor_lr = config['pomdp_model']['actor_lr']
        self._value_lr = config['pomdp_model']['value_lr']
        self._lambda = config['pomdp_model']['lambda']
        self._actor_entropy_scale = config['pomdp_model']['actor_entropy_scale']
        self._discount = config['pomdp_model']['discount']
        self._kld_scale = config['pomdp_model']['kld_scale']
        self._value_mix_rate = config['pomdp_model']['value_mix_rate']
        self._update_interval = config['pomdp_model']['update_interval']
        if torch.cuda.is_available() and config['pomdp_model']['device']:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self._buffer = Buffer(1000000, self._observ_shape, self._action_shape, self._seq_len, self._batch_size)
        self._model_path = str(Path(__file__).parent.resolve() / 'SavedModels' /config['files']['model_name'])
        self._pomdp = POMDPModel(self._state_cls_size, self._state_cat_size, self._action_shape, self._observ_shape,
                                 self._rnn_input_size, self._rnn_hidden_size, self._device,
                                 self._wm_lr, self._actor_lr, self._value_lr, self._lambda, self._actor_entropy_scale,
                                 self._discount, self._kld_scale)
        self._set_model()

        self._rendering = config['env']['rendering']
        self._video_path = Path(__file__).parent.resolve() / 'Results' / config['files']['video_name']
        self._pattern_fixed = config['env']['pattern_fixed']
        self._pomdp_mode = config['env']['pomdp_mode']
        self._max_period = config['env']['max_period']
        self._cycle_range = tuple((config['env']['cycle_range']['s'], config['env']['cycle_range']['e']))
        self._seed = config['env']['seed']
        if config['env']['env_name'] == 'Simulator':
            self._simulator = Simulator(rendering=self._rendering, path=self._video_path, pattern_fixed=self._pattern_fixed,
                                        pomdp_mode=self._pomdp_mode, max_period=self._max_period, cycle_range=self._cycle_range,
                                        seed=self._seed)
        elif config['env']['env_name'] == 'Simulator_Markovian':
            self._simulator = Simulator_Markovian(rendering=self._rendering, path=self._video_path)

        self._metrics_path = Path(__file__).parent.resolve() / 'Results' / config['files']['metrics_name']
        self._train_metrics = {'wm_loss': [], 'a_loss': [], 'v_loss': []}

    def _set_model(self):
        if Path(self._model_path).exists():
            self._pomdp.load_model(self._model_path)
        else:
            self._pomdp.save_model(self._model_path)

    def save_train_metrics(self):
        df = pd.DataFrame(self._train_metrics)
        df.to_csv(self._metrics_path, encoding='cp949')

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
        wm_loss = []
        a_loss = []
        v_loss = []

        # Dynamics learning
        for step in range(1, num_step + 1):
            rnn_hidden, kld_loss, observ_loss, model_loss, reward_loss = self._pomdp.world_model_loss(observ, action, reward)
            self._pomdp.world_model_optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._pomdp.world_model_parameters, max_norm=1)
            self._pomdp.world_model_optimizer.step()
            wm_loss.append(model_loss.item())

            if step % 5 == 0:
                print(f"{step}/{num_step}  ", f"Model loss: {model_loss.item():.6f} ",
                      f"KLD loss: {kld_loss.item():.6f} ", f"Observation loss: {observ_loss.item():.6f} ",
                      f"Reward loss: {reward_loss.item():.6f} ")

        # imagine test
        o, a, r = self._pomdp.imagine_test(rnn_hidden, self._horizon)
        print(f"imagine test \nobserv:{o[0]} \naction:{a[0]} \nreward:{r[0]}")
        self._pomdp.save_model(self._model_path)

        # Behavior learning
        for step in range(1, (num_step + 1)):
            actor_loss, value_loss = self._pomdp.actor_critic_loss(rnn_hidden, self._horizon)

            self._pomdp.actor_optimizer.zero_grad()
            self._pomdp.value_model_optimizer.zero_grad()
            actor_loss.backward()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._pomdp.actor_parameters, max_norm=1)
            torch.nn.utils.clip_grad_norm_(self._pomdp.value_model_parameters, max_norm=1)
            self._pomdp.actor_optimizer.step()
            self._pomdp.value_model_optimizer.step()
            a_loss.append(actor_loss.item())
            v_loss.append(value_loss.item())

            if step % self._update_interval == 0:
                self._pomdp.update_target_value(self._value_mix_rate)

            if step % 5 == 0:
                print(f"{step}/{num_step}  ", f"Actor loss: {actor_loss.item():.6f} ", f"Value loss: {value_loss.item():.6f} ")

            if step % 100 == 0:
                # imagine test
                o, a, r = self._pomdp.imagine_test(rnn_hidden, self._horizon)
                print(f"imagine test \nobserv:{o[:1]} \naction:{a[:1]} \nreward:{r[:1]}")

        self._train_metrics['wm_loss'].append(np.mean(wm_loss))
        self._train_metrics['a_loss'].append(np.mean(a_loss))
        self._train_metrics['v_loss'].append(np.mean(v_loss))
