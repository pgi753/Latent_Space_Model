import torch
import pandas as pd
from simulator import Simulator, Simulator_Markovian
from models import POMDPModel
from pathlib import Path


class Evaluator:
    def __init__(self, config):
        # pomdp model
        self._state_cls_size = config['pomdp_model']['state_cls_size']
        self._state_cat_size = config['pomdp_model']['state_cat_size']
        self._state_sample_size = config['pomdp_model']['state_sample_size']
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
        if torch.cuda.is_available() and config['pomdp_model']['device']:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self._model_path = str(Path(__file__).parent.resolve() / 'SavedModels' /config['files']['model_name'])
        self._pomdp = POMDPModel(self._state_cls_size, self._state_cat_size, self._state_sample_size, self._action_shape,
                                 self._observ_shape, self._rnn_input_size, self._rnn_hidden_size, self._seq_len, self._batch_size,
                                 self._wm_lr, self._actor_lr, self._value_lr, self._lambda, self._actor_entropy_scale,
                                 self._discount, self._kld_scale, self._device)
        self._pomdp.load_model(self._model_path)

        # env
        self._rendering = config['env']['rendering']
        self._video_path = Path(__file__).parent.resolve() / 'Results' / 'test_result.mp4'
        self._pattern_fixed = config['env']['pattern_fixed']
        self._pomdp_mode = config['env']['pomdp_mode']
        self._max_period = config['env']['max_period']
        self._cycle_range = tuple((config['env']['cycle_range']['s'], config['env']['cycle_range']['e']))
        self._seed = config['env']['seed']
        if config['env']['env_name'] == 'Simulator':
            self._simulator = Simulator(rendering=False, path=self._video_path,
                                        pattern_fixed=self._pattern_fixed,
                                        pomdp_mode=self._pomdp_mode, max_period=self._max_period,
                                        cycle_range=self._cycle_range,
                                        seed=self._seed)
        elif config['env']['env_name'] == 'Simulator_Markovian':
            self._simulator = Simulator_Markovian(rendering=False, path=self._video_path)

        self._result_path = Path(__file__).parent.resolve() / 'Results' /  config['files']['result_name']
        self._result = {'total_reward': [], 'reward_rate':[], 'success_rate':[], 'failure_rate':[]}

    def reset(self):
        self._simulator.reset()
        self._pomdp.reset(initial_action=self._simulator.sample_action())

    def save_result(self):
        df = pd.DataFrame(self._result)
        df.to_csv(self._result_path, encoding='cp949')

    def step(self, num_steps = 1):
        action = self._pomdp.prev_action
        observ_list = []
        reward_list = []
        action_list = []
        total_reward = 0
        optimal_reward = 0
        success = 0
        failure = 0

        for i in range(num_steps):
            observ, reward = self._simulator.step(action)
            action = self._pomdp.step(observ)
            total_reward += reward
            if reward >= 0:
                success += 1
            else:
                failure += 1

            if observ[0] == 1:
                optimal_reward += 3
            else:
                optimal_reward += 1
            observ_list.append(observ)
            reward_list.append(reward)
            action_list.append(action)

        reward_rate = total_reward / optimal_reward
        success_rate = success / num_steps
        failure_rate = failure / num_steps
        self._result['total_reward'].append(total_reward)
        self._result['reward_rate'].append(reward_rate)
        self._result['success_rate'].append(success_rate)
        self._result['failure_rate'].append(failure_rate)
        return observ_list, reward_list, action_list
