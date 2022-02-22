import yaml
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from simulator import Simulator
from models import BeliefStateUpdateModel, TransitionModel, ObservationModel, RewardModel


class DataSet:
    def __init__(
            self,
            capacity: int,
            obs_type=np.float32,
            action_type=np.float32,
    ):
        self._observation = np.zeros((capacity, 1), dtype=obs_type)
        self._action = np.zeros((capacity, 1), dtype=action_type)
        self._obs_type = obs_type
        self._action_type = action_type

    @property
    def observation(self):
        return self._observation

    @property
    def action(self):
        return self._action

    def append(self, index, obs, action):
        self._observation[index] = obs
        self._action[index] = action

class WorldModel(nn.Module):
    def __init__(self, env, device, config):
        super().__init__()
        self._env = env
        self._device = device
        self._horizon = 8
        self._capacity = 5000
        self._data_set = DataSet(capacity=self._capacity)
        self._train_step = self._capacity//self._horizon
        self._category_size = 30
        self._layers = 1
        self._node_size = 64
        self._learning_rate = 2e-4

        probs = np.ones(self._category_size)/self._category_size
        self._prev_belief_state = td.Categorical(torch.tensor(probs, dtype=torch.float32)).probs.to(self._device)    # init_belief_state
        self._one_hot_state = torch.tensor(np.identity(n=self._category_size), dtype=torch.float32).to(self._device)
        self._initialize()

    def _initialize(self):
        self.BeliefStateUpdateModel = BeliefStateUpdateModel(category_size = self._category_size,       # b_t=f(b_{t-1}, a_{t-1}, o_t)
                                                             action_size = 1,
                                                             obs_size = 1,
                                                             layers = self._layers,
                                                             node_size = self._node_size
                                                             ).to(self._device)

        self.TransitionModel = TransitionModel(state_size = self._category_size,                        # q(s_t|s_{t-1}, a_{t-1})
                                               action_size = 1,
                                               category_size = self._category_size,
                                               layers = self._layers,
                                               node_size = self._node_size
                                               ).to(self._device)

        self.ObservationModel = ObservationModel(state_size = self._category_size,                      # p(o_t|s_t)
                                                 obs_size = 1,
                                                 layers = self._layers,
                                                 node_size = self._node_size
                                                 ).to(self._device)

        self.RewardModel = RewardModel(state_size = self._category_size,                                # w(r_t|s_t)
                                       layers = 1,
                                       node_size= self._node_size,
                                       ).to(self._device)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def fill_dataset(self):
        for i in range(self._capacity):
            obs = self._env.step(action = 0)
            action = np.array([0])
            self._data_set.append(i, obs, action)

    def model_train(self):
        obs = torch.tensor(self._data_set.observation, dtype=torch.float32).to(self._device)
        actions = torch.tensor(self._data_set.action, dtype=torch.float32).to(self._device)
        for i in range(self._train_step):
            print(f"train_step:{i}")
            observation = obs[i * self._horizon : (i + 1) * self._horizon]
            action = actions[i * self._horizon : (i + 1) * self._horizon]
            model_loss, belief_state = self.model_loss(observation, action)
            print(f"loss:{model_loss} \nbelief_state:{belief_state}\n")
            self._optimizer.zero_grad()
            model_loss.backward()
            self._optimizer.step()
        print("fin")

    def model_loss(self, obs, actions):
        total_loss = torch.tensor(0, dtype=torch.float32).to(self._device)
        for t in range(self._horizon):
            belief_state_dist = self.BeliefStateUpdateModel(self._prev_belief_state, actions[t], obs[t])    # b_{t-1}, a_{t-1}, o_t
            belief_state = belief_state_dist.probs

            first_loss_term = self._first_loss_term(belief_state, obs[t])
            second_loss_term = self._second_loss_term(self._prev_belief_state, belief_state, actions[t])
            print(first_loss_term, second_loss_term)
            total_loss += (first_loss_term + second_loss_term)

            self._prev_belief_state = belief_state.detach()
        return total_loss, belief_state

    def _first_loss_term(self, belief_state, obs):
        loss = torch.tensor(0, dtype=torch.float32).to(self._device)
        for n in range(self._category_size):
            obs_dist = self.ObservationModel(self._one_hot_state[n])
            loss += -(belief_state[n] * obs_dist.log_prob(obs).squeeze(0))
        return loss

    def _second_loss_term(self, prev_bf, current_bf, actions):
        loss = torch.tensor(0, dtype=torch.float32).to(self._device)
        for n in range(self._category_size):
            transition = torch.tensor(0, dtype=torch.float32).to(self._device)
            for k in range(self._category_size):
                state_dist = self.TransitionModel(self._one_hot_state[k], actions)
                transition +=  prev_bf[k] * state_dist.probs[n]
            loss += current_bf[n] * torch.log(current_bf[n]/transition)
        return loss

    def _reward_loss_term(self):
        loss = torch.tensor(0, dtype=torch.float32).to(self._device)
        for n in range(self._category_size):
            reward = self.RewardModel(self._one_hot_state[n])
        return loss


def main():
    # load config file
    with open('config.yml') as f:
        try:
            config_file = yaml.safe_load(f)
            print(config_file)
        except yaml.YAMLError as exc:
            print(exc)
    config = config_file['default']

    # GPU settings
    if torch.cuda.is_available() and config['gpu']:
        device = torch.device('cuda')
        torch.cuda.manual_seed(config['seed'])
    else:
        device = torch.device('cpu')

    # env(sim) settings
    sim = Simulator(pattern= (5, 3), rand=False)
    # # sim test
    # for _ in range(20):
    #     print(sim.step(0), end=' ')

    # WorldModel train and imagine
    model = WorldModel(env=sim, device=device, config=config)
    model.fill_dataset()
    model.model_train()

if __name__ == '__main__':
    main()



