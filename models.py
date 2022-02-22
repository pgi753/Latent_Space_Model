import torch
import torch.nn as nn
import torch.distributions as td


class BeliefStateUpdateModel(nn.Module):     # b_t=f(b_{t-1}, a_{t-1}, o_t)
    def __init__(self, category_size, action_size, obs_size, layers, node_size):
        super().__init__()
        self._input_size = category_size + action_size + obs_size
        self._output_size = category_size
        self._layers = layers
        self._node_size = node_size
        self._activation = nn.ELU()
        self._model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self._activation]
        for i in range(self._layers):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self._activation]
        model += [nn.Linear(self._node_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, prev_bf, prev_action, obs):
        input = torch.cat((prev_bf, prev_action, obs))
        dist_input = self._model(input)
        return td.Categorical(logits=dist_input)


class TransitionModel(nn.Module):       # q(s_t|s_{t-1}, a_{t-1})
    def __init__(self, state_size, action_size, category_size, layers, node_size):
        super().__init__()
        self._input_size = state_size + action_size
        self._output_size = category_size
        self._layers = layers
        self._node_size = node_size
        self._activation = nn.ELU()
        self._model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self._activation]
        for i in range(self._layers):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self._activation]
        model += [nn.Linear(self._node_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, prev_state, prev_action):
        input = torch.cat((prev_state, prev_action))
        dist_input = self._model(input)
        return td.Categorical(logits=dist_input)


class ObservationModel(nn.Module):       # p(o_t|s_t)
    def __init__(self, state_size, obs_size, layers, node_size):
        super().__init__()
        self._input_size = state_size
        self._output_size = obs_size
        self._layers = layers
        self._node_size = node_size
        self._activation = nn.ELU()
        self._model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self._activation]
        for i in range(self._layers):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self._activation]
        model += [nn.Linear(self._node_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, state):
        dist_input = self._model(state)
        return td.Bernoulli(logits=dist_input)


class RewardModel(nn.Module):       # w(r_t|s_t)
    def __init__(self, state_size, layers, node_size):
        super().__init__()
        self._input_size = state_size
        self._output_size = 1
        self._layers = layers
        self._node_size = node_size
        self._activation = nn.ELU()
        self._model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self._activation]
        for i in range(self._layers):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self._activation]
        model += [nn.Linear(self._node_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, state):
        dist_input = self._model(state)
        return td.Normal(dist_input, 1)


