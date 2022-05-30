import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim
from utility import get_parameters, FreezeParameters

class POMDPModel:
    def __init__(self, state_cls_size, state_cat_size, state_sample_size, action_shape, observ_shape, rnn_input_size,
                 rnn_hidden_size, device, wm_lr, actor_lr, value_lr, _lambda, actor_entropy_scale, discount, kld_scale):
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self._state_sample_size = state_sample_size
        self._action_shape = action_shape
        self._observ_shape = observ_shape
        self._rnn_input_size = rnn_input_size
        self._rnn_hidden_size = rnn_hidden_size
        self._device = device
        self._wm_lr = wm_lr
        self._actor_lr = actor_lr
        self._value_lr = value_lr
        self._lambda = _lambda
        self._actor_entropy_scale = actor_entropy_scale
        self._discount = discount
        self._kld_scale = kld_scale
        self._rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=1).to(device)
        self._rnn_hidden_to_belief_vector = BeliefVector(rnn_hidden_size=rnn_hidden_size, state_cls_size=state_cls_size,
                                                         state_cat_size=state_cat_size).to(device)
        self._action_observ_to_rnn_input = RNNInput(action_shape=action_shape, observ_shape=observ_shape,
                                                    rnn_input_size=rnn_input_size).to(device)
        self._transition_matrix = TransitionMatrix(action_shape=action_shape, state_cls_size=state_cls_size,
                                                   state_cat_size=state_cat_size).to(device)
        self._transition_matrix2 = TransitionMatrix2(action_shape=action_shape, state_cls_size=state_cls_size,
                                                     state_cat_size=state_cat_size).to(device)
        self._observ_decoder = ObservDecoder(state_cls_size=state_cls_size, state_cat_size=state_cat_size,
                                             observ_shape=observ_shape).to(device)
        self._reward_model = RewardModel(state_cls_size=state_cls_size, state_cat_size=state_cat_size).to(device)
        self._actor = Actor(state_cls_size=state_cls_size, state_cat_size=state_cat_size, action_shape=action_shape).to(device)
        self._value_model = ValueModel(state_cls_size=state_cls_size, state_cat_size=state_cat_size).to(device)
        self._target_value_model = ValueModel(state_cls_size=state_cls_size, state_cat_size=state_cat_size).to(device)
        self._target_value_model.load_state_dict(self._value_model.state_dict())
        self._prev_rnn_hidden = torch.zeros((1, 1, rnn_hidden_size), dtype=torch.float32).to(device).detach()
        self._prev_action = torch.zeros(action_shape, dtype=torch.float32).to(device).detach()
        self._world_model_modules = [self._rnn, self._rnn_hidden_to_belief_vector, self._action_observ_to_rnn_input,
                                     self._transition_matrix2, self._observ_decoder, self._reward_model]
        self._world_model_optimizer = optim.Adam(get_parameters(self._world_model_modules), lr=self._wm_lr)
        self._actor_optimizer = optim.Adam(get_parameters([self._actor]), lr=self._actor_lr)
        self._value_model_optimizer = optim.Adam(get_parameters([self._value_model]), lr=self._value_lr)

    @property
    def prev_action(self):
        return self._prev_action.cpu().numpy().astype(dtype=np.int32)

    @property
    def prev_rnn_hidden(self):
        return self._prev_rnn_hidden.cpu().numpy().astype(dtype=np.float)

    @property
    def world_model_parameters(self):
        return get_parameters(self._world_model_modules)

    @property
    def actor_parameters(self):
        return self._actor.parameters()

    @property
    def value_model_parameters(self):
        return self._value_model.parameters()

    @property
    def world_model_optimizer(self):
        return self._world_model_optimizer

    @property
    def actor_optimizer(self):
        return self._actor_optimizer

    @property
    def value_model_optimizer(self):
        return self._value_model_optimizer

    def reset(self, initial_action):
        self._prev_rnn_hidden = torch.zeros((1, 1, self._rnn_hidden_size), dtype=torch.float32).to(self._device).detach()
        self._prev_action = torch.tensor(initial_action, dtype=torch.float32).to(self._device).detach()

    def step(self, observ):
        with torch.no_grad():
            observ = torch.tensor(observ, dtype=torch.float32).to(self._device)
            rnn_input = self._action_observ_to_rnn_input(self._prev_action, observ)
            rnn_input = torch.unsqueeze(torch.unsqueeze(rnn_input, 0), 0)
            rnn_output, rnn_hidden = self._rnn(rnn_input, self._prev_rnn_hidden)
            bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, 0, :])
            action_logit, action_prob, action_dist = self._actor(bv_prob)
            action = F.one_hot(action_dist.sample(), num_classes=self._action_shape[-1])
            action = action.type(dtype=torch.float32)
            self._prev_action = action
            self._prev_rnn_hidden = rnn_hidden
        return action.cpu().numpy().astype(dtype=np.int32)

    def update_target_value(self, mix_rate):
        for param, target_param in zip(self._value_model.parameters(), self._target_value_model.parameters()):
            target_param.data.copy_(mix_rate * param.data + (1 - mix_rate) * target_param.data)

    def get_all_states(self):
        states = torch.tensor(list(itertools.product(range(self._state_cat_size), repeat=self._state_cls_size))).to(self._device)
        states_one_hot = F.one_hot(states, num_classes=self._state_cat_size).type(torch.float32).to(self._device)
        return states, states_one_hot

    def get_some_states(self):
        states = np.random.randint(self._state_cat_size, size=(self._state_sample_size, self._state_cls_size))
        states = torch.tensor(states, dtype=torch.int64).to(self._device)
        states_one_hot = F.one_hot(states, num_classes=self._state_cat_size).type(torch.float32).to(self._device)
        return states, states_one_hot

    def world_model_loss(self, observ, action, reward):
        batch_size = observ.shape[1]
        rnn_input = self._action_observ_to_rnn_input(action, observ)
        init_rnn_hidden = torch.zeros((1, batch_size, self._rnn_hidden_size), dtype=torch.float32).to(self._device)
        rnn_output, rnn_hidden = self._rnn(rnn_input, init_rnn_hidden)
        rnn_combined = torch.cat((init_rnn_hidden, rnn_output), dim=0)
        bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_combined)

        # tr_logit, tr_prob, tr_dist = self._transition_matrix(action)
        tr_prob, tr_dist = self._transition_matrix2(action)

        bv_prev = torch.unsqueeze(bv_prob[:-1], -2)
        prior = torch.squeeze(torch.matmul(bv_prev, tr_prob), dim=-2)
        posterior = bv_prob[1:]
        kld_loss = self.kld_loss(prior, posterior)
        obs_loss = self.observ_loss(observ, posterior)
        reward_loss = self.reward_loss(reward, posterior)
        model_loss = (kld_loss * self._kld_scale) + obs_loss + reward_loss
        return rnn_hidden.detach(), kld_loss.detach(), obs_loss.detach(), model_loss, reward_loss.detach()

    def kld_loss(self, prior, posterior):
        kld = torch.mean(torch.sum(torch.sum(posterior * torch.log((posterior+torch.finfo(torch.float32).eps) / prior),
                                             dim=-1), dim=-1))
        return kld

    def observ_loss(self, observ, posterior):
        sequence_size, batch_size = observ.shape[0], observ.shape[1]
        # states, states_one_hot = self.get_all_states()
        states, states_one_hot = self.get_some_states()
        num_states = states.shape[0]
        ob_logit, ob_prob, ob_dist = self._observ_decoder(states_one_hot)
        ob_dist = ob_dist.expand((sequence_size, batch_size, num_states))
        ob = observ.unsqueeze(-len(self._observ_shape)-1).expand((-1, -1, num_states, *self._observ_shape))
        ob = torch.argmax(ob, dim=-1)
        lp = ob_dist.log_prob(ob)
        post = posterior.unsqueeze(dim=-3).expand((-1, -1, num_states, -1, -1))
        states = states.unsqueeze(dim=-1).expand((sequence_size, batch_size, num_states, -1, -1))
        pr = torch.prod(torch.gather(post, dim=-1, index=states).squeeze(dim=-1), dim=-1)
        obs_loss = -torch.mean(torch.sum(pr * lp, dim=-1))
        return obs_loss

    def reward_loss(self, reward, posterior):
        sequence_size, batch_size = reward.shape[0], reward.shape[1]
        # states, states_one_hot = self.get_all_states()
        states, states_one_hot = self.get_some_states()
        num_states = states.shape[0]
        rew_dist = self._reward_model(states_one_hot)
        rew_dist = rew_dist.expand((sequence_size, batch_size, num_states))
        rew = reward.unsqueeze(-1).expand((-1, -1, num_states))
        lp = rew_dist.log_prob(rew)
        post = posterior.unsqueeze(dim=-3).expand((-1, -1, num_states, -1, -1))
        states = states.unsqueeze(dim=-1).expand((sequence_size, batch_size, num_states, -1, -1))
        pr = torch.prod(torch.gather(post, dim=-1, index=states).squeeze(dim=-1), dim=-1)
        reward_loss = -torch.mean(torch.sum(pr * lp, dim=-1))
        return reward_loss

    def actor_critic_loss(self, rnn_hidden, horizon):
        with FreezeParameters(self._world_model_modules + [self._target_value_model]):
            # imagination
            imag_state, belief_vector, policy_entropy, imag_log_prob = self.rollout_imagination(horizon, rnn_hidden)
            imag_reward_dist = self._reward_model(imag_state[:-1])                      # predict reward    t = 0,...,H-1
            imag_reward = imag_reward_dist.mean                                         # mean or sample
            imag_value = self._target_value_model(imag_state, belief_vector)            # predict value
            discount_arr = self._discount * torch.ones_like(imag_reward)

        # value estimation
        lambda_returns = self.value_estimation(imag_reward, imag_value, discount_arr)

        # actor, critic loss
        actor_loss = self.actor_loss(lambda_returns, imag_value, imag_log_prob, policy_entropy)
        value_loss = self.value_loss(imag_state, belief_vector, lambda_returns)
        return actor_loss, value_loss

    def value_estimation(self, reward, value, discount):
        target_value = reward[-1] + discount[-1] * value[-1]    # H-1
        timesteps = list(range(reward.shape[0] - 2, -1, -1))    # H-2 ~ 0
        outputs = [target_value]
        for t in timesteps:
            target_value = reward[t] + discount[t] * ((1 - self._lambda) * value[t + 1] + self._lambda * target_value)
            outputs.append(target_value)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

    def actor_loss(self, lambda_returns, imag_value, imag_log_prob, policy_entropy):
        advantage = (lambda_returns - imag_value[:-1]).detach()
        objective = imag_log_prob * advantage
        # actor_loss = -torch.mean(torch.sum(objective + self._actor_entropy_scale * policy_entropy, dim=1))
        actor_loss = -torch.mean(objective + self._actor_entropy_scale * policy_entropy)
        return actor_loss

    def value_loss(self, imag_state, belief_vector, lambda_returns):
        value_state = imag_state[:-1].detach()
        value_bv = belief_vector[:-1].detach()
        value_target = lambda_returns.detach()

        value = self._value_model(value_state, value_bv)
        # value_loss = torch.mean(torch.sum(((value - value_target) ** 2) / 2, dim=1))
        value_loss = torch.mean(((value - value_target) ** 2) / 2)
        return value_loss

    def rollout_imagination(self, horizon, rnn_hidden):
        bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])     # b0
        state = bv_dist.sample()                                                                # sample s0 based on b0
        state_list = [F.one_hot(state, num_classes=self._state_cat_size).type(torch.float32)]   # append s0
        belief_vector_list = [bv_prob]                                                          # append b0
        img_action_entropy = []
        img_action_log_probs = []

        for h in range(horizon-1):
            action_logit, action_prob, action_dist = self._actor(bv_prob.detach())              # sample a_t by using actor model
            act = action_dist.sample()
            action = F.one_hot(act, num_classes=self._action_shape[-1]).type(dtype=torch.float32)

            # tr_logit, tr_prob, tr_dist = self._transition_matrix(action.detach())               # imagine (= transition)
            tr_prob, tr_dist = self._transition_matrix2(action.detach())
            tr_sample = tr_dist.sample()                                                        # sample s_{t+1} based on q(s_{t+1}|s_t,a_t)
            state = torch.gather(tr_sample, dim=-1, index=state.unsqueeze(dim=-1)).squeeze(dim=-1)
            state_one_hot = F.one_hot(state, num_classes=self._state_cat_size).type(torch.float32)

            ob_logit, ob_prob, ob_dist = self._observ_decoder(state_one_hot)                    # sample o_{t+1} based on obs model
            observ = F.one_hot(ob_dist.sample(), num_classes=self._observ_shape[-1]).type(torch.float32)

            rnn_input = self._action_observ_to_rnn_input(action.detach(), observ)               # b_{t+1} = f(b_t, a_t, o_{t+1})
            rnn_input = torch.unsqueeze(rnn_input, 0)
            rnn_output, rnn_hidden = self._rnn(rnn_input, rnn_hidden)
            bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])

            state_list.append(state_one_hot)                                                    # t = 0,...,H
            belief_vector_list.append(bv_prob)                                                  # t = 0,...,H
            img_action_entropy.append(action_dist.entropy())                                    # t = 0,...,H-1
            img_action_log_probs.append(action_dist.log_prob(act))                              # t = 0,...,H-1
        return torch.stack(state_list, dim=0), torch.stack(belief_vector_list, dim=0), \
               torch.stack(img_action_entropy, dim=0), torch.stack(img_action_log_probs, dim=0)

    def imagine_test(self, rnn_hidden, horizon):
        with torch.no_grad():
            obs = []
            rew = []
            act = []
            bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])     # b0
            state = bv_dist.sample()                                                                # sample s0 based on b0

            for h in range(horizon):
                action_logit, action_prob, action_dist = self._actor(bv_prob)       # sample a_t by using actor model
                ac = action_dist.sample()
                action = F.one_hot(ac, num_classes=self._action_shape[-1]).type(dtype=torch.float32).detach()

                # tr_logit, tr_prob, tr_dist = self._transition_matrix(action)        # imagine (= transition)
                # tr_sample = tr_dist.sample()
                tr_prob, tr_dist = self._transition_matrix2(action)
                tr_sample = tr_dist.sample()                                        # sample s_{t+1} based on q(s_{t+1}|s_t,a_t)
                state = torch.gather(tr_sample, dim=-1, index=state.unsqueeze(dim=-1)).squeeze(dim=-1)
                state_one_hot = F.one_hot(state, num_classes=self._state_cat_size).type(torch.float32)

                ob_logit, ob_prob, ob_dist = self._observ_decoder(state_one_hot)    # sample o_{t+1} based on obs model
                observ = F.one_hot(ob_dist.sample(), num_classes=self._observ_shape[-1]).type(torch.float32)

                rnn_input = self._action_observ_to_rnn_input(action, observ)        # b_{t+1} = f(b_t, a_t, o_{t+1})
                rnn_input = torch.unsqueeze(rnn_input, 0)
                rnn_output, rnn_hidden = self._rnn(rnn_input, rnn_hidden)
                bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])

                obs.append(torch.argmax(observ, dim=-1))
                act.append(ac)
                reward_dist = self._reward_model(state_one_hot)
                reward = reward_dist.mean
                rew.append(reward)

            obs, act, rew = torch.stack(obs, dim=0), torch.stack(act, dim=0), torch.stack(rew, dim=0)
            o = obs.detach().cpu().squeeze(dim=1).numpy()
            o = np.transpose(o)

            a = act.detach().cpu().squeeze(dim=1).numpy()
            a = np.transpose(a)

            r = rew.detach().cpu().squeeze(dim=1).numpy()
            r = np.round(r).astype(np.int64)
            r = np.transpose(r)
        return o, a, r

    def save_model(self, path):
        save_dict = {
            "RNN": self._rnn.state_dict(),
            "BeliefVector": self._rnn_hidden_to_belief_vector.state_dict(),
            "RNNInput": self._action_observ_to_rnn_input.state_dict(),
            # "TransitionMatrix": self._transition_matrix.state_dict(),
            "TransitionMatrix2": self._transition_matrix2.state_dict(),
            "ObservDecoder": self._observ_decoder.state_dict(),
            "RewardModel": self._reward_model.state_dict(),
            "Actor": self._actor.state_dict(),
            "ValueModel": self._value_model.state_dict()
        }
        torch.save(save_dict, path)

    def load_model(self, path):
        saved_dict = torch.load(path, map_location=self._device)
        self._rnn.load_state_dict(saved_dict["RNN"])
        self._rnn_hidden_to_belief_vector.load_state_dict(saved_dict["BeliefVector"])
        self._action_observ_to_rnn_input.load_state_dict(saved_dict["RNNInput"])
        # self._transition_matrix.load_state_dict(saved_dict["TransitionMatrix"])
        self._transition_matrix2.load_state_dict(saved_dict["TransitionMatrix2"])
        self._observ_decoder.load_state_dict(saved_dict["ObservDecoder"])
        self._reward_model.load_state_dict(saved_dict["RewardModel"])
        self._actor.load_state_dict(saved_dict["Actor"])
        self._value_model.load_state_dict(saved_dict["ValueModel"])


class BeliefVector(nn.Module):
    def __init__(self, rnn_hidden_size, state_cls_size, state_cat_size):
        super().__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self._model = MLP(input_shape=(rnn_hidden_size,), output_shape=(state_cls_size, state_cat_size),
                          num_hidden_layers=1, hidden_size=rnn_hidden_size)

    def forward(self, rnn_hidden):
        logit = self._model(rnn_hidden)
        prob = F.softmax(logit, dim=-1)
        dist = td.independent.Independent(td.categorical.Categorical(logits=logit), reinterpreted_batch_ndims=1)
        return logit, prob, dist


class RNNInput(nn.Module):
    def __init__(self, action_shape, observ_shape, rnn_input_size):
        super().__init__()
        self._action_shape = action_shape
        self._observ_shape = observ_shape
        input_size = int(np.prod(action_shape) + np.prod(observ_shape))
        self._model = MLP(input_shape=(input_size,), output_shape=(rnn_input_size,),
                          num_hidden_layers=-1, hidden_size=0)

    def forward(self, action, observ):
        shp = action.shape[:-len(self._action_shape)]
        action = torch.reshape(action, (*shp, -1))
        observ = torch.reshape(observ, (*shp, -1))
        x = torch.cat((action, observ), dim=-1)
        y = self._model(x)
        return y


class TransitionMatrix(nn.Module):
    def __init__(self, action_shape, state_cls_size, state_cat_size):
        super().__init__()
        self._action_shape = action_shape
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self._model = MLP(input_shape=action_shape, output_shape=(state_cls_size, state_cat_size, state_cat_size),
                          num_hidden_layers=-1, hidden_size=state_cls_size * state_cat_size * state_cat_size)

    def forward(self, action):
        logit = self._model(action)
        prob = F.softmax(logit, dim=-1)
        dist = td.categorical.Categorical(logits=logit)
        return logit, prob, dist


class TransitionMatrix2(nn.Module):
    def __init__(self, action_shape, state_cls_size, state_cat_size):
        super().__init__()
        action_size = int(action_shape[0])
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self.transition_matrix = nn.Parameter(torch.zeros((state_cls_size, action_size, state_cat_size, state_cat_size),
                                                          requires_grad=True, dtype=torch.float32))

    def forward(self, action):
        """
        :param action: (seq_len, batch_size, action_size) (seq_len, 60, 2)
        tr_matrix: (class_size, action_size, cat_size, cat_size) (1, 2, 64, 64)
        tr_prob: (seq_len, batch_size, class_size, cat_size, cat_size) (seq_len, 60, 1, 64, 64)
        :return: tr_prob
        """
        dim = action.dim()
        if dim == 2:
            action = action.unsqueeze(dim=0)
        sequence_size, batch_size = action.shape[0], action.shape[1]
        action = torch.argmax(action, dim=-1, keepdim=True).unsqueeze(dim=-2).unsqueeze(dim=-1).unsqueeze(dim=-1)
        action = action.expand((-1, -1, self._state_cls_size, -1, self._state_cat_size, self._state_cat_size))
        tr_matrix = F.softmax(self.transition_matrix, dim=-1).unsqueeze(dim=0).unsqueeze(dim=0)
        tr_matrix = tr_matrix.expand((sequence_size, batch_size, -1, -1, -1, -1))
        prob = torch.gather(input=tr_matrix, dim=3, index=action).squeeze(dim=3)
        if dim == 2:
            prob = prob.squeeze(dim=0)
        dist = td.categorical.Categorical(probs=prob)
        return prob, dist


class ObservDecoder(nn.Module):
    def __init__(self, state_cls_size, state_cat_size, observ_shape):
        super().__init__()
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self._observ_shape = observ_shape
        self._model = MLP(input_shape=(state_cls_size, state_cat_size), output_shape=observ_shape,
                          num_hidden_layers=1, hidden_size=(state_cls_size * state_cat_size))

    def forward(self, state):
        logit = self._model(state)
        prob = F.softmax(logit, dim=-1)
        reinterpreted_batch_ndims = len(self._observ_shape) - 1
        dist = td.independent.Independent(td.categorical.Categorical(logits=logit),
                                          reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        return logit, prob, dist


class RewardModel(nn.Module):
    def __init__(self, state_cls_size, state_cat_size):
        super().__init__()
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self._model = MLP(input_shape=(state_cls_size, state_cat_size), output_shape=(1,),
                          num_hidden_layers=3, hidden_size=(state_cls_size * state_cat_size))

    def forward(self, state):
        mean = torch.squeeze(self._model(state), dim=-1)
        dist = td.Normal(mean, 0.4)
        dist = td.independent.Independent(dist, reinterpreted_batch_ndims=0)
        return dist


class Actor(nn.Module):
    def __init__(self, state_cls_size, state_cat_size, action_shape):
        super().__init__()
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self._action_shape = action_shape
        self._model = MLP(input_shape=(state_cls_size, state_cat_size), output_shape=action_shape,
                          num_hidden_layers=1, hidden_size=(state_cls_size * state_cat_size))

    def forward(self, belief_vector):
        logit = self._model(belief_vector)
        prob = F.softmax(logit, dim=-1)
        reinterpreted_batch_ndims = len(self._action_shape) - 1
        dist = td.independent.Independent(td.categorical.Categorical(logits=logit),
                                          reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        return logit, prob, dist


class ValueModel(nn.Module):
    def __init__(self, state_cls_size, state_cat_size):
        super().__init__()
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
        self._model = MLP(input_shape=(state_cls_size, state_cat_size * 2), output_shape=(1,),
                          num_hidden_layers=1, hidden_size=(state_cls_size * state_cat_size * 2))

    def forward(self, state, belief_vector):
        x = torch.cat((state, belief_vector), dim=-1)
        y = torch.squeeze(self._model(x), dim=-1)
        return y


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, num_hidden_layers, hidden_size):
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._input_size = int(np.prod(input_shape))
        self._output_size = int(np.prod(output_shape))
        self._num_hidden_layers = num_hidden_layers
        self._hidden_size = hidden_size
        self._activation = nn.ELU()
        self._model = self.build_model()

    def build_model(self):
        if self._num_hidden_layers >= 0:
            model = [nn.Linear(self._input_size, self._hidden_size)]
            model += [self._activation]
            for i in range(self._num_hidden_layers):
                model += [nn.Linear(self._hidden_size, self._hidden_size)]
                model += [self._activation]
            model += [nn.Linear(self._hidden_size, self._output_size)]
        else:
            model = [nn.Linear(self._input_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, x):
        shp = x.shape[:-len(self._input_shape)]
        x = torch.reshape(x, (*shp, self._input_size))
        y = self._model(x)
        y = torch.reshape(y, (*shp, *self._output_shape))
        return y
