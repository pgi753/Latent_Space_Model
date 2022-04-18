import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim

from typing import Iterable

def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


class POMDPModel:
    def __init__(self, state_cls_size, state_cat_size, action_shape, observ_shape, rnn_input_size, rnn_hidden_size,
                 device, wm_lr, actor_lr, value_lr , _lambda, actor_entropy_scale):
        self._state_cls_size = state_cls_size
        self._state_cat_size = state_cat_size
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
        self._rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=1).to(device)
        self._rnn_hidden_to_belief_vector = BeliefVector(rnn_hidden_size=rnn_hidden_size, state_cls_size=state_cls_size,
                                                         state_cat_size=state_cat_size).to(device)
        self._action_observ_to_rnn_input = RNNInput(action_shape=action_shape, observ_shape=observ_shape,
                                                    rnn_input_size=rnn_input_size).to(device)
        self._transition_matrix = TransitionMatrix(action_shape=action_shape, state_cls_size=state_cls_size,
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
                               self._transition_matrix, self._observ_decoder, self._reward_model]
        self._world_model_optimizer = optim.Adam(self.get_parameters(self._world_model_modules), lr=self._wm_lr)
        self._actor_optimizer = optim.Adam(self.get_parameters([self._actor]), lr=self._actor_lr)
        self._value_model_optimizer = optim.Adam(self.get_parameters([self._value_model]), lr=self._value_lr)

    def reset(self, initial_action):
        self._prev_rnn_hidden = torch.zeros((1, 1, self._rnn_hidden_size), dtype=torch.float32).to(self._device).detach()
        self._prev_action = torch.tensor(initial_action, dtype=torch.float32).to(self._device).detach()

    @property
    def prev_action(self):
        return self._prev_action.cpu().numpy().astype(dtype=np.int32)

    @property
    def prev_rnn_hidden(self):
        return self._prev_rnn_hidden.cpu().numpy().astype(dtype=np.float)

    @staticmethod
    def get_parameters(modules):
        model_parameters = []
        for module in modules:
            model_parameters += list(module.parameters())
        return model_parameters

    @staticmethod
    def seq_to_batch(sequence_data):
        """
        converts a sequence of length L and batch_size B to a single batch of size L*B
        """
        shp = tuple(sequence_data.shape)
        batch_data = torch.reshape(sequence_data, [shp[0] * shp[1], *shp[2:]])
        return batch_data

    @staticmethod
    def compute_return(
            reward: torch.Tensor,
            value: torch.Tensor,
            discount: torch.Tensor,
            bootstrap: torch.Tensor,
            lambda_: float
    ):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

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

    def update_target_value(self):
        mix = 1
        for param, target_param in zip(self._value_model.parameters(), self._target_value_model.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def world_model_loss(self, observ, action, reward):
        batch_size = observ.shape[1]
        rnn_input = self._action_observ_to_rnn_input(action, observ)
        init_rnn_hidden = torch.zeros((1, batch_size, self._rnn_hidden_size), dtype=torch.float32).to(self._device)
        rnn_output, rnn_hidden = self._rnn(rnn_input, init_rnn_hidden)
        rnn_combined = torch.cat((init_rnn_hidden, rnn_output), dim=0)
        bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_combined)
        tr_logit, tr_prob, tr_dist = self._transition_matrix(action)
        bv_prev = torch.unsqueeze(bv_prob[:-1], -2)
        prior = torch.squeeze(torch.matmul(bv_prev, tr_prob), dim=-2)
        posterior = bv_prob[1:]
        kld_loss = self.kld_loss(prior, posterior)
        obs_loss = self.observ_loss(observ, posterior)
        reward_loss = self.reward_loss(reward, posterior)
        model_loss = 5 * kld_loss + obs_loss + reward_loss
        return rnn_hidden.detach(), kld_loss.detach(), obs_loss.detach(), model_loss, reward_loss.detach()

    def kld_loss(self, prior, posterior):
        kld = torch.mean(torch.sum(torch.sum(posterior * torch.log((posterior+torch.finfo(torch.float32).eps) / prior),
                                             dim=-1), dim=-1))
        return kld

    def observ_loss(self, observ, posterior):
        sequence_size, batch_size = observ.shape[0], observ.shape[1]
        states, states_one_hot = self.get_all_states()
        num_states = states.shape[0]
        ob_logit, ob_prob, ob_dist = self._observ_decoder(states_one_hot)
        ob_dist = ob_dist.expand((sequence_size, batch_size, num_states))
        ob = observ.unsqueeze(-len(self._observ_shape)-1).expand((-1, -1, num_states, *self._observ_shape))
        ob = torch.argmax(ob, dim=-1)
        lp = ob_dist.log_prob(ob)
        post = posterior.unsqueeze(dim=-3).expand((-1, -1, num_states, -1, -1))
        states = states.unsqueeze(dim=-1).expand((sequence_size, batch_size, num_states, -1, -1))
        pr = torch.prod(torch.gather(post, dim=-1, index=states).squeeze(dim=-1), dim=-1)
        obs_loss = -torch.mean(torch.sum(pr*lp, dim=-1))
        return obs_loss

    def reward_loss(self, reward, posterior):
        sequence_size, batch_size = reward.shape[0], reward.shape[1]
        states, states_one_hot = self.get_all_states()
        num_states = states.shape[0]
        rew_mean, rew_dist = self._reward_model(states_one_hot)
        rew_dist = rew_dist.expand((sequence_size, batch_size, num_states))
        rew = reward.unsqueeze(-1).expand((-1, -1, num_states))
        lp = rew_dist.log_prob(rew)
        post = posterior.unsqueeze(dim=-3).expand((-1, -1, num_states, -1, -1))
        states = states.unsqueeze(dim=-1).expand((sequence_size, batch_size, num_states, -1, -1))
        pr = torch.prod(torch.gather(post, dim=-1, index=states).squeeze(dim=-1), dim=-1)
        reward_loss = -torch.mean(torch.sum(pr * lp, dim=-1))
        return reward_loss

    def imagine_test(self, rnn_hidden, horizon):
        with FreezeParameters(self._world_model_modules):
            obs = []
            rew = []
            act = []
            bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])  # b0
            state = bv_dist.sample()  # sample s0 based on b0

            for h in range(horizon):
                action_logit, action_prob, action_dist = self._actor(bv_prob)  # sample a_t by using actor model
                action = F.one_hot(action_dist.sample(), num_classes=self._action_shape[-1]).type(dtype=torch.float32).detach()
                ac = torch.argmax(action, dim=-1)

                tr_logit, tr_prob, tr_dist = self._transition_matrix(action)  # imagine (= transition)
                tr_sample = tr_dist.sample()  # sample s_{t+1} based on q(s_{t+1}|s_t,a_t)
                state = torch.gather(tr_sample, dim=-1, index=state.unsqueeze(dim=-1)).squeeze(dim=-1)
                state_one_hot = F.one_hot(state, num_classes=self._state_cat_size).type(torch.float32)

                ob_logit, ob_prob, ob_dist = self._observ_decoder(state_one_hot)  # sample o_{t+1} based on obs model
                observ = F.one_hot(ob_dist.sample(), num_classes=self._observ_shape[-1]).type(torch.float32)

                rnn_input = self._action_observ_to_rnn_input(action, observ)  # b_{t+1} = f(b_t, a_t, o_{t+1})
                rnn_input = torch.unsqueeze(rnn_input, 0)
                rnn_output, rnn_hidden = self._rnn(rnn_input, rnn_hidden)
                bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])

                obs.append(torch.argmax(observ, dim=-1))
                act.append(ac)
                reward_mean, reward_dist = self._reward_model(state_one_hot)
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

    def actor_critic_loss(self, rnn_hidden, horizon):
        # with torch.no_grad():
        #     rnn_hidden = self.seq_to_batch(rnn_hidden)

        with FreezeParameters(self._world_model_modules):
            imag_state, belief_vector, imag_log_prob, policy_entropy = self.rollout_imagination(horizon, rnn_hidden)    # imagine

        with FreezeParameters(self._world_model_modules + [self._value_model] + [self._target_value_model]):
            imag_reward_mean, imag_reward_dist = self._reward_model(imag_state)     # predict reward
            imag_reward = imag_reward_dist.mean                                     # mean -> sample
            imag_value_dist = self._target_value_model(imag_state, belief_vector)   # predict value
            imag_value = imag_value_dist.mean
            discount_arr = 0.99 * torch.ones_like(imag_reward)

        actor_loss, discount, lambda_returns = self.actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self.value_loss(imag_state, belief_vector, discount, lambda_returns)

        return actor_loss, value_loss

    def actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        lambda_returns = self.compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1],
                                             bootstrap=imag_value[-1], lambda_= self._lambda)
        advantage = (lambda_returns - imag_value[:-1]).detach()
        objective = imag_log_prob[:-1] * advantage

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)

        policy_entropy = policy_entropy[:-1]
        actor_loss = -torch.sum(torch.mean(discount * (objective + self._actor_entropy_scale * policy_entropy), dim=1))
        return actor_loss, discount, lambda_returns

    def value_loss(self, imag_state, belief_vector, discount, lambda_returns):
        with torch.no_grad():
            value_state = imag_state[:-1].detach()
            value_bv = belief_vector[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self._value_model(value_state, value_bv)
        value_loss = -torch.mean(value_discount * value_dist.log_prob(value_target))
        return value_loss

    def get_all_states(self):
        states = torch.tensor(list(itertools.product(range(self._state_cat_size), repeat=self._state_cls_size))).to(self._device)
        states_one_hot = F.one_hot(states, num_classes=self._state_cat_size).type(torch.float32).to(self._device)
        return states, states_one_hot

    @property
    def world_model_optimizer(self):
        return self._world_model_optimizer

    @property
    def actor_optimizer(self):
        return self._actor_optimizer

    @property
    def value_model_optimizer(self):
        return self._value_model_optimizer


    def rollout_imagination(self, horizon, rnn_hidden):
        # rnn_hidden = torch.unsqueeze(prev_rnn_hidden, dim=0)           # 1, 40, 64
        bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])     # b0
        state = bv_dist.sample()                                            # sample s0 based on b0
        state_list = []
        belief_vector_list = []
        img_action_entropy = []
        img_action_log_probs = []

        for h in range(horizon):
            action_logit, action_prob, action_dist = self._actor(bv_prob)   # sample a_t by using actor model
            action = F.one_hot(action_dist.sample(), num_classes=self._action_shape[-1]).type(dtype=torch.float32).detach()
            act = torch.argmax(action, dim=-1)

            tr_logit, tr_prob, tr_dist = self._transition_matrix(action)    # imagine (= transition)
            tr_sample = tr_dist.sample()                                    # sample s_{t+1} based on q(s_{t+1}|s_t,a_t)
            state = torch.gather(tr_sample, dim=-1, index=state.unsqueeze(dim=-1)).squeeze(dim=-1)
            state_one_hot = F.one_hot(state, num_classes=self._state_cat_size).type(torch.float32)

            ob_logit, ob_prob, ob_dist = self._observ_decoder(state_one_hot)    # sample o_{t+1} based on obs model
            observ = F.one_hot(ob_dist.sample(), num_classes=self._observ_shape[-1]).type(torch.float32)

            rnn_input = self._action_observ_to_rnn_input(action, observ)        # b_{t+1} = f(b_t, a_t, o_{t+1})
            rnn_input = torch.unsqueeze(rnn_input, 0)
            rnn_output, rnn_hidden = self._rnn(rnn_input, rnn_hidden)
            bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])

            state_list.append(state_one_hot)
            belief_vector_list.append(bv_prob)
            img_action_entropy.append(action_dist.entropy())
            img_action_log_probs.append(action_dist.log_prob(act))

        return torch.stack(state_list, dim=0), torch.stack(belief_vector_list, dim=0),\
               torch.stack(img_action_entropy, dim=0), torch.stack(img_action_log_probs, dim=0)

    def imagination(self, init_rnn_hidden, init_action, horizon):
        init_rnn_hidden = torch.tensor(init_rnn_hidden, dtype=torch.float32).to(self._device)
        init_action = torch.tensor(init_action, dtype=torch.float32).to(self._device)
        rnn_hidden = init_rnn_hidden.unsqueeze(dim=0)
        action = init_action
        bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])
        state = bv_dist.sample()
        state_list = []
        observ_list = []
        belief_vector_list = []
        action_list = []
        reward_list = []
        for h in range(horizon):
            tr_logit, tr_prob, tr_dist = self._transition_matrix(action)
            tr_sample = tr_dist.sample()
            state = torch.gather(tr_sample, dim=-1, index=state.unsqueeze(dim=-1)).squeeze(dim=-1)
            state_one_hot = F.one_hot(state, num_classes=self._state_cat_size).type(torch.float32)
            ob_logit, ob_prob, ob_dist = self._observ_decoder(state_one_hot)
            observ = F.one_hot(ob_dist.sample(), num_classes=self._observ_shape[-1]).type(torch.float32)
            reward_mean, reward_dist = self._reward_model(state_one_hot)
            # reward = reward_dist.sample()
            reward = reward_dist.mean
            rnn_input = self._action_observ_to_rnn_input(action, observ)
            rnn_input = torch.unsqueeze(rnn_input, 0)
            rnn_output, rnn_hidden = self._rnn(rnn_input, rnn_hidden)
            bv_logit, bv_prob, bv_dist = self._rnn_hidden_to_belief_vector(rnn_hidden[0, :, :])
            action_logit, action_prob, action_dist = self._actor(bv_prob)
            action = F.one_hot(action_dist.sample(), num_classes=self._action_shape[-1]).type(dtype=torch.float32)
            state_list.append(state)
            observ_list.append(observ)
            belief_vector_list.append(bv_prob)
            action_list.append(action)
            reward_list.append(reward)
        return torch.stack(state_list, dim=0), torch.stack(observ_list, dim=0), \
               torch.stack(belief_vector_list, dim=0), torch.stack(action_list, dim=0), torch.stack(reward_list, dim=0)


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
        self._model = MLP(input_shape=(state_cls_size, state_cat_size), output_shape=(1, ),
                          num_hidden_layers=3, hidden_size=(state_cls_size * state_cat_size))

    def forward(self, state):
        mean = torch.squeeze(self._model(state), dim=-1)
        dist = td.Normal(mean, 0.5)
        dist = td.independent.Independent(dist, reinterpreted_batch_ndims= 0)
        return mean, dist


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
        self._model = MLP(input_shape=(state_cls_size, state_cat_size*2), output_shape=(1,),
                          num_hidden_layers=3, hidden_size=(state_cls_size * state_cat_size))

    def forward(self, state, belief_vector):
        input = torch.cat((state, belief_vector), dim=-1)
        mean = torch.squeeze(self._model(input), dim= -1)
        dist = td.Normal(mean, 1)
        dist = td.independent.Independent(dist, reinterpreted_batch_ndims=0)
        return dist


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
