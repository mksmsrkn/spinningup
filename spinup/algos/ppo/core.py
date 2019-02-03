import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes=(64,64), activation=nn.Tanh, output_activation=None):
        super(MLP, self).__init__()
        layers = []
        prev_h = in_dim
        for h in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_h, h))
            layers.append(activation())
            prev_h = h
        layers.append(nn.Linear(h, hidden_sizes[-1]))
        if output_activation:
            layers.append(output_activation(-1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

# Credit: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_vars(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

"""
Policies
"""
class MLPCategorical(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(64,64), activation=nn.Tanh,
                 output_activation=nn.Softmax):
        super(MLPCategorical, self).__init__()
        self.probs = MLP(in_dim, list(hidden_sizes)+[out_dim], activation, output_activation)

    def forward(self, x, a = None):
        probs = self.probs(x)
        dist = Categorical(probs)
        pi = dist.sample()
        logp = dist.log_prob(a) if a is not None else None
        logp_pi = dist.log_prob(pi)
        return pi, logp, logp_pi

class MLPGaussian(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(64,64), activation=nn.Tanh, output_activation=None):
        super(MLPGaussian, self).__init__()
        self.mu = MLP(in_dim, list(hidden_sizes)+[out_dim], activation, output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(out_dim, dtype=torch.float32))

    def forward(self, x, a = None):
        mu = self.mu(x)
        sigma = self.log_std.exp()
        dist = Normal(mu, sigma)
        pi = dist.sample()
        logp = dist.log_prob(a).sum(dim=1) if a is not None else None
        logp_pi = dist.log_prob(pi).sum()
        return pi, logp, logp_pi

"""
Actor-Critics
"""
class ActorCritic(nn.Module):
    def __init__(self, state, hidden_sizes=(64,64), activation=nn.Tanh,
                 output_activation=None, policy=None, action_space=None):
        super(ActorCritic, self).__init__()
        assert len(state) == 1
        # default policy builder depends on action space
        if policy is None and isinstance(action_space, Box):
            self.policy = MLPGaussian(state[0], action_space.shape[0], hidden_sizes, activation, output_activation)
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = MLPCategorical(state[0], action_space.n, hidden_sizes, activation, nn.Softmax)
        else:
            self.policy = MLP(state[0], hidden_sizes, activation, output_activation)
        self.value = MLP(state[0], list(hidden_sizes)+[1], activation, None)

    def forward(self, x, a = None):
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value(x)
        return pi, logp, logp_pi, v