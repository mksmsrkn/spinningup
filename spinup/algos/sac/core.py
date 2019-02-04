import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from gym.spaces import Box

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes=(64,64), activation=nn.Tanh,
                 output_activation=None, output_scaler=1, do_squeeze = False):
        super(MLP, self).__init__()
        self.output_scaler = output_scaler
        self.do_squeeze = do_squeeze
        layers = []
        prev_h = in_dim
        for h in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_h, h))
            layers.append(activation())
            prev_h = h
        layers.append(nn.Linear(h, hidden_sizes[-1]))
        if output_activation:
            try:
                out = output_activation(-1) # Sigmoid specific case
            except:
                out = output_activation()
            layers.append(out)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        if self.do_squeeze: x.squeeze_()
        return x * self.output_scaler

# Credit: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_vars(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).float()
    clip_low = (x < l).float()
    return x + ((u - x)*clip_up + (l - x)*clip_low).detach()

"""
Policies
"""
def apply_squashing_func(mu, pi, logp_pi):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= (torch.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + EPS)).sum(dim=1)
    return mu, pi, logp_pi

class MLPGaussian(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(64,64),
                 activation=nn.Tanh, output_activation=None, act_limit=1.0):
        super(MLPGaussian, self).__init__()
        self.act_limit = act_limit
        self.net = MLP(in_dim, list(hidden_sizes), activation, activation, do_squeeze = False)
        self.mu = [nn.Linear(hidden_sizes[-1], out_dim)]
        if output_activation is not None: self.mu.append(output_activation())
        self.log_sigma = [nn.Linear(hidden_sizes[-1], out_dim), nn.Tanh()]
        self.mu = nn.Sequential(*self.mu)
        self.log_sigma = nn.Sequential(*self.log_sigma)

    def forward(self, x, a = None):
        x = self.net(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        """ Note from Josh Achiam @ OpenAI
        Because algorithm maximizes trade-off of reward and entropy,
        entropy must be unique to state---and therefore log_stds need
        to be a neural network output instead of a shared-across-states
        learnable parameter vector. But for deep Relu and other nets,
        simply sticking an activationless dense layer at the end would
        be quite bad---at the beginning of training, a randomly initialized
        net could produce extremely large values for the log_stds, which
        would result in some actions being either entirely deterministic
        or too random to come back to earth. Either of these introduces
        numerical instability which could break the algorithm. To
        protect against that, we'll constrain the output range of the
        log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
        slightly different from the trick used by the original authors of
        SAC---they used tf.clip_by_value instead of squashing and rescaling.
        I prefer this approach because it allows gradient propagation
        through log_std where clipping wouldn't, but I don't know if
        it makes much of a difference.
        """
        log_sigma = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sigma + 1)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        # rsample() - https://pytorch.org/docs/stable/distributions.html#pathwise-derivative
        pi = dist.rsample() # reparametrization
        logp_pi = dist.log_prob(pi).sum(dim=1)

        mu *= self.act_limit
        pi *= self.act_limit
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        return mu, pi, logp_pi

"""
Actor-Critics
"""
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(400,300), activation=nn.ReLU, #torch.relu, # nn.ReLU
                 output_activation=None, action_space=None, policy = MLPGaussian):
        super(ActorCritic, self).__init__()
        assert isinstance(action_space, Box)
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.policy = policy(state_dim, act_dim, list(hidden_sizes),
                             activation, output_activation, act_limit)

        self.q1 = MLP(state_dim + act_dim, list(hidden_sizes)+[1], activation, do_squeeze = True)
        self.q2 = MLP(state_dim + act_dim, list(hidden_sizes)+[1],  activation, do_squeeze = True)
        self.v = MLP(state_dim, list(hidden_sizes)+[1], activation, do_squeeze = True)

    def forward(self, x, a = None):
        mu, pi, logp_pi = self.policy(x)
        if a is None:
            return mu, pi, logp_pi
        else:
            q1 = self.q1(torch.cat([x, a],dim=1))
            q1_pi = self.q1(torch.cat([x, pi],dim=1))
            q2 = self.q2(torch.cat([x, a],dim=1))
            q2_pi = self.q2(torch.cat([x, pi],dim=1))
            v = self.v(x)
            return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v