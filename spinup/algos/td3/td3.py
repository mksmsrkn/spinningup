import numpy as np
import torch
from torch import optim
from torch import Tensor
import gym
import time
from spinup.algos.td3 import core
from spinup.utils.logx import EpochLogger

# TODO multiple devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

TD3 (Twin Delayed DDPG)

"""
def td3(env_fn, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A reference to ActorCritic class which after instantiation
            takes state, ``x``, and action, ``a``, and returns:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x`` and actions in
                                           | ``a``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x`` and actions in
                                           | ``a``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x``:
                                           | q1(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # https://pytorch.org/docs/master/notes/randomness.html#cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    actor_critic_main = actor_critic(obs_dim, **ac_kwargs).to(device)
    # Note that the action placeholder going to targer actor_critic here
    # is irrelevant, because we only need q_targ(s, pi_targ(s)).
    actor_critic_target = actor_critic(obs_dim, **ac_kwargs).to(device)

    # Count variables
    var_counts = tuple(core.count_vars(model) for model in [actor_critic_main.policy,
                                                            actor_critic_main.q1,
                                                            actor_critic_main.q2,
                                                            actor_critic_main])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Optimizers
    pi_optimizer = optim.Adam(actor_critic_main.policy.parameters(), lr=pi_lr)
    main_q_parameters = [*actor_critic_main.q1.parameters(),*actor_critic_main.q2.parameters()]
    q_optimizer = optim.Adam(main_q_parameters, lr=q_lr)

    # Initializing targets to match main variables
    actor_critic_target.load_state_dict(actor_critic_main.state_dict())

    def get_action(o, noise_scale):
        a = actor_critic_main(Tensor(o.reshape(1,-1)).to(device))
        a = a.cpu().detach().numpy()
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all TD3 updates at the end of the trajectory
            (in accordance with source code of TD3 published by
            original authors).
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                x, x2, a, r, d = [Tensor(batch[k]).to(device) for k in
                                  ['obs1', 'obs2', 'acts', 'rews', 'done']]

                _, q1, q2, q1_pi = actor_critic_main(x, a)
                pi_targ = actor_critic_target.policy(x2)

                # Target policy smoothing, by adding clipped noise to target actions
                epsilon = torch.randn_like(pi_targ) * target_noise
                epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
                a2 = pi_targ + epsilon
                a2 = torch.clamp(a2, -act_limit, act_limit)
                _, q1_targ, q2_targ, _ = actor_critic_target(x2, a2)

                # Bellman backup for Q functions, using Clipped Double-Q targets
                min_q_targ = torch.min(q1_targ, q2_targ)
                backup = (r + gamma*(1-d)*min_q_targ).detach()

                # TD3 losses
                q1_loss = ((q1-backup)**2).mean()
                q2_loss = ((q2-backup)**2).mean()
                q_loss = q1_loss + q2_loss

                # Q-learning update
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()
                logger.store(LossQ=q_loss,
                             Q1Vals=q1.cpu().detach().numpy(),
                             Q2Vals=q2.cpu().detach().numpy())

                if j % policy_delay == 0:
                    pi_loss = -q1_pi.mean() # NOTE: q1_pi is outdated
                    # Delayed policy update
                    pi_optimizer.zero_grad()
                    pi_loss.backward()
                    pi_optimizer.step()
                    logger.store(LossPi=pi_loss)

                    # Polyak averaging for target variables
                    # Credits: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
                    params = zip(actor_critic_target.parameters(), actor_critic_main.parameters())
                    for ac_target, ac_main in params:
                        ac_target.data.copy_(ac_main.data * (1.0 - polyak) + ac_target.data * polyak)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, actor_critic_main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    td3(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)