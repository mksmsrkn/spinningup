**Note:** This repository is an attempt to convert the [OpenAI's SpinningUP](https://github.com/openai/spinningup/) from Tensorflow to Pytorch.

**Status:** In-progress, see the list of what's working at the bottom of the README.

**Status OpenAI:** Active (under active development, breaking changes may occur)

Welcome to Spinning Up in Deep RL with Pytorch! 
===============================================

This is an educational resource originally produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


----------
#### Working:
* Setup process (now in python 3.7)
* VPG (CPU, single GPU)
* PPO (CPU, single GPU)
* DDPG (CPU, single GPU)
* TD3 (CPU, single GPU)

#### TODO (up next):
* Adapt Logger (model/env saver) to Pytorch
* MPI pytorch
* Tensorflow vs Pytorch Benchmark

#### Known issues:
* A difference has been noticed when training a simple environment, for TF default parameters would get Cartpole AverageEpRet to 100  for pytorch it barely gets over 40, (if the lr is changed to 0.01 for both PI and V learners, then it easily reaches 200)
* Network parameter counter doesn't show the same results for TF and Pytorch (e.g. --hid[h] [16, 32] Number of parameters with TF (pi: 690, v: 657), with Pytorch (pi: 226, v: 193) for CartPole)