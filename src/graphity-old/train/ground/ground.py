import graphity

from argparse import ArgumentParser
import os
import itertools
import functools

import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.distributions import Categorical
from numpy.random import Generator, PCG64

import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger

import numpy as np
from numpy.random import default_rng
import torch.tensor

def group_trainer(train_info, task_dist, train_fn, logger):
    try:
        logger.log_metainfo(train_info)
        for epoch in range(train_info['epochs']):
            env, task_samples = train_info['env'], task_dist.gather()

            env.reset_sampler()
            seed = env.reset()

            logger.log_seed(epoch, seed)
            train_fn(task_samples)
            # Reset the timestep counter for annealing agents.
            for task in task_samples:
                if isinstance(task.agent, graphity.agent.mdp.SimulatedAnnealingAgent): task.agent.end_epoch()

            for task in task_samples: logger.log_task(epoch, task)
        logger.close()

    except Exception as e:
        logger.mark_corrupt()
        logger.close()
        raise e



def train_ground_search(config, dist, train_fn, logger):
    print("Reached here!!")
    tasks = dist.gather()
    adapt_steps = range(1)
    def run_single_timestep(engine, timestep):
        for task in tasks:
            task.sample(task, epoch=engine.state.epoch)
        engine.state.timestep = timestep

    trainer = ignite.engine.Engine(run_single_timestep)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        pass

    @trainer.on(Events.EPOCH_STARTED)
    def reset_environment_state(engine):
        for task in tasks: task.env.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_agents(engine):
        for task in tasks:
            train_fn(task)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trajectories(engine):
        for task in tasks: logger.log_task(engine.state.epoch, task)


    trainer.run(adapt_steps, max_epochs=config['epochs'])

def run(config):
    env = graphity.environment.SpinGlassSimulator()
    ss = graphity.strategy.random_sampling_strategy(1)
    agent = graphity.agent.core.ForwardAgent(ss)
    tasks = []
    train_ground_search(config, tasks)
