import argparse
import enum
import functools
import pickle
import os

import librl.agent.pg
import librl.agent.mdp
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import numpy as np
import pytest
import torch

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.reward, librl.replay.episodic
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc

import graphity.agent.mdp
import graphity.agent.grad
import graphity.environment.sim
import graphity.environment.reward
import graphity.task.task
import graphity.task.sampler

#######################
#       Logging       #
#######################
class DirLogger:
    def __init__(self, H, log_dir=None):
        self.log_dir = log_dir
        self.H = H
        if log_dir:
            os.makedirs(self.log_dir)

    def __call__(self, epochs, task_samples):
        energy_list = []
        for task_idx, task in enumerate(task_samples):
            for trajectory_idx, trajectory in enumerate(task.trajectories):
                rewards = [trajectory.reward_buffer[idx]for idx in range(min(trajectory.done, len(trajectory.reward_buffer)))]
                rewards = torch.stack(rewards)
                mindex = torch.argmin(rewards)
                energy_list.append(np.exp(trajectory.reward_buffer[mindex].item()))
                #print(trajectory.reward_buffer)

        if self.log_dir:
            subdir = os.path.join(self.log_dir, f"epoch-{epochs}")
         # I would be very concerned if the subdir already exists
            os.makedirs(subdir)
            for task_idx, task in enumerate(task_samples):
                task_subdir = os.path.join(subdir, f"task-{task_idx}")
                os.makedirs(task_subdir)
                for trajectory_idx, trajectory in enumerate(task.trajectories):
                    with open(os.path.join(task_subdir, f"traj{trajectory_idx}.pkl"), "wb") as fptr:
                        pickle.dump(trajectory, fptr)

        rewards = len(task_samples) * [None]
        for idx, task in enumerate(task_samples): rewards[idx] = sum(task.trajectories[0].reward_buffer)
        print(f"R^bar_({epochs:04d}) = {(sum(rewards)/len(rewards)).item():07f}. Best was {round(min(energy_list))}.")


######################
#    Init Agents     #
######################
def random_helper(hypers, env, *args):
    agent = librl.agent.mdp.RandomAgent(env.observation_space, env.action_space)
    return agent

def metropolis_helper(hypers, env, *args):
    agent = graphity.agent.mdp.MetropolisMarkovAgent()
    return agent

def grad_follower_helper(hypers, env, *args):
    agent = graphity.agent.grad.GradientFollowingAgent(env.H)
    return agent

def vpg_helper(hypers, env, critic_net, policy_net):
    agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent.train()
    return agent
    

def pgb_helper(hypers, env, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PGB(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

def ppo_helper(hypers, env, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PPO(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

#######################
#  Init Environments  #
#######################
def build_env(hypers):
    sampler = graphity.task.sampler.CachedSampler(graph_size=hypers['n'])
    env = graphity.environment.sim.Simulator(hypers['H'], hypers['n'], sampler=sampler)
    flatten = graphity.FlattenInput(env.observation_space.shape)
    policy_kernel = librl.nn.core.RecurrentKernel(flatten.output_dimension, 512, 10)
    policy_linked = librl.nn.core.SequentialKernel([flatten, policy_kernel])
    policy_net = librl.nn.actor.BiCategoricalActor(policy_linked, env.action_space, env.observation_space)
    x = flatten.output_dimension
    critic_kernel = librl.nn.core.RecurrentKernel(flatten.output_dimension, 512, 10)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return env, critic_net, policy_net 

#######################
#  Train Loop  #
#######################
def cc_episodic_trainer(train_info, task_dist, train_fn, log_fn):
    for epoch in range(train_info['epochs']):
        task_samples = task_dist.sample(train_info['task_count'])
        envs = {}
        for task in task_samples:
            if id(task.env) in envs: pass
            else: envs[id(task.env)] = task.env
        #for env in envs: envs[env].reset_sampler()
        train_fn(task_samples)
        log_fn(epoch, task_samples)

#######################
#     Entry point     #
#######################
def main(args):
    hypers = {}
    hypers['H'] = args.H
    hypers['n'] = args.n
    hypers['device'] = 'cpu'
    hypers['epochs'] = args.epochs
    hypers['task_count'] = args.task_count
    hypers['episode_length'] = args.episode_length

    env, critic, actor = build_env(hypers)
    critic, actor = critic.to(hypers['device']), actor.to(hypers['device'])
    agent = args.alg(hypers, env, critic, actor)
    print(agent)

    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(graphity.task.task.GraphTask, env=env, agent=agent, sampler=env.sampler))
    cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step, log_fn=DirLogger(args.H, args.log_dir))


# Invoke main, construct CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generator network on a task for multiple epochs and record results.")
    parser.add_argument("--log-dir", dest="log_dir", help="Directory in which to store logs.")
    # Choose RL algorthim.
    learn_alg_group = parser.add_mutually_exclusive_group()
    learn_alg_group.add_argument("--random", action='store_const', const=random_helper, dest='alg', help="Perform a random search of the state space.")
    learn_alg_group.add_argument("--metropolis", action='store_const', const=metropolis_helper, dest='alg', help="Use a MCMC method to back out bad actions.")
    learn_alg_group.add_argument("--grad", action='store_const', const=grad_follower_helper, dest='alg', help="Search state space using grad descent.")
    learn_alg_group.add_argument("--vpg", action='store_const', const=vpg_helper, dest='alg', help="Train a RL agent using VPG.")
    learn_alg_group.add_argument("--pgb", action='store_const', const=pgb_helper, dest='alg', help="Train a RL agent using PGB.")
    learn_alg_group.add_argument("--ppo", action='store_const', const=ppo_helper, dest='alg', help="Train a RL agent using PPO.")
    learn_alg_group.set_defaults(alg=vpg_helper)
    
    # Hamiltonian choices
    hamiltonian_group = parser.add_mutually_exclusive_group()
    hamiltonian_group.add_argument("--masked-a2d", action='store_const', const=graphity.environment.reward.LogASquaredD(2), dest='H', help="Mask out diagonal (default) when computing H.")
    hamiltonian_group.add_argument("--unmasked-a2d", action='store_const', const=graphity.environment.reward.LogASquaredD(2, keep_diag=True), dest='H', help="Keep diagonal when computing H.")
    hamiltonian_group.set_defaults(H=graphity.environment.reward.LogASquaredD(2))
    
    # Task distribution hyperparams.
    parser.add_argument("-n", default=6, type=int, help="Number of nodes in graph.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs for which to train / evaluate agents.")
    parser.add_argument("--task-count", dest="task_count", default=5, type=int, help="Numbers of episodes (trial runs) in each epoch. Within each epoch, each task is presented with the same seed graph")
    parser.add_argument("--timesteps", dest="episode_length", default=100, type=int, help="Number of evolution steps for each task.")
    args = parser.parse_args()
    main(args)

#######################
#  Integration Tests  #
#######################
@pytest.mark.parametrize("alg", ["vpg", "pgb"])
@pytest.mark.parametrize("type", ["mlp","cnn", "joint"])
def test_all(alg, type):
    from types import SimpleNamespace

    args = {}
    args['epochs'] = 100
    args['task_count'] = 1
    args['episode_length'] = 100
    args['adapt_steps'] = 3

    if alg == "vpg": args['alg'] = vpg_helper
    elif alg == "pgb": args['alg'] = pgb_helper
    elif alg == "ppo": args['alg'] = ppo_helper

    if type == "mlp": args['type'] = build_mlp
    elif type == "cnn": args['type'] = build_cnn
    elif type == "joint": args['type'] = build_joint
    
    args['log_dir'] = f"test-LSTM-{alg}-{type}"
    
    return main(SimpleNamespace(**args))