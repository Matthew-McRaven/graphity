"""
This program demonstrates how to train an agent end-to-end.
It demonstrates all available agents.
It also shows how to create a task distribution to sample from.
"""
import torch

import librl.agent.pg
import librl.nn.core, librl.nn.critic, librl.nn.actor
import graphity.agent.markov, graphity.agent.grad
import graphity.grad
import graphity.environment.reward, graphity.environment.sim
import graphity.hypers, graphity.train
import librl.task, graphity.task

# Sample program that demonstrates how to create an agent & environment.
# Then. train this agent for some number of epochs, determined by our hypers.
def main():
	hypers = graphity.hypers.get_default_hyperparams()

	hypers['device'] = 'cpu'
	hypers['adapt_steps'] = 2
	hypers['actor_loss_mul'] = -1
	hypers['actor_layers'] = [28, 14]
	hypers['epochs'] = 100
	hypers['episode_count'] = 4
	hypers['episode_length'] = 100
	hypers['graph_size'] = 6
	hypers['toggles_per_step'] = 1

	# Environment definition
	H = graphity.environment.reward.LogASquaredD(2)
	env = graphity.environment.sim.Simulator(graph_size=hypers['graph_size'], H=H)
	
	# Neural-network based agents
	value_kernel = librl.nn.core.MLPKernel(2*(hypers['graph_size'],))
	value_net = librl.nn.critic.ValueCritic(value_kernel, hypers)
	policy_kernel = librl.nn.core.MLPKernel(2*(hypers['graph_size'],))
	policy_net = librl.nn.actor.BiCategoricalActor(policy_kernel, env.action_space, env.observation_space)

	# Vanilla policy gradient
	#policy_loss = graphity.nn.update_rules.VPG(hypers)
	#agent = graphity.agent.pg.REINFORCEAgent(hypers, policy_net)
	# Actor-critic policy gradient methods.
	# Change policy loss fn to change behavior of agent.
	#policy_loss = graphity.nn.update_rules.PGB(value_net, hypers)
	policy_loss = librl.nn.pg_loss.PPO(value_net, hypers)

	agent = librl.agent.pg.ActorCriticAgent(hypers, value_net, policy_net, policy_loss)

	# Show the NN configuration on the console.
	print(agent)

	dist = librl.task.TaskDistribution()
	# Define different sampling methods for points
	#random_sampler = graphity.task.RandomSampler(hypers['graph_size']**2)
	#checkpoint_sampler = graphity.task.CheckpointSampler(random_sampler) # Suspiciously wrong.
	# Create a single task definition from which we can sample.
	dist.add_task(librl.task.Task.Definition(graphity.task.GraphTask, loss=H, sample_fn=graphity.train.independent_sample, env=env, agent=agent))

	#graphity.train.episodic_trainer(hypers, dist, graphity.train.basic_task_loop)
	graphity.train.episodic_trainer(hypers, dist, graphity.train.meta_task_loop)


if __name__ == "__main__":
	main()