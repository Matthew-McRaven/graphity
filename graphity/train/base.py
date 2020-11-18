import functools
import os

import torch

import graphity.train
import librl.agent.pg as pg

def episodic_trainer(hypers,  task_dist, train_fn):
    for epoch in range(hypers['epochs']):
        task_samples = task_dist.sample(hypers['episode_count'])
        train_fn(hypers, task_samples)

        rewards, mu_act = len(task_samples) * [None],  len(task_samples) * [None]
        for idx, task in enumerate(task_samples):
            mu_act[idx] = torch.mean(task.trajectories[0].action_buffer.type(torch.float32), (0))
            rewards[idx] = sum(task.trajectories[0].reward_buffer.view(-1))

        mean_reward = (sum(rewards)/len(rewards)).item()
        mean_action = functools.reduce(lambda x, y: x+y, mu_act, 0).mean()
        max_action = functools.reduce(lambda x, y: torch.max(x.abs(), y.abs()), mu_act)
        print(f"R^bar_({epoch}) = {mean_reward} with {mean_action:.4f} {max_action.data}.")

def episodic_trainer_ray(hypers, env, agent, task_dist, train_fn):
    from ray import tune
    for epoch in range(hypers['epochs']):
        task_samples = task_dist.sample(hypers['episode_count'])
        train_fn(hypers, env, agent, task_samples)

        rewards, mu_act = len(task_samples) * [None],  torch.zeros((len(task_samples), hypers['output_size']))
        for idx, task in enumerate(task_samples):
            mu_act[idx] = torch.mean(task.trajectories[0].action_buffer, (0))
            rewards[idx] = sum(task.trajectories[0].reward_buffer.view(-1))
        mean_reward = (sum(rewards)/len(rewards)).item()
        mean_action = torch.mean(mu_act**2)
        max_action,_ = torch.max(mu_act.abs(), 0)
        print(f"R^bar_({epoch}) = {mean_reward} with {mean_action:.4f} {max_action.data}.")
        mean_forward, mean_backward = 0, 0
        c_forward, c_backward = 0,0
        for idx, task in enumerate(task_samples):
            if task.start_params[0]:
                mean_forward += rewards[idx]
                c_forward += 1
            else:
                mean_backward += rewards[idx]
                c_backward += 1
        mean_forward, mean_backward = (mean_forward/c_forward).item(), (mean_backward/c_backward).item()
        tune.report(epoch_reward_mean=mean_reward, epoch_reward_mean_forward=mean_forward, epoch_reward_mean_backward=mean_backward)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "model.pth")
            os.makedirs(checkpoint_dir,exist_ok=True)
            torch.save(agent, path)

@functools.singledispatch
def rl_loss(agent, episode_iterable):
    raise NotImplemented(f"Loss not implemented for {type(agent)}")

@rl_loss.register(pg.REINFORCEAgent)
def _(agent, tasks_iterable):
    graphity.train.update_for_agent(agent, tasks_iterable, 1, 'actor_loss', -1, 'actor_optimizer')

@rl_loss.register(pg.ActorCriticAgent)
def _(agent, tasks_iterable):
    graphity.train.update_for_agent(agent, tasks_iterable, agent.hypers['critic_steps'], 'critic_loss', 1, 'critic_optimizer')
    graphity.train.update_for_agent(agent, tasks_iterable, 1, 'actor_loss', -1, 'actor_optimizer')

# Assumes there is no interaction between different agents.
# If components of an agent are shared between tasks, gradient updates are no longer independent.
def basic_task_loop(hypers, task_samples):
    # Determine which agents are present in this training run.
    agents = { (id(task.agent),task.agent) for task in task_samples}
    for task in task_samples: task.sample(task)
    # Collate tasks by the agent running them.
    collated_tasks = {id_agent:[] for id_agent, _ in agents}
    for task in task_samples: collated_tasks[id(task.agent)].append(task)
    for id_agent, agent in agents: rl_loss(agent, collated_tasks[id_agent])

@functools.singledispatch
def meta_loss(agent, task):
    raise NotImplemented("Fail")
@meta_loss.register(pg.REINFORCEAgent)
def _(agent, task):
    return agent.hypers['actor_loss_mul'] * agent.actor_loss(task)
@meta_loss.register(pg.ActorCriticAgent)
def _(agent, task):
    return agent.critic_loss(task), (agent.hypers['actor_loss_mul'] * agent.actor_loss(task))

@functools.singledispatch
def meta_optim_step(agenr):
    raise NotImplemented("Fail")
@meta_optim_step.register(pg.REINFORCEAgent)
def _(agent):
    # TODO: Grab grad clip from agent's hypers.
    #torch.nn.utils.clip_grad_norm(agent.parameters(), 40)
    agent.actor_optimizer.step()
@meta_optim_step.register(pg.ActorCriticAgent)
def _(agent):
    # TODO: Grab grad clip from agent's hypers.
    #torch.nn.utils.clip_grad_norm(agent.parameters(), 40)
    agent.actor_optimizer.step()
    agent.critic_optimizer.step()

# Implement MAML algorithm for meta-RL tasks
# Assumes there is no interaction between different agents.
# If components of an agent are shared between tasks, gradient updates are no longer independent.
def meta_task_loop(hypers, task_samples):
    # Collect all unique agents in our task list.
    agents = { (id(task.agent),task.agent) for task in task_samples}
    # Make sure we do not persist grads between training epochs.
    for _, agent in agents: agent.zero_grad()
    # Keep track of our starting parmeters.
    # Since we have potential redundant agents, only keep one copy of our params per agent.
    slow_parameters = {id_agent : agent.steal() for id_agent, agent in agents}
    # However, each task will have unique parameters after being adapted.
    size = hypers['episode_count']
    adapted_parameters, adapted_grads, adapted_rewards = size*[None], size*[None], size*[None]
    
    ####################
    # Task Adaptation  #
    ####################
    for idx, task in enumerate(task_samples):
        agent = task.agent
        agent.stuff(slow_parameters[id(agent)])
        # Perform an arbitrary number of adaptation steps
        for _ in range(hypers['adapt_steps']): basic_task_loop(hypers, [task])
        
        # Sample a task for the meta-adaptation step and compute the loss based on agent type.
        task.sample(task)
        for loss in meta_loss(agent, task): loss.backward()

        # All pytorch optimizers perform updates using grads.
        # If we copy out the grads and apply them later, pytorch won't be any wiser
        # as to how those grads got there. Use "clone" to prevent grads from being deleted.
        adapted_grads[idx] = [(p.grad.clone().detach() if p.grad != None else None)for p in agent.parameters()]
        adapted_rewards[idx] = functools.reduce(lambda x, y: x+sum(y.reward_buffer.view(-1)), task.trajectories, 0)

        # Prevent meta-grads from polluting the next task's update.
        agent.zero_grad()
        adapted_parameters[idx] = agent.steal()

    ####################
    # Meta-update step #
    ####################
    # We want to apply gradient updates to our starting weights,
    # and we want to discard any existing gradient updates
    for id_agent, agent in agents: agent.stuff(slow_parameters[id_agent]), agent.zero_grad()

    # Apply the i'th task's gradient update to the i'th task's agent.
    for idx, task in enumerate(task_samples):
        agent = task.agent
        for param, grad in zip(agent.parameters(), adapted_grads[idx]):
            # Some parts of a model may be frozen, so we can skip those params.
            if not param.requires_grad or grad is None: continue
            # If our grad object doesn't exist, we must create one in place.
            if param.grad is None: param.grad = grad
            # Otherwise accumulate grads in place.
            else: param.grad.add_(grad)
    
    # And apply those using our existing optimizer.
    for _, agent in agents: meta_optim_step(agent)
