import torch

import graphity.train

# Train on a task distribution using only an actor/critics loss functions.
# There is no meta-update step in this loop
def basic_task_loop(hypers, env, agent, task_dist, reuse_tasks=True):
    task_samples = task_dist.sample(hypers['episode_count']) if reuse_tasks else None
    for epoch in range(hypers['epochs']):
        task_samples = task_dist.sample(hypers['episode_count']) if not reuse_tasks else task_samples
        base_sampler(hypers, env, agent, task_samples)
        if agent.allow_update:
            base_update_critic(hypers, env, agent, task_samples)
            base_update_actor(hypers, env, agent, task_samples)
        print(f"Finished epoch {epoch}")

# TODO: Meta loop is not complete
# Implement MAML algorithm for meta-RL tasks
def meta_task_loop(hypers, env, agent, task_dist):
    for epoch in range(hypers['epochs']):
        # Make sure we do not persist grads between training epochs.
        agent.zero_grad()
        task_samples = task_dist.sample(hypers['episode_count'])

        # Mystr keep track of our starting parmeters and our adapted parmeters.
        slow_parameters = agent.steal()
        adapted_parameters = hypers['episode_count'] * [None]
        adapted_grads = hypers['episode_count'] * [None]
        adaptation_steps = 2

        ####################
        # Task Adaptation  #
        ####################
        for idx, task in enumerate(task_samples):
            agent.stuff(slow_parameters)
            # Perform an arbitrary number of adaptation steps
            for adapt_step in range(adaptation_steps):
                base_sampler(hypers, env, agent, [task])
                base_update_critic(hypers, env, agent, [task])
                base_update_actor(hypers, env, agent, [task])
            # Sample a task for the meta-adaptation step.
            base_sampler(hypers, env, agent, [task])
            # Compute the loss of the adapated agent on the task.
            (-agent.critic_loss(task)).backward()
            (-agent.actor_loss(task)).backward()
            # All pytorch optimizers perform updates using grads.
            # If we copy out the grads and apply them later, pytorch won't be any wiser
            # as to how those grads got there. Use "clone" to prevent grads from being deleted.
            adapted_grads[idx] = [p.grad.clone() for p in agent.parameters()]
            # Prevent out meta-grads from polluting the next task's update.
            agent.zero_grad()
            adapted_parameters[idx] = agent.steal()

        ####################
        # Meta-update step #
        ####################
        # We want to apply gradient updates to our starting weights,
        # and we want to discard any existing gradient updates
        agent.stuff(slow_parameters)
        agent.zero_grad()
        # Accumulate grad updates across tasks.
        for adapted in adapted_grads:
            for param, grad in zip(agent.parameters(), adapted):
                param.grad.add_(grad)
        # And apply those using our existing optimizer.
        agent.critic_optimizer.step()
        agent.actor_optimizer.step()

        print(f"Finished epoch {epoch}")

# Methods to perform non-meta updates to actors and critics
def base_update_critic(hypers, env, agent, task_samples):
    # TODO: Make check for critic more pythonic.
    for i in range(20 if hasattr(agent, 'critic_net') else 0):
        for task in task_samples:
            loss = -agent.critic_loss(task)
            loss.backward()
        # Don't move optimization step into inner task loop, otherwise we will overfit.
        agent.critic_optimizer.step()
        agent.critic_optimizer.zero_grad()

def base_update_actor(hypers, env, agent, task_samples):
    # TODO: Make check for actor more pythonic.
    for task in task_samples:
        for i in range(1 if hasattr(agent, 'actor_net') else 0):
            loss = -agent.actor_loss(task)
            loss.backward()
    # Don't move optimization step into inner task loop, otherwise we will overfit.
    agent.actor_optimizer.step()
    agent.actor_optimizer.zero_grad()

# Methods to fill a task's replay buffer
def base_sampler(hypers, env, agent, task_samples):
    for task in task_samples:
        task.clear_replay()
        graphity.train.sample_trajectories(task, agent, env, hypers)
        #print(task.action_buffer.actions.shape)
        # Let task determine if it wants to suggest different starting point next epoch.
        min_g, min_e, _ = task.min()
        task.sampler.checkpoint(min_g, min_e)

    min_g, min_en, min_t, min_ep = *task_samples[0].min(), 0
    for l_ep, ep in enumerate(task_samples[1:]):
        l_g, l_en, l_t = ep.min()
        if min_en  < l_en:
            min_g, min_en, min_t, min_ep = l_g, l_en, l_t, l_ep
    # Put info about the thing our our screen.
    print(f"Min ({min_ep}, {min_t}): {min_en}")#\n{min_g}")

