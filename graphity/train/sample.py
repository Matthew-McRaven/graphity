import functools

import torch

import graphity.replay

# Given a task (which contains a replay buffer),
# run the agent on the task until the buffers are full.
def sample_trajectories(task, agent, env, hypers):
    for i in range(task.trajectory_count):
        state = torch.tensor(env.reset()).to(hypers['device'])
        episode = graphity.replay.Episode(env.observation_space, env.action_space, hypers['episode_length'])

        for t in range(hypers['episode_length']):
            
            episode.log_state(t, state)

            action, logprob_action = agent.act(state)
            episode.log_action(t, action, logprob_action)
            if agent.policy_based: episode.log_policy(t, agent.policy_latest)
            x = action.view(-1).detach().cpu().numpy()
            state, reward, done, _ = env.step(x)
            state, reward = torch.tensor(state).to(hypers['device']), torch.tensor(reward).to(hypers['device'])

            episode.log_rewards(t, reward)
            if done or t+1==hypers['episode_length']: episode.log_done(t+1)
        task.add_trajectory(episode)

# Methods to fill a task's replay buffer
def independent_sample(task):
    task.clear_trajectories()
    task.init_env()
    sample_trajectories(task, task.agent, task.env, task.agent.hypers)

def update_for_agent(agent, tasks, it, loss_fn, loss_mul, optim):
    for i in range(it):
        # Iterate across all tasks, accumulating losses using the current task's agent's loss function.
        # Must init reduce with 0 or else first x is an task, not a scalar.
        # Provide lambda returning 0 in case the specified loss function DNE.
        losses = functools.reduce(lambda x,y: x + getattr(agent, loss_fn, lambda _: 0)(y), tasks, 0)
        # Compute mean and apply optional loss multiplier.
        losses *= (loss_mul / functools.reduce(lambda x,y: x + y.trajectory_count, tasks, 0))
        losses.backward()

        torch.nn.utils.clip_grad_norm_(agent.parameters(), 5)
        getattr(agent, optim).step(), agent.zero_grad()