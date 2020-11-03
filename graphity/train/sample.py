log_actions = True
log_rewards = True
log_state = True
summarize_epoch = False

# Given a task (which contains a replay buffer),
# run the agent on the task until the buffers are full.
def sample_trajectories(task, agent, env, hypers):
    state = env.reset(task.sample_starting_point())

    for t in range(hypers['episode_length']):
        if log_state:
            task.log_state(t, state)

        action, logprob_action = agent.act(state, hypers['toggles_per_step'])

        if log_actions:
            task.log_action(t, action, logprob_action)
            if agent.policy_based:
                task.log_policy(t, agent.policy_latest)

        state, reward = env.step(action)

        if log_rewards:
            task.log_rewards(t, reward)

        # Allow the agent to receive immediate feedback from the environment
        # if it is requested.
        if agent.allow_callback:
            agent.act_callback(state, reward)
        if summarize_epoch:
            assert 0