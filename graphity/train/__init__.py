log_actions = True
log_rewards = True
log_state = True
summarize_epoch = False

import graphity.replay
import torch

# TODO: use logger to report important information during training.
# TODO: Implement k-fold validation acrss trajectories.
# TODO: Allow user to signal if they want to allocate replay buffers.
def simulate_epoch(hypers, agent, env, logger=None):
    # TODO: Only allocate a replay buffer if is needed AND requested.
    state_buffer = graphity.replay.StateBuffer(hypers['episode_count'], hypers['episode_length'], (hypers['graph_size'], hypers['graph_size']))
    action_buffer = graphity.replay.ActionBuffer(hypers['episode_count'], hypers['episode_length'], (2,))
    reward_buffer = graphity.replay.RewardBuffer(hypers['episode_count'], hypers['episode_length'], (1,))
    policy_buffer = None

    # Don't allocate buffers for policy replay if it is unused. Can waste a bunch of memory.
    if agent.policy_based:
        policy_buffer = graphity.replay.PolicyBuffer(hypers['episode_count'], hypers['episode_length'])

    for epoch in range(hypers['epochs']):

        # Clear all buffers at start of epoch.
        # Failing to do so will anger the optimizer.
        state_buffer.clear()
        action_buffer.clear()
        reward_buffer.clear()
        if policy_buffer != None:
            policy_buffer.clear()

        for episode in range(hypers['episode_count']):
            state = env.reset()
            for t in range(hypers['episode_length']):

                if log_state:
                    # Log current state.
                    state_buffer.log_state(episode, t, state)

                action, logprob_action = agent.act(state)

                if log_actions:
                    # Log action to logger
                    action_buffer.log_action(episode, t, action, logprob_action)
                    if agent.policy_based and policy_buffer != None:
                        policy_buffer.log_policy(episode, t, agent.policy_latest)

                state, reward = env.step(action)

                # Log reward at t.
                if log_rewards:
                    reward_buffer.log_rewards(episode, t, reward)

                # Allow the agent to receive immediate feedback from the environment
                # if it is requested/
                if agent.allow_callback:
                    agent.act_callback(state, reward)

        # If the agent is allowed to learn, assume it needs all replay buffers.
        if agent.allow_update:
            agent.update(state_buffer, action_buffer, reward_buffer, policy_buffer)

        if summarize_epoch:
            assert 0

        # Find the state with the smallest energy
        mindex = torch.argmin(reward_buffer.rewards)
        # Compute the episode, time of the minimum energy
        episode, t = mindex // reward_buffer.episode_len, mindex % reward_buffer.episode_len
        # Put info about the thing our our screen.
        print(f"({episode},{t}):{reward_buffer.rewards[episode,t]}\n{state_buffer.states[episode, t]}")
        print(f"Finished epoch {epoch}")