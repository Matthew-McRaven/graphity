log_actions = True
log_rewards = True
log_state = True
log_policies = True
summarize_epoch = False

import graphity.replay
import torch

def simulate_epoch(hypers, agent, env, state_buffer, action_buffer, reward_buffer, policy_buffer=None):
    for epoch in range(hypers['epochs']):
        # Clear all buffers at start of epoch
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

                if log_rewards:
                    # Log reward at t.
                    reward_buffer.log_rewards(episode, t, reward)
                # Record (state, action, reward, is_done) tuple to logger.
                if agent.allow_callback:
                    agent.act_callback(state, reward)
                #print(f"{t}")

        # If the agent is allowed to learn, pass the logger and any replay buffers to the agent so it may learn.
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