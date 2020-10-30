log_actions = True
log_rewards = True
log_state = True
import graphity.replay
def simulate_epoch(hypers, agent, env, logger:graphity.replay.BaseReplay):
    for epoch in range(hypers['epochs']):
        
        if logger.clear_each_epoch:
            logger.clear()

        for episode in range(hypers['episode_count']):
            state = env.reset()
            for t in range(hypers['episode_length']):
                if log_state:
                    # Log current state.
                    logger.log_state(epoch, episode, t, state)
                if agent.probability_based:
                    # Record mu, cov to logger.
                    action, mu, cov = agent.act(state)
                    if log_actions:
                        logger.log_actions(epoch, episode, t, action, mu, cov)
                else:
                    action = agent.act(state)
                    if log_actions:
                        # Log action to logger
                        logger.log_actions(epoch, episode, t, action)
                state, reward = env.step(action)
                if log_rewards:
                    # Log reward at t.
                    logger.log_rewards(epoch, episode, t, reward)
                # Record (state, action, reward, is_done) tuple to logger.
                if agent.allow_callback:
                    agent.act_callback(state, reward)
                print(f"{t}")
        # If the agent is allowed to learn, pass the logger and any replay buffers to the agent so it may learn.
        if agent.allow_update:
            agent.update(logger)
        if logger.summarize_epoch:
            logger.summarize_epoch()
        print(f"Finished epoch {epoch}")