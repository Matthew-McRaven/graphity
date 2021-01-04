import graphity
import graphity.agent.core

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
                if isinstance(task.agent, graphity.agent.core.SimulatedAnnealingAgent): task.agent.end_epoch()

            for task in task_samples: logger.log_task(epoch, task)
        logger.close()

    except Exception as e:
        logger.mark_corrupt()
        logger.close()
        raise e
