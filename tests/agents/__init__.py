import librl.task

import graphity.task


def create_task(env, agent, hypers):
    # Define different sampling methods for points
    random_sampler = graphity.task.RandomSampler(env.graph_size)
    #checkpoint_sampler = graphity.task.CheckpointSampler(random_sampler) # Suspiciously wrong.
    dist = librl.task.TaskDistribution()
    # Create a single task definition from which we can sample.
    dist.add_task(librl.task.Task.Definition(graphity.task.GraphTask, sampler=random_sampler, agent=agent, env=env))
    return dist