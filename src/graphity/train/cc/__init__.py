import torch
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