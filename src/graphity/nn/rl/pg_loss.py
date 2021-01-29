import torch
import torch.distributions, torch.nn.init
import torch.optim
from overrides import overrides, EnforceOverrides

import librl.calc
import librl.reward
import librl.task
##########################
# Policy Gradient Losses #
##########################

# Compute the loss of a multi-trajecory task.
class PolicyLoss(EnforceOverrides):
    def __init__(self):
        super(PolicyLoss, self).__init__()

    # Override this method to compute the loss of a single trajectory.
    # Should return a single value.
    def compute_trajectory_loss(self, trajectory):
        raise NotImplementedError("Implement in subclass")

    # When called with a CC task, run the loss algorithm over each trajectory in the task.
    # This will implicitly perform the outer mean over the the number of trajectories,
    # and defer the computation of per-trajectory loss to a subclass via compute_trajectory_loss.
    def __call__(self, task):
        assert isinstance(task, librl.task.ContinuousControlTask)
        losses = []
        for trajectory in task.trajectories: losses.append(self.compute_trajectory_loss(trajectory))
        return sum(losses) / len(losses)

# For any class base on log_prob*reward, this class is convenient.
# Pass in a function(trajectory)->array[rewards], and this will do the rest.
# Especially useful for VPG, PGB
class LogProbBased(PolicyLoss):
    def __init__(self, reward_fn, explore_bonus_fn=lambda _:0):
        super(LogProbBased, self).__init__()
        assert callable(reward_fn) and callable(explore_bonus_fn)
        self.reward_fn = reward_fn
        self.explore_bonus_fn = explore_bonus_fn
    
    @overrides # type: ignore
    def compute_trajectory_loss(self, trajectory):
        lp_accumulator, reward = 0, self.reward_fn(trajectory)
        if(torch.is_tensor(trajectory.logprob_buffer)):
            lp_accumulator =  trajectory.logprob_buffer[:trajectory.done] * reward
        else:
            for idx in range(min(trajectory.done, len(reward))):
                lp_accumulator += trajectory.logprob_buffer[idx] * reward[idx]
        return sum(lp_accumulator + self.explore_bonus_fn(trajectory))
            

class VPG(LogProbBased):
    def __init__(self, gamma=0.975, explore_bonus_fn=lambda _: 0): 
        assert callable(explore_bonus_fn)
        super(VPG, self).__init__(librl.reward.to_go_reward(gamma=gamma), explore_bonus_fn=explore_bonus_fn)
class PGB(LogProbBased):
    def __init__(self, critic_fn, gamma=0.975, explore_bonus_fn=lambda _: 0): 
        assert callable(explore_bonus_fn)
        super(PGB, self).__init__(librl.reward.baseline_to_go(critic_fn, gamma=gamma), explore_bonus_fn=explore_bonus_fn)

# Proximal policy optimization update / loss function.
class PPO(PolicyLoss):
    def __init__(self,  critic_fn, gamma=0.975, lambd=.99, epsilon=.2, c_1=.5, explore_bonus_fn = lambda _: 0):
        assert callable(explore_bonus_fn)
        self.critic_fn = critic_fn
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.c_1 = c_1
        self.explore_bonus_fn = explore_bonus_fn

    @overrides # type: ignore
    def compute_trajectory_loss(self, trajectory):
        device = trajectory.reward_buffer.device
        # Don't propogate gradients into critic when updating actor.
        with torch.no_grad(): estimated_values = librl.reward._estimate_values(trajectory, self.critic_fn)
        # Augment rewards with a bonus for exploration, like policy entropy.
        augmented_rewards = trajectory.reward_buffer[:trajectory.done] + self.explore_bonus_fn(trajectory)
        discounted = librl.calc.discounted_returns(augmented_rewards, gamma=self.gamma)
        A =  librl.calc.gae(trajectory.reward_buffer[:trajectory.done], estimated_values, self.gamma)
        log_prob_old = librl.calc.old_log_probs(trajectory.action_buffer[:trajectory.done], trajectory.policy_buffer[:trajectory.done], device)
        # Compute indiviudal terms of the PPO algorithm.
        ratio = trajectory.logprob_buffer[:trajectory.done].exp() / (log_prob_old.exp() + 1e-6)
        lhs, rhs = ratio * A, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * A # type: ignore
        minterm = torch.min(lhs, rhs) # type: ignore
        subterm = self.c_1 * (discounted-estimated_values.view(-1)).pow(2)
        return torch.sum(minterm - subterm) # type: ignore