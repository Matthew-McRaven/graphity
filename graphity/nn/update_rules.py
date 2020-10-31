import torch
import torch.distributions, torch.nn.init
import torch.optim

import graphity.agent
import graphity.replay

# Vanilla policy gradient update / loss function.
class VPG:
    def __init__(self, gamma=.9):
        self.gamma = gamma
    def __call__(self, state_buffer, action_buffer, reward_buffer, policy_buffer):
        discounted = graphity.replay.ReturnAccumulator(reward_buffer, gamma=self.gamma)
        ret = torch.zeros((reward_buffer.episode_count,))
        for episode in range(reward_buffer.episode_count):
            episode_states = state_buffer.states[episode, :]
            episode_states = episode_states.view(*(episode_states.shape[0:1]),-1)
            ret[episode] = torch.sum(action_buffer.logprob_actions[episode, :]
                                     * discounted.discounted_rewards[episode, :])
        # Perform outer summation and divide by number of terms.
        loss = torch.mean(ret)
        return loss
        
# Policy gradient with baseline update / loss function.
class PGB:
    def __init__(self, critic_net, hypers):
        self.critic_net = critic_net
        self.gamma = hypers['gamma']
    def __call__(self, state_buffer, action_buffer, reward_buffer, policy_buffer):
        discounted = graphity.replay.ReturnAccumulator(reward_buffer, gamma=self.gamma)
        ret = torch.zeros((reward_buffer.episode_count,))
        for episode in range(reward_buffer.episode_count):
            episode_states = state_buffer.states[episode, :]
            episode_states = episode_states.view(*(episode_states.shape[0:1]),-1)
            value = self.critic_net(episode_states)
            ret[episode] = torch.sum(action_buffer.logprob_actions[episode, :]
                                     * discounted.discounted_rewards[episode, :] - value)
        # Perform outer summation and divide by number of terms.
        loss = torch.mean(ret)
        return loss

# Proximal policy optimization update / loss function.
class PPO:
    def __init__(self, critic_net, hypers):
        self.critic_net = critic_net
        # Save hyperparameters
        self.hypers = hypers
        self.gamma = hypers['gamma']
        self.lambd = hypers['lambda']
        self.epsilon = hypers['epsilon']
        self.c = hypers['c_1']

    def __call__(self, state_buffer, action_buffer, reward_buffer, policy_buffer):
        # I compute A and pi_old(a_it | s_it) in a seperate vectorized loop
        # rather than computing it as I need them.
        # This is a time/space complexity tradeoff, but it seems to work in practice.
        A =  torch.zeros([reward_buffer.episode_count, reward_buffer.episode_len, 1],
                            dtype=torch.float32, device=reward_buffer.rewards.device)
        log_prob_old = torch.zeros([action_buffer.episode_count, action_buffer.episode_len, 1],
                            dtype=torch.float32, device=action_buffer.logprob_actions.device)
        # Store the predicted values of the network in a list rather than a tensor,
        # since each episode may be a different length
        value_list = []
        for episode in range(state_buffer.episode_count):
            episode_states = state_buffer.states[episode, :]
            episode_states = episode_states.view(*(episode_states.shape[0:1]),-1)
            value_list.append(self.critic_net(episode_states))
        #print(value_list)
        previous = torch.zeros((1,), device=state_buffer.states.device)
        for episode in range(reward_buffer.episode_count):
            value = value_list[episode]
            # See my notes on utils.ReturnAccumulator for validty of computing array backwards.
            # This approach was re-used because it moves python operations to vectorized C operations.
            for t in range(reward_buffer.episode_len - 2, -1, -1):
                reward = (reward_buffer.rewards[episode][t]
                            + self.gamma * value[t+1] - value[t])
                previous = A[episode, t] = reward + self.lambd*self.gamma*previous
                # Compute the log probability of the state/action pair using the previous steps' policy.
                pi_old = policy_buffer.policies[episode][t]
                new_actions = action_buffer.actions[episode, t]
                if len(new_actions.shape) == 1:
                    new_actions = new_actions.view(1,-1)
                # Dealing in indepedent log probs, so sum instead of multiply.
                log_prob_old[episode, t+1] = pi_old.log_prob(new_actions).sum()

        # Now that I've pre-computed \hat{A} and all rewards, I can vectorize summation by episode.
        ret = torch.zeros((action_buffer.episode_count,))
        for episode in range(action_buffer.episode_count):
            # Probabilities stored in log domain, so we must exponentiate.
            # Add the 1e-6 term since log_prob_old may approach -inf, meaning the exp(old)=>0.
            ratio = action_buffer.logprob_actions[episode, :].exp() / (log_prob_old[episode, :].exp() + 1e-6)
            # Compute the left arg and right arg of the min expression.
            lhs = ratio * A[episode, :]
            rhs = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * A[episode, :]
            # Minimized term in  objective function for PPO.
            minterm = torch.min(lhs, rhs)
            # Subtracted term in objective function for PPO.
            subterm = self.c * (value_list[episode]).pow(2)
            ret[episode] = torch.sum(minterm - subterm)
        # Perform outer summation and divide by number of terms.
        loss = torch.mean(ret)
        print(f"My loss is: {loss}")
        return loss