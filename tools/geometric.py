import torch

import graphity.environment.reward, graphity.environment.sim
import graphity.agent.markov

def main(args):
    # Allocate simulation and agent.
    agent = graphity.agent.markov.RandomAgent()
    env = graphity.environment.sim.Simulator(graph_size=args.size)

    # Initialize simulation to random graph.
    state = env.reset()
    done, step, reward = False, 0, float('inf')
    
    # Record the per-trial minimum energy and lowest energy state.
    # Hopefully it can be shown to be geometri
    min_reward, min_state = env.H(state), state

    while step < args.max_steps:
        step += 1
        # Generate a new action and take it.
        action = agent(state)
        state, reward = env.step(action)
        # If we run for a long time, be kind and provide proof of forward progress.
        if step % 10_000 == 0:
            print(f"A total of {step} steps have elapsed. Best is {min_reward}.")
        # Record if current state is a new min.
        if reward < min_reward:
            min_reward, min_state = reward, state

    print(min_state, min_reward)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='This program generates a random graph of size (size), and randomly toggles edges.')
    parser.add_argument('--size', type=int, default=8,
                        help='Number of nodes in the graph.')
    parser.add_argument('--max-steps', dest='max_steps', default=1_000,
                        help='Number of toggle steps to perform.')
    parser.add_argument('--print-steps', dest='print_steps', default=10_000,
                        help='Print progress to console after this many steps elapse.')
    main(parser.parse_args())