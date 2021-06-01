# All imports from python standard library. Alphabatized, one per line.
# Which are empty for now...

# All imports from the standard library. Alphabatized, one per line.
import functools

# All imports from modules we got from pip. Alphabatized, one per line.
import matplotlib.pyplot as plt
import networkx as nx
import torch

# All imports from modules we wrote ourselves. Alphabatized, one per line.
import graphity.agent.det
import graphity.environment.lattice
from graphity.strategy.anneal import ConstBeta
import graphity.task
import graphity.strategy.site

def main(args):
    lattice_shape = (args.size, args.size)
    # Allocate simulation and agent.
    H = graphity.environment.lattice.IsingHamiltonian()
    random_sampler = graphity.task.RandomGlassSampler(lattice_shape)
    ss = graphity.strategy.site.RandomSearch()
    agent = graphity.agent.det.ForwardAgent(ConstBeta(args.beta), ss)
    env = graphity.environment.lattice.RejectionSimulator(glass_shape=lattice_shape, H=H)

    # Initialize simulation to random graph.
    state, delta_e = env.reset()
    done, step, reward = False, 0, float('inf')
    
    # Record the per-trial minimum energy and lowest energy state.
    # Hopefully it can be shown to be geometric
    min_reward, _ = env.H(state, )
    min_state = state

    while step < args.max_steps:
        step += 1
        # Generate a new action and take it.
        action, logprob_action = agent(state, delta_e)
        state, delta_e, reward, _, extra_info = env.step(action)

        # If we run for a long time, be kind and provide proof of forward progress.
        if step % 10_000 == 0:
            print(f"A total of {step} steps have elapsed. Best is {min_reward}.")
            
        # Record if current state is a new min.
        if reward < min_reward:
            min_reward, min_state = reward, state

    print(min_state, min_reward)
    # Since we may be working on the GPU, we must explictly transfer to CPU before moving to numpy.
    # Must view(...) changes the dimensions of the reward tensor from 1xnxn to nxn.
    # Detach deletes any stored gradient information  (important when using machine learning!)
    # CPU forces tensor to CPU, which is required to convert from a tensor to a numpy array.
    adj = min_state.view(args.size, -1).detach().cpu().numpy()
    # Convert from adjacency matrix to NetworkX object.
    # See documentation for information about the library:
    #    https://networkx.github.io/documentation/stable/
    #    https://networkx.github.io/
    #    https://pypi.org/project/networkx/
    # If you have questions about what algorithmsare implemented on these graphs, see:
    #    https://networkx.github.io/documentation/stable/reference/index.html
    as_graph = nx.from_numpy_matrix(adj)

    # Drawing example taken from:
    #    https://networkx.github.io/documentation/latest/auto_examples/drawing/plot_weighted_graph.html
    pos = nx.spring_layout(as_graph) 
    # Draw nodes & edges.
    nx.draw_networkx_nodes(as_graph, pos, node_size=700)
    nx.draw_networkx_edges(as_graph, pos, width=6)
    # Render the graph to the screen.
    plt.axis("off")
    plt.show()

# Python idiom that prevents main() from being called unless this file was called directly from command line.
# If this file is grabbed via import, main() will not execute automatically.
if __name__ == "__main__":
    # Import a library that will let use create a command line parser.
    import argparse
    parser = argparse.ArgumentParser(description='This program generates a random graph of size (size), and randomly toggles edges.')
    parser.add_argument('--size', type=int, default=8,
                        help='Number of nodes in the graph.')
    parser.add_argument('--beta', type=int, default=1/2,
                        help='Inverse temperature of the system.')
    parser.add_argument('--max-steps', dest='max_steps', default=1_000,
                        help='Number of toggle steps to perform.')
    parser.add_argument('--print-steps', dest='print_steps', default=10_000,
                        help='Print progress to console after this many steps elapse.')
    main(parser.parse_args())