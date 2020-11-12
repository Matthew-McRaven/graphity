# All imports from python standard library. Alphabatized, one per line.
# Which are empty for now...

# All imports from modules we got from pip. Alphabatized, one per line.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

# All imports from modules we wrote ourselves. Alphabatized, one per line.
import graphity.environment.reward
import graphity.environment.sim
import graphity.agent.markov

# A function to check whether a graph is a complete graph or not
# returns 1 if it is, -1 if not.
def is_complete_graph(gr):
    n = gr.number_of_nodes() 
    vlist=list(gr.nodes) 
    degs = np.array([gr.degree[i] for i in vlist])
    # print('degs ',degs)
    if np.sum(degs==n-1) != n: 
        return -1
    return 1

# A function to check whether a graph is a cycle graph or not
# returns 1 if it is, -1 if not.
def is_cycle_graph(gr):
    if not nx.is_connected(gr):
        return -1
    n = gr.number_of_nodes() 
    vlist = list(gr.nodes) 
    degs = np.array([gr.degree[i] for i in vlist])
    # print('degs ',degs)
    if not (np.sum(degs==2) == n ): 
        return -1
    return 1

# A function to check whether a graph is a path graph or not
# returns 1 if it is, -1 if not.
def is_path_graph(gr):
    if is_cycle_graph(gr)==1: return 1      #cycles are paths
    n = gr.number_of_nodes() 
    vlist=list(gr.nodes) 
    degs = np.array([gr.degree[i] for i in vlist])
    # print('degs ',degs)
    if not (np.sum(degs==2) == n-2 & np.sum(degs==1) == 2 ): 
        return -1
    return 1

# Tests whether a graph is a d-sphere or not.
# Returns the dimension d of the sphere if graph is d-sphere, returns -1
# otherwise. 

def is_dsphere(gr):
    n = gr.number_of_nodes()    
    if n==2:
        if gr.number_of_edges()==0:
            return 0                    # a graph with 2 isolated nodes is a 0-sphere
    vlist = list(gr.nodes)

    # print('vlist ', vlist)

    # check whether the graph is pure or not
    clqs = list(nx.find_cliques(gr))
    numOfclqs = len(clqs)
    # print('clqs ', clqs)
    # print('num of clqs ',len(clqs))

    clqsizes = np.zeros(numOfclqs)
    for i in range(numOfclqs):
        clqsizes[i] = len(clqs[i])

    # print('clqsizes ',clqsizes)

    if np.sum(clqsizes==clqsizes[0]) != numOfclqs: 
        return -1           #graph is not pure, so is not geometric
    else:
        dim = clqsizes[0] 
        if dim==2:
            if nx.is_tree(gr) or nx.is_forest(gr):
                return -1       # trees and forests are not geometric
            elif n > 3:
                return 1        # n cycles are geometric 1-graphs for n > 3

        # test geometricness recursively on the unit spheres
        unitSpheres = []
        for v in vlist:
            unitSpheres.append(gr.subgraph(gr.neighbors(v)))

        # for i in range(n):
        #     print('unit spheres ',list(unitSpheres[i].nodes))
        
        truthVal = 1
        uSphIndex = 0
        while uSphIndex < n: 
            truthVal = is_dsphere(unitSpheres[uSphIndex])
            if truthVal == -1:
                return -1
            uSphIndex += 1
        return dim - 1



# Tests whether a graph is a geometric d-graph or not.
# Returns the dimension d if the graph is a d-graph, returns -1 
# otherwise

def is_dgraph(gr):
    n = gr.number_of_nodes()
    e = gr.number_of_edges()
    vlist = list(gr.nodes())
    if e == 0:
        if n==2: return 0       # the disconnected graph with 2 nodes is a 0-sphere 
        return -1               # disconnected graphs are not geometric for n != 2 
    if is_complete_graph(gr)==1: 
        return -1      #complete graphs are not grometric

    # check whether the graph is pure or not
    clqs = list(nx.find_cliques(gr))
    numOfclqs = len(clqs)
    # print('clqs ', clqs)
    # print('num of clqs ',len(clqs))

    clqsizes = np.zeros(numOfclqs)
    for i in range(numOfclqs):
        clqsizes[i] = len(clqs[i])

    # print('clqsizes ',clqsizes)

    if np.sum(clqsizes==clqsizes[0]) != numOfclqs: 
        return -1           # graph is not pure, so is not geometric
    else:
        dim = clqsizes[0] 
        if dim == 2: 
            return is_path_graph(gr) # a path graph is a 1-graph

        # test geometricness recursively on the unit spheres
        unitSpheres = []
        for v in vlist:
            unitSpheres.append(gr.subgraph(gr.neighbors(v)))

        # for i in range(n):
        #     print('unit spheres ',list(unitSpheres[i].nodes))
        
        truthVal = 1
        uSphIndex = 0
        while uSphIndex < n: 
            truthVal = is_dgraph(unitSpheres[uSphIndex])
            if truthVal == -1:
                return -1
            uSphIndex += 1
        return dim - 1

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
    # pos = nx.spring_layout(as_graph) 
    # # Draw nodes & edges.
    # nx.draw_networkx_nodes(as_graph, pos, node_size=700)
    # nx.draw_networkx_edges(as_graph, pos, width=6)
    # Render the graph to the screen.
    # plt.axis("off")
    # plt.show()
 
    sph=nx.from_numpy_matrix(np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, \
1, 0, 1], [0, 1, 1, 0, 1, 1, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0], [1, 0, \
1, 1, 1, 0, 0, 1], [1, 1, 0, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 1, 0]]))

#     spos = nx.spring_layout(sph) 
#    # Draw nodes & edges.
#     nx.draw_networkx_nodes(sph, spos, node_size=700)
#     nx.draw_networkx_edges(sph, spos, width=6)
#     # Render the graph to the screen.
#     plt.axis("off")
#     # plt.show()

    disk=nx.from_numpy_matrix(np.array([[0, 0, 0, 0, 1, 1, 1, 1],[0, 0, 1, 1, 0, 0, 1, 1],[0, 1, 0, 1, 0, \
1, 0, 1],[0, 1, 1, 0, 1, 0, 1, 0],[1, 0, 0, 1, 0, 1, 1, 0],[1, 0, \
1, 0, 1, 0, 0, 1],[1, 1, 0, 1, 1, 0, 0, 1],[1, 1, 1, 0, 0, 1, 1, 0]]))
    dpos = nx.spring_layout(disk) 
   # Draw nodes & edges.
    nx.draw_networkx_nodes(disk, dpos, node_size=700)
    nx.draw_networkx_edges(disk, dpos, width=6)
    # Render the graph to the screen.
    plt.axis("off")
    # plt.show()
    # print('is_dsphere check on sph', is_dsphere(sph))
    # print('is_dsphere check on disk', is_dsphere(disk))

    k5=nx.complete_graph(5)

    print('is_dgraph check on sph', is_dgraph(sph))
    print('is_dgraph check on disk', is_dgraph(disk))
    print('is_dgraph check on k5', is_dgraph(k5))

    # print('is_complete_graph check 1', is_complete_graph(k5))
    # print('is_complete_graph check 2', is_complete_graph(disk))

# Python idiom that prevents main() from being called unless this file was called directly from command line.
# If this file is grabbed via import, main() will not execute automatically.
if __name__ == "__main__":
    # Import a library that will let use create a command line parser.
    import argparse
    parser = argparse.ArgumentParser(description='This program generates a random graph of size (size), and randomly toggles edges.')
    parser.add_argument('--size', type=int, default=8,
                        help='Number of nodes in the graph.')
    parser.add_argument('--max-steps', dest='max_steps', default=1_000,
                        help='Number of toggle steps to perform.')
    parser.add_argument('--print-steps', dest='print_steps', default=10_000,
                        help='Print progress to console after this many steps elapse.')
    main(parser.parse_args())