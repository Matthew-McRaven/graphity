import torch

import graphity.environment.toggle
import librl.task

# Single task object is not shared between episodes within an epoch.
# Otherwise, relay buffers will overwrite each other.
# May re-use tasks acros epochs, so long as you clear_replay() first.
# TODO: Optionally allocate replay buffers on first use.
class GraphTask(librl.task.ContinuousControlTask):
    def __init__(self, sampler=None, **kwargs):
        super(GraphTask, self).__init__(**kwargs)

        self.sampler = sampler

    def init_env(self):
        self.env.sampler = self.sampler



# Find the minimum energy state and return the (state, energy, index) tuple.
def min(graphtask):
    assert 0 and "This doesn't work."
    # Find the state with the smallest energy
    mindex = torch.argmin(graphtask.reward_buffer)
    # Compute the episode, time of the minimum energy
    episode, t = mindex // graphtask.reward_buffer.shape[-1], mindex % graphtask.reward_buffer.shape[-1],

    graph = graphtask.state_buffer[episode][t]
    for (i,j) in graphtask.action_buffer.actions[episode][t]:
        graphity.environment.toggle.toggle_edge(int(i), int(j), graph, False)
    
    return graph, graphtask.reward_buffer[episode][t], t