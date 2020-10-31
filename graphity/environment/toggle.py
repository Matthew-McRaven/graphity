
def toggle_edge(i,j, next_state, allow_self_loop=False):
    # If our action falls on the diagonal, only allow change if we allow self loops.
    if i == j:
        if allow_self_loop:
            next_state[i][j] = next_state[i][j] ^ 1
    # Otherwise, toggle undirected edge between both nodes.
    else:
        next_state[i][j] = next_state[i][j] ^ 1
        next_state[j][i] = next_state[i][j]