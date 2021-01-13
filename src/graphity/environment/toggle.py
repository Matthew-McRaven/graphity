
def toggle_edge(i,j, next_state, allow_self_loop=False, enforce_symmetry=True):
    # If our action falls on the diagonal, only allow change if we allow self loops.
    if i == j and allow_self_loop:
        next_state[i][j] = next_state[i][j] ^ 1
    # Otherwise, toggle undirected edge between both nodes.
    else: 
        if enforce_symmetry: 
            next_state[i, j] ^= 1
            next_state[j, i] ^= 1
        else: next_state[i, j] = next_state[i, j]^1
    return next_state