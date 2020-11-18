import torch

# Get hyperparameters about graph search:
def get_default_hyperparams():
    ret = {}
    ret['allow_cuda'] = True
    ret['epochs'] = 10
    ret['episode_count'] = 1
    ret['episode_length'] = 10000
    ret['graph_size'] = 10
    ret['toggles_per_step'] = 2
    # Neural net configurations
    ret['critic_steps'] = 10
    ret['dropout'] = .1
    ret['l2'] = .1
    ret['alpha'] = .0003
    # RL configurations
    ret['gamma'] = .95
    ret['lambda'] = .99
    ret['epsilon'] = .5
    ret['c_1'] = 1
    return ret


def to_cuda(tensor, allow_cuda):
    if allow_cuda and torch.cuda.is_available():
        return tensor.to(device="cuda")
    else:
        return tensor