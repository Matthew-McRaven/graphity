import torch

#
def get_default_hyperparams():
    ret = {}
    ret['allow_cuda'] = True
    ret['epochs'] = 10
    ret['episode_count'] = 1
    ret['episode_length'] = 10000
    ret['graph_size'] = 10
    return ret

def to_cuda(tensor, allow_cuda):
    if allow_cuda and torch.cuda.is_available():
        return tensor.to(device="cuda")
    else:
        return tensor