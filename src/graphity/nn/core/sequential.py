import functools
import types

import more_itertools
import torch
import torch.nn as nn
import torch.optim

# This kernel combines a list of modules into a sequence.
# Please ensure elements of list have compatible in/out sizes.
# This is necessary over raw nn.Sequential(...) since I 
# need to save/restore hidden states. 
class SequentialKernel(nn.Module):
    def __init__(self, neural_modules):
        super(SequentialKernel, self).__init__()
        self._neural_modules_list = list(more_itertools.always_iterable(neural_modules))
        self.input_dimensions = self._neural_modules_list[0].input_dimensions
        self.output_dimension = self._neural_modules_list[-1].output_dimension

        self.neural_modules = nn.Sequential(*self._neural_modules_list)

                
    def recurrent(self):
        for mod in self._neural_modules_list:
            if mod.recurrent(): return True
        else: return False

    def save_hidden(self):
        assert self.recurrent()
        return {id(self.mod): mod.save_hidden() for mod in self._neural_modules_list if mod.recurrent()}
            

    def restore_hidden(self, state_dict=None):
        for mod in self._neural_modules_list:
            if mod.recurrent():
                if state_dict != None and id(mod) in state_dict: mod.restore_hidden(state_dict[id(mod)])

    def forward(self, input):
        return self.neural_modules(input)