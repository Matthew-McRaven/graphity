import functools
import types

import more_itertools
import torch
import torch.nn as nn
import torch.optim

# Forward takes in two inputs, and passes them through two internal neural modules.
# It combine the results via a single bilinear layer.
# Can extract realtionships 
class BilinearKernel(nn.Module):
    def __init__(self, left_module, right_module, output_size, nlo = None):
        super(BilinearKernel, self).__init__()

        self.nlo = nlo if nlo != None else lambda x: x

        self._left_module = left_module
        self._right_module = right_module


        self.output_dimension = (output_size, )

        self._left_output_size = functools.reduce(lambda x, y: x*y, left_module.output_dimension, 1)
        self._right_output_size = functools.reduce(lambda x, y: x*y, right_module.output_dimension, 1)
        self.bilinear_layer = nn.Bilinear(self._left_output_size, self._right_output_size, output_size)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

                
    def recurrent(self):
        return self._left_module.recurrent() or self._right_module.recurrent()

    def save_hidden(self):
        ret = {}
        assert self.recurrent()
        if self._left_module.recurrent(): ret[id(self._left_module)] = self._left_module.save_hidden()
        # If the both networks are the same, don't save it twice.
        if self._left_module == self._right_module: pass
        elif self._right_module.recurrent(): ret[id(self._right_module)] = self._right_module.save_hidden()
        return ret
            
    def restore_hidden(self, state_dict=None):
        if self._left_module.recurrent():
            id_left = id(self._left_module)
            if state_dict != None and id_left in state_dict: self._left_module.restore_hidden(state_dict[id_left])
            else: self._left_module.restore_hidden()
        # If the both networks are the same, don't restore twice
        if self._left_module == self._right_module: pass
        elif self._right_module.recurrent():
            id_right = id(self._right_module)
            if state_dict != None and id_right in state_dict: self._right_module.restore_hidden(state_dict[id_right])
            else: self._right_module.restore_hidden()

    def forward(self, linput, rinput):
        loutput = self._left_module(linput).view(-1, self._left_output_size)
        routput = self._right_module(rinput).view(-1, self._right_output_size)

        output = self.bilinear_layer(loutput, routput)
        output = self.nlo(output)
        assert not torch.isnan(output).any() # type: ignore

        return output

# Share one network between two inputs.
# The supplied module is responsible for handling various data sizes.
# Inspired by Koch et al. 2015 `Siamese Neural Networks for One-shot Image Recognition`
class SiameseKernel(BilinearKernel):
    def __init__(self, module, output_size, nlo = torch.nn.Sigmoid()):
        super(SiameseKernel, self).__init__(module, module, output_size, nlo=nlo)

# Feed a single input through two different neural modules.
# Join the result using bilinear layer
class JoinKernel(BilinearKernel):
    def __init__(self, left_module, right_module, output_size, nlo = None):
        super(JoinKernel, self).__init__(left_module, right_module, output_size, nlo=nlo)
    def forward(self, input):
        return BilinearKernel.forward(self, input, input)