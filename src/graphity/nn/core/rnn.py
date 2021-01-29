import functools
import types

import more_itertools
import torch
import torch.nn as nn
import torch.optim

class RecurrentKernel(nn.Module):
    def __init__(self, input_dimensions, hidden_size, num_layers, recurrent_unit="LSTM", bidirectional=False, dropout=None):
        super(RecurrentKernel, self).__init__()
        dropout = dropout if dropout else self.get_default_hyperparameters().dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_dimensions = list(more_itertools.always_iterable(input_dimensions))
        self._input_size = functools.reduce(lambda x, y: x*y, self.input_dimensions, 1)
        self.output_dimension = (hidden_size, )

        
        if recurrent_unit.upper() == "LSTM": rec_init = nn.LSTM
        elif recurrent_unit.upper() == "GRU": rec_init = nn.GRU
        elif recurrent_unit.upper() == "RNN": rec_init = nn.RNN
        else:rec_init = lambda *x: (_ for _ in ()).throw(NotImplementedError("Choose an implemented recurrent unit"))
        
        self.recurrent_layer = rec_init(self._input_size, hidden_size, num_layers = num_layers, bidirectional=bidirectional, batch_first=False)
        self.init_hidden()
        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def init_hidden(self):

        self.hidden_state = torch.zeros(self.num_layers, 1, self.hidden_size) # type: ignore
        self.cell_state = torch.zeros(self.num_layers, 1, self.hidden_size) # type: ignore
        nn.init.kaiming_normal_(self.hidden_state)
        nn.init.kaiming_normal_(self.cell_state)

    def recurrent(self):
        return True

    def save_hidden(self):
        return self.hidden_state

    def restore_hidden(self, state=None):
        if state == None:
            self.init_hidden()
        else:
            assert 0
            hidden_state, cell_state = state
            # Assert that state be correct shape for calling forward().
            # Assert here rather than forward(), since it may not be obvious why forward fails
            # when cell state is of wrong shape.
            hidden_state = hidden_state.view(self.num_layers, -1, self.hidden_size)
            cell_state = cell_state.view(self.num_layers, -1, self.hidden_size)

            self.hidden_state, self.cell_state = hidden_state, cell_state

    @staticmethod
    def get_default_hyperparameters():
        ret = {}
        ret['dropout'] = .1
        ret['layer_list'] = [200, 100]
        ret['recurrent_unit'] = 'RNN'

        return types.SimpleNamespace(**ret)

    def forward(self, input):
        # Treat the entire input as a single seqence of data.
        d0, d1 = 1, -1
        # Assume data is not batched
        if len(input.shape) == 1: assert self._input_size == input.shape[-1]
        elif len(input.shape) == 2: 
            assert self._input_size == input.shape[0] * input.shape[0]
            d0 = input.shape[0]
        elif len(input.shape) == 3: 
            raise NotImplementedError("No clue how to do this.")
        # TODO: Figure out how to batch hidden states
        input = input.view(d0, d1, self._input_size)
        # Push observations through feed forward layers.
        if self.hidden_state.shape[1] == 1 and input.shape[1] != 1:
            hidden_state = self.hidden_state.repeat((1,input.shape[1],1))
        else: hidden_state = self.hidden_state

        if self.cell_state.shape[1] == 1 and input.shape[1] != 1:
            cell_state = self.cell_state.repeat((1,input.shape[1],1))
        else: cell_state = self.cell_state

        h0 = hidden_state.to(input.device), cell_state.to(input.device)
        output, h1 = self.recurrent_layer(input.float(), h0)

        self.hidden_state, self.cell_state = h1
        # We really dont care about the progress / history of our state data.
        self.hidden_state, self.cell_state = self.hidden_state.detach(), self.cell_state.detach()
        assert not torch.isnan(output).any() # type: ignore

        return output