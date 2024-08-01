import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class PositionalEncoding_RGB(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding_RGB, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

def calc_receptive_field(layers, imsize, layer_names=None):
    if layer_names is not None:
        print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]

    for l_id, layer in enumerate(layers):
        conv = [
            layer[key][-1] if type(layer[key]) in [list, tuple] else layer[key]
            for key in ['kernel_size', 'stride', 'padding']
        ]
        currentLayer = outFromIn(conv, currentLayer)
        if 'maxpool' in layer:
            conv = [
                (layer['maxpool'][key][-1] if type(layer['maxpool'][key])
                 in [list, tuple] else layer['maxpool'][key]) if
                (not key == 'padding' or 'padding' in layer['maxpool']) else 0
                for key in ['kernel_size', 'stride', 'padding']
            ]
            currentLayer = outFromIn(conv, currentLayer, ceil_mode=False)
    return currentLayer

def outFromIn(conv, layerIn, ceil_mode=True):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out
    
class DebugModule(nn.Module):
    """
    Wrapper class for printing the activation dimensions 
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.debug_log = True

    def debug_line(self, layer_str, output, memuse=1, final_call=False):
        if self.debug_log:
            namestr = '{}: '.format(self.name) if self.name is not None else ''
            # print('{}{:80s}: dims {}'.format(namestr, repr(layer_str),
            #                                  output.shape))

            if final_call:
                self.debug_log = False
                # print()

class VGGNet(DebugModule):

    conv_dict = {
        'conv1d': nn.Conv1d,
        'conv2d': nn.Conv2d,
        'conv3d': nn.Conv3d,
        'fc1d': nn.Conv1d,
        'fc2d': nn.Conv2d,
        'fc3d': nn.Conv3d,
    }

    pool_dict = {
        'conv1d': nn.MaxPool1d,
        'conv2d': nn.MaxPool2d,
        'conv3d': nn.MaxPool3d,
    }

    norm_dict = {
        'conv1d': nn.BatchNorm1d,
        'conv2d': nn.BatchNorm2d,
        'conv3d': nn.BatchNorm3d,
        'fc1d': nn.BatchNorm1d,
        'fc2d': nn.BatchNorm2d,
        'fc3d': nn.BatchNorm3d,
    }

    def __init__(self, n_channels_in, layers):
        super(VGGNet, self).__init__()

        self.layers = layers

        n_channels_prev = n_channels_in
        for l_id, lr in enumerate(self.layers):
            l_id += 1
            name = 'fc' if 'fc' in lr['type'] else 'conv'
            conv_type = self.conv_dict[lr['type']]
            norm_type = self.norm_dict[lr['type']]
            self.__setattr__(
                '{:s}{:d}'.format(name, l_id),
                conv_type(n_channels_prev,
                          lr['n_channels'],
                          kernel_size=lr['kernel_size'],
                          stride=lr['stride'],
                          padding=lr['padding']))
            n_channels_prev = lr['n_channels']
            self.__setattr__('bn{:d}'.format(l_id), norm_type(lr['n_channels']))
            if 'maxpool' in lr:
                pool_type = self.pool_dict[lr['type']]
                padding = lr['maxpool']['padding'] if 'padding' in lr[
                    'maxpool'] else 0
                self.__setattr__(
                    'mp{:d}'.format(l_id),
                    pool_type(kernel_size=lr['maxpool']['kernel_size'],
                              stride=lr['maxpool']['stride'],
                              padding=padding),
                )

    def forward(self, inp):
        self.debug_line('Input', inp)
        out = inp
        for l_id, lr in enumerate(self.layers):
            l_id += 1
            name = 'fc' if 'fc' in lr['type'] else 'conv'
            out = self.__getattr__('{:s}{:d}'.format(name, l_id))(out)
            out = self.__getattr__('bn{:d}'.format(l_id))(out)
            out = nn.ReLU(inplace=True)(out)
            self.debug_line(self.__getattr__('{:s}{:d}'.format(name, l_id)),
                            out)
            if 'maxpool' in lr:
                out = self.__getattr__('mp{:d}'.format(l_id))(out)
                self.debug_line(self.__getattr__('mp{:d}'.format(l_id)), out)

        self.debug_line('Output', out, final_call=True)

        return out



class NetFC(DebugModule):

    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(NetFC, self).__init__()
        self.fc7 = nn.Conv3d(input_dim, hidden_dim, kernel_size=(1, 1, 1))
        self.bn7 = nn.BatchNorm3d(hidden_dim)
        self.fc8 = nn.Conv3d(hidden_dim, embed_dim, kernel_size=(1, 1, 1))

    def forward(self, inp):
        out = self.fc7(inp)
        self.debug_line(self.fc7, out)
        out = self.bn7(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.fc8(out)
        self.debug_line(self.fc8, out, final_call=True)
        return out

class NetFC_2D(DebugModule):

    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(NetFC_2D, self).__init__()
        self.fc7 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.bn7 = nn.BatchNorm2d(hidden_dim)
        self.fc8 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=(1, 1))

    def forward(self, inp):
        out = self.fc7(inp)
        self.debug_line(self.fc7, out)
        out = self.bn7(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.fc8(out)
        self.debug_line(self.fc8, out, final_call=True)
        return out