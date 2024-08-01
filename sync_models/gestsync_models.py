import torch
import torch.nn as nn
from torch.autograd import Variable

from sync_models.modules import *

import math


class Transformer_RGB(nn.Module):

    def __init__(self):
        super().__init__()

        self.net_vid = self.build_net_vid()
        self.ff_vid = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1024)
            )

        self.pos_encoder = PositionalEncoding_RGB(d_model=512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.net_aud = self.build_net_aud()
        self.lstm = nn.LSTM(512, 256, num_layers=1, bidirectional=True, batch_first=True)

        self.ff_aud = NetFC_2D(input_dim=512, hidden_dim=512, embed_dim=1024)


        self.logits_scale = nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.logits_scale.weight)

        self.fc = nn.Linear(1,1)

    def build_net_vid(self):
        layers = [
            {
                'type': 'conv3d',
                'n_channels': 64,
                'kernel_size': (5, 7, 7),
                'stride': (1, 3, 3),
                'padding': (0),
                'maxpool': {
                    'kernel_size': (1, 3, 3),
                    'stride': (1, 2, 2)
                }
            },
            {
                'type': 'conv3d',
                'n_channels': 128,
                'kernel_size': (1, 5, 5),
                'stride': (1, 2, 2),
                'padding': (0, 0, 0),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 2, 2),
                'padding': (0, 1, 1),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 1, 2),
                'padding': (0, 1, 1),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 1, 1),
                'padding': (0, 1, 1),
                'maxpool': {
                    'kernel_size': (1, 3, 3),
                    'stride': (1, 2, 2)
                }
            },
            {
                'type': 'fc3d',
                'n_channels': 512,
                'kernel_size': (1, 4, 4),
                'stride': (1, 1, 1),
                'padding': (0),
            },
        ]
        return VGGNet(n_channels_in=3, layers=layers)

    def build_net_aud(self):
        layers = [
            {
                'type': 'conv2d',
                'n_channels': 64,
                'kernel_size': (3, 3),
                'stride': (2, 2),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (3, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'conv2d',
                'n_channels': 192,
                'kernel_size': (3, 3),
                'stride': (1, 2),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (3, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'conv2d',
                'n_channels': 384,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
            },
            {
                'type': 'conv2d',
                'n_channels': 256,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
            },
            {
                'type': 'conv2d',
                'n_channels': 256,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (2, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'fc2d',
                'n_channels': 512,
                'kernel_size': (4, 2),
                'stride': (1, 1),
                'padding': (0, 0),
            },
        ]
        return VGGNet(n_channels_in=1, layers=layers)

    def forward_vid(self, x, return_feats=False):
        out_conv6 = self.net_vid(x).squeeze(-1).squeeze(-1)
        # print("Conv: ", out_conv6.shape)        # Bx1024x21x1x1

        out = self.pos_encoder(out_conv6.transpose(1,2))
        out_trans = self.transformer_encoder(out)
        # print("Transformer: ", out_trans.shape)     # Bx21x1024

        out = self.ff_vid(out_trans).transpose(1,2)
        # print("MLP output: ", out.shape)                            # Bx1024

        if return_feats:
            return out, out_conv6
        else:
            return out

    def forward_aud(self, x):
        out = self.net_aud(x)
        out = self.ff_aud(out)
        out = out.squeeze(-1)
        return out

# class Transformer_RGB(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.net_lip = self.build_net_vid()
#         # self.ff_lip = NetFC(input_dim=512, hidden_dim=512, embed_dim=1024)
#         self.ff_lip = nn.Sequential(
#                 nn.Linear(512, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 1024)
#             )

#         self.pos_encoder = PositionalEncoding_RGB(d_model=512)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

#         self.net_aud = self.build_net_aud()
#         self.lstm = nn.LSTM(512, 256, num_layers=1, bidirectional=True, batch_first=True)

#         self.ff_aud = NetFC_2D(input_dim=512, hidden_dim=512, embed_dim=1024)

#         _, _, _, self.start_offset = calc_receptive_field(self.net_lip.layers,
#                                                           imsize=400)

#         self.logits_scale = nn.Linear(1, 1, bias=False)
#         torch.nn.init.ones_(self.logits_scale.weight)

#         self.fc = nn.Linear(1,1)

#     def build_net_vid(self):
#         layers = [
#             {
#                 'type': 'conv3d',
#                 'n_channels': 64,
#                 'kernel_size': (5, 7, 7),
#                 'stride': (1, 3, 3),
#                 'padding': (0),
#                 'maxpool': {
#                     'kernel_size': (1, 3, 3),
#                     'stride': (1, 2, 2)
#                 }
#             },
#             {
#                 'type': 'conv3d',
#                 'n_channels': 128,
#                 'kernel_size': (1, 5, 5),
#                 'stride': (1, 2, 2),
#                 'padding': (0, 0, 0),
#             },
#             {
#                 'type': 'conv3d',
#                 'n_channels': 256,
#                 'kernel_size': (1, 3, 3),
#                 'stride': (1, 2, 2),
#                 'padding': (0, 1, 1),
#             },
#             {
#                 'type': 'conv3d',
#                 'n_channels': 256,
#                 'kernel_size': (1, 3, 3),
#                 'stride': (1, 1, 2),
#                 'padding': (0, 1, 1),
#             },
#             {
#                 'type': 'conv3d',
#                 'n_channels': 256,
#                 'kernel_size': (1, 3, 3),
#                 'stride': (1, 1, 1),
#                 'padding': (0, 1, 1),
#                 'maxpool': {
#                     'kernel_size': (1, 3, 3),
#                     'stride': (1, 2, 2)
#                 }
#             },
#             {
#                 'type': 'fc3d',
#                 'n_channels': 512,
#                 'kernel_size': (1, 4, 4),
#                 'stride': (1, 1, 1),
#                 'padding': (0),
#             },
#         ]
#         return VGGNet(n_channels_in=3, layers=layers)

#     def build_net_aud(self):
#         layers = [
#             {
#                 'type': 'conv2d',
#                 'n_channels': 64,
#                 'kernel_size': (3, 3),
#                 'stride': (2, 2),
#                 'padding': (1, 1),
#                 'maxpool': {
#                     'kernel_size': (3, 3),
#                     'stride': (2, 2)
#                 }
#             },
#             {
#                 'type': 'conv2d',
#                 'n_channels': 192,
#                 'kernel_size': (3, 3),
#                 'stride': (1, 2),
#                 'padding': (1, 1),
#                 'maxpool': {
#                     'kernel_size': (3, 3),
#                     'stride': (2, 2)
#                 }
#             },
#             {
#                 'type': 'conv2d',
#                 'n_channels': 384,
#                 'kernel_size': (3, 3),
#                 'stride': (1, 1),
#                 'padding': (1, 1),
#             },
#             {
#                 'type': 'conv2d',
#                 'n_channels': 256,
#                 'kernel_size': (3, 3),
#                 'stride': (1, 1),
#                 'padding': (1, 1),
#             },
#             {
#                 'type': 'conv2d',
#                 'n_channels': 256,
#                 'kernel_size': (3, 3),
#                 'stride': (1, 1),
#                 'padding': (1, 1),
#                 'maxpool': {
#                     'kernel_size': (2, 3),
#                     'stride': (2, 2)
#                 }
#             },
#             {
#                 'type': 'fc2d',
#                 'n_channels': 512,
#                 'kernel_size': (4, 2),
#                 'stride': (1, 1),
#                 'padding': (0, 0),
#             },
#         ]
#         return VGGNet(n_channels_in=1, layers=layers)

#     def forward_vid(self, x, return_feats=False):
#         out_conv6 = self.net_lip(x).squeeze(-1).squeeze(-1)
#         # print("Conv: ", out_conv6.shape)        # Bx512x20x1x1

#         out = self.pos_encoder(out_conv6.transpose(1,2))
#         out_trans = self.transformer_encoder(out)
#         # print("Transformer: ", out_trans.shape)     # Bx20x512

#         out = self.ff_lip(out_trans).transpose(1,2)
#         # print("MLP output: ", out.shape)                            # Bx512

#         if return_feats:
#             return out, out_conv6
#         else:
#             return out

#     def forward_aud(self, x):
#         out = self.net_aud(x)
#         out = self.ff_aud(out)
#         out = out.squeeze(-1)
#         return out

