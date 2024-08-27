# width-wise early exit approach based on ResNet https://arxiv.org/pdf/2405.03222

import torch
import torch.nn as nn


class blPollin(nn.Module):

    def __init__(self):
        super(blPollin, self).__init__()
