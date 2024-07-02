# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:22:17 2022

@author: JyGuo
"""

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as models
from Compact_Transformers.src import cct_7_3x1_32_sine_c100,cct_7_3x1_32_sine
RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

class PreTranNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.3):
        super(PreTranNet, self).__init__()
        
        
        
        self.dim_reconstruct = nn.Linear(172, 3072)
        self.CCT=cct_7_3x1_32_sine_c100(pretrained=True,progress=True,num_classes=840)
                                                       # this function is to calculate the cell state information based on previous ones
        self.activ=nn.RReLU()
        self.final=nn.Linear(840, 12)
    def forward(self, input, time, device):
        batch_size, feature_dim = input.size()
        
        """
        transmat = torch.zeros(batch_size,24, 12).to(device)
        
        """
        
       
        encode_input = input

        encode = self.dim_reconstruct(encode_input)

        output = self.CCT(encode.view(batch_size, 3,32,32))
        output = self.final(self.activ(output))
        output = torch.sigmoid(output)                                        # output size is batch_size by timestep ,  fk
 
        return output
