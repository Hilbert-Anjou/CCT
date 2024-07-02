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

class TransNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.3):
        super(TransNet, self).__init__()
        self.CCT=cct_7_3x1_32_sine_c100(pretrained=True, progress=True,num_classes=1000)
        #self.CCT=cct_7_3x1_32_sine(pretrained=True, progress=True)
        
        self.first_conv  = nn.Sequential(                       # apply ResNet to replace CNN backbone
            #torch.nn.Dropout2d(p=0.3), 
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        #self.Encoder = models.resnet18(pretrained=False)
        self.dim_reduce = nn.Linear(825, 1024)
        
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
       
        self.nn_output = nn.Sequential(nn.Linear(24000, 1),nn.ReLU())
        
        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

        
    
    def forward(self, input, time, device):
        batch_size, time_step, feature_dim = input.size()
        
        transmat = torch.zeros(batch_size,24, 1000).to(device)

        for t in range(time_step):   # to process the input data based on each timestep (variable t)
            encode_input = input[:,t,:].reshape(input[:,t,:].shape[0],1,33,25)

            encode_input = self.first_conv(encode_input)                      # the next 3 lines are for resnet
            
            #encode = self.Encoder(encode_input)
            encode = self.dim_reduce(encode_input.view(batch_size*3,-1))
            trans = self.CCT(encode.view(batch_size, 3,32,32))
            
            transmat= torch.cat((transmat[:,1:,:], trans.view(batch_size,1,1000)), 1)
            
        """

        rnn_output = local_h
        
        origin_single_h = out[..., :self.hidden_dim]
        
        if self.dropres > 0.0:
            origin_single_h = self.nn_dropres(origin_single_h)
        rnn_output = rnn_output + origin_single_h
        
        #rnn_output = torch.cat([rnn_output,origin_single_h],dim=1)

        rnn_output = rnn_output.contiguous().view(-1, local_h.size(-1))    # the size now is batch_size*timestep by hidden_dim
        if self.dropout > 0.0:
            rnn_output = self.nn_dropout(rnn_output)
        """
        output = self.nn_output(transmat.view(batch_size,-1))
        output = output.contiguous().view(batch_size) # reshape
        output = torch.sigmoid(output)                                        # output size is batch_size by timestep ,  fk
                                                                              # already done the sigmoid function here
                                                                              # this output is for mortality prediction
        
        """
        output_discharge = self.nn_output_discharge(rnn_output)               # only copy the last fc layer
        output_discharge = output_discharge.contiguous().view(batch_size) # reshape
        output_discharge = torch.sigmoid(output_discharge)
        """
        """
        local_dis_two = tmp_dis.permute(1, 0)                                 # copy the conv and fc layer
        local_dis_two = torch.cumsum(local_dis_two, dim=1)
        local_dis_two = torch.softmax(local_dis_two, dim=1)
        local_h_two = tmp_h.permute(1, 2, 0)
        local_h_two = local_h_two * local_dis_two.unsqueeze(1)
        
        #Re-calibrate Progression patterns
        local_theme_two = torch.mean(local_h_two, dim=-1)
        local_theme_two = self.nn_scale_discharge(local_theme_two)
        local_theme_two = torch.relu(local_theme_two)
        local_theme_two = self.nn_rescale_discharge(local_theme_two)
        
        local_theme_two = torch.sigmoid(local_theme_two)
        # before conv, the size is batch_size by hidden_dim by conv_size
        local_h_two = self.nn_conv_discharge(local_h_two).squeeze(-1)
        local_h_two = local_theme_two * local_h_two
        h_two.append(local_h_two)
        
        rnn_output_two = local_h_two
        
        origin_single_h_two = out[..., :self.hidden_dim]
        
        if self.dropres > 0.0:
            origin_single_h_two = self.nn_dropres(origin_single_h_two)
        rnn_output_two = rnn_output_two + origin_single_h_two
        rnn_output_two = rnn_output_two.contiguous().view(-1, rnn_output_two.size(-1))
        if self.dropout > 0.0:
            rnn_output_two = self.nn_dropout(rnn_output_two)
            
        output_discharge = self.nn_output_discharge(rnn_output_two)
        output_discharge = output_discharge.contiguous().view(batch_size) # reshape
        output_discharge = torch.sigmoid(output_discharge)
        """
        
        #return output, output_discharge, torch.stack(distance)
        #return output, torch.stack(distance)
        return output
