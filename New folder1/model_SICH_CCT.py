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

class StageTranNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.3):
        super(StageTranNet, self).__init__()
        
        
        self.first_conv  = nn.Sequential(                       # apply ResNet to replace CNN backbone
            #torch.nn.Dropout2d(p=0.3), 
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        self.Encoder = models.resnet18(pretrained=False)
        self.dim_reconstruct = nn.Linear(825, 3072)
        
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        self.output_dim = output_dim
        self.levels = levels
        self.chunk_size = hidden_dim // levels
        
        self.kernel = nn.Linear(int(input_dim+1), int(hidden_dim*4+levels*2))           # kernel and RNN input dimension input_dim + 1
        nn.init.xavier_uniform_(self.kernel.weight)                                     # since the input was input_data concatenate ones (matrix)
        nn.init.zeros_(self.kernel.bias)                                                # concatenation is done in the model
        self.recurrent_kernel = nn.Linear(int(hidden_dim+1), int(hidden_dim*4+levels*2))# input is only input_data and the all one matrix
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)
        
        self.nn_scale = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), 1)
        self.nn_output = nn.Linear(int(self.conv_dim), int(output_dim))
        
        self.nn_avg = nn.Linear(24, 1)
        
        
        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)
        #self.CCT=cct_7_3x1_32_sine_c100(pretrained=True, progress=True,num_classes=840)
        self.CCT=torch.load('model/CCT_only_0.80auc.pt')
        self.out1=nn.Sequential(nn.Linear(20160, 4096),torch.nn.RReLU())
        self.out2=nn.Sequential(nn.Linear(4096,1024),torch.nn.RReLU())
        self.out3=nn.Linear(1024,1)
    def cumax(self, x, mode='l2r'):
        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x
                                                                              # this function is to calculate the cell state information based on previous ones
    
    def forward(self, input, time, device):
        batch_size, time_step, feature_dim = input.size()
        
        
        transmat = torch.zeros(batch_size,24, 840).to(device)
        
        
        
       
        
        for t in range(time_step):   # to process the input data based on each timestep (variable t)
            encode_input = input[:,t,:]
            #encode_input = self.first_conv(encode_input.reshape(input[:,t,:].shape[0],1,33,25))                      
            
            #encode_input = self.Encoder(encode_input)

            encode = self.dim_reconstruct(encode_input)
            trans = self.CCT(encode.view(batch_size, 3,32,32))
            
            transmat= torch.cat((transmat[:,1:,:], trans.view(batch_size,1,840)), 1)
          

        #Re-weighted convolution operation
        
        local_h = transmat.permute(0, 2, 1)                                  # after permutation, the size is batch_size by hidden_dim by conv_size
        
        local_theme = self.nn_avg(local_h).view(batch_size,-1)                        # to weight the past health state info based on the relavant time gap
        
        #Re-calibrate Progression patterns
        #local_theme = torch.mean(transmat, dim=1)                         # z_t in the paper, the average of past health state
        local_theme = self.nn_scale(local_theme)                          # input: hidden_dim, output: hidden_dim //6
        #local_theme = F.prelu(local_theme,weight=prelu)
        local_theme = torch.relu(local_theme) 
        local_theme = self.nn_rescale(local_theme)                        # input: hidden_dim//6, output: hidden_dim
        
        local_theme = torch.sigmoid(local_theme)                          # the size of local_h, this is the attention part
        # before conv, the size is batch_size by hidden_dim by conv_size
        local_h = self.nn_conv(local_h).squeeze(-1)                       # after conv, the size is batch_size by hidden_dim
        local_h = local_theme * local_h                                   # weight based on 
          

        rnn_output = local_h
        
        origin_single_h = transmat[:,-1, :]
        
        if self.dropres > 0.0:
            origin_single_h = self.nn_dropres(origin_single_h)
        rnn_output = rnn_output + origin_single_h
        
        #rnn_output = torch.cat([rnn_output,origin_single_h],dim=1)

        rnn_output = rnn_output.contiguous().view(-1, local_h.size(-1))    # the size now is batch_size*timestep by hidden_dim
        if self.dropout > 0.0:
            rnn_output = self.nn_dropout(rnn_output)
        
        output = self.nn_output(rnn_output)
        output = output.contiguous().view(batch_size) # reshape
    
        output = torch.sigmoid(output)                                        # output size is batch_size by timestep ,  fk
                                                                              # already done the sigmoid function here
                                                                              # this output is for mortality prediction
        
       

        #return output, output_discharge, torch.stack(distance)
        #return output, torch.stack(distance)
        return output
