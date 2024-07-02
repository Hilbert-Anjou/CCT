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
from model import Transformer
from tab_transformer_pytorch import TabTransformer,FTTransformer
RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

class StageTranNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.3):
        super(StageTranNet, self).__init__()
        
        
        
        
        #assert hidden_dim % levels == 0
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
        self.extraction=nn.Linear(825,256)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        """
        self.vanilla=TabTransformer(categories = (24,),      # tuple containing the number of unique values within each category
                                    num_continuous = 512,                # number of continuous values
                                    dim = 128,                           # dimension, paper set at 32
                                    dim_out = 512,                        # binary prediction, but could be anything
                                    depth = 12,                          # depth, paper recommended 6
                                    heads = 16,                          # heads, paper recommends 8
                                    attn_dropout = 0,                 # post-attention dropout
                                    ff_dropout = 0,                   # feed forward dropout
                                    mlp_hidden_mults = (8, 4),          # relative multiples of each hidden dimension of the last mlp to logits
                                    mlp_act = nn.ReLU()).to(self.device)
        """
        self.FT=FTTransformer(categories = (24,),      # tuple containing the number of unique values within each category
                                    num_continuous = 256,                # number of continuous values
                                    dim = 32,                           # dimension, paper set at 32
                                    dim_out = 100,                        # binary prediction, but could be anything
                                    depth = 6,                          # depth, paper recommended 6
                                    heads = 8,                          # heads, paper recommends 8
                                    attn_dropout = 0.1,                 # post-attention dropout
                                    ff_dropout = 0.1                    # feed forward dropout
                                )
        
        #self.vanilla=Transformer(n_src_vocab=1, n_trg_vocab=100, src_pad_idx=0, trg_pad_idx=0).to(self.device)
        
        self.CCT=cct_7_3x1_32_sine_c100(pretrained=True, progress=True,num_classes=840)
        
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
        
        
        transmat = torch.zeros(batch_size,24, 100).to(device)
        
        
        
       
        
        for t in range(time_step):   
            #encode_input = input[:,t,:].reshape(input[:,t,:].shape[0],1,33,25)
            trans=self.extraction(input[:,t,:])
            #trans = self.CCT(encode.view(batch_size, 3,32,32))
            catgory=(torch.ones((batch_size,1),dtype=torch.int32)*t).to(device)
            trans=self.FT(catgory,trans)
            transmat= torch.cat((transmat[:,1:,:], trans.view(batch_size,1,100)), 1)
            
            
        #Re-weighted convolution operation
        
        local_h = transmat.permute(0, 2, 1)                                  # after permutation, the size is batch_size by hidden_dim by conv_size
        
        #local_theme = self.nn_avg(local_h).view(batch_size,-1)                        # to weight the past health state info based on the relavant time gap
        local_theme = self.nn_avg(local_h).view(batch_size,-1)
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

        rnn_output = rnn_output + origin_single_h
        
        #rnn_output = torch.cat([rnn_output,origin_single_h],dim=1)

        rnn_output = rnn_output.contiguous().view(-1, local_h.size(-1))    # the size now is batch_size*timestep by hidden_dim
        output = self.nn_output(rnn_output)
        output = output.contiguous().view(batch_size) # reshape
        output = torch.sigmoid(output)                                        # output size is batch_size by timestep ,  fk
                                                                              # already done the sigmoid function here
                                                                              # this output is for mortality prediction

        return output
