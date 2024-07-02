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
import numpy as np
import torchvision.models as models
from Compact_Transformers.src import cct_7_3x1_32_sine_c100,cct_7_3x1_32_sine,cct_14_7x2_224,cct_14_7x2_384
RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

class MixTranNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.):
        super(MixTranNet, self).__init__()
        self.first_conv  = nn.Sequential(                       # apply ResNet to replace CNN backbone
            #torch.nn.Dropout2d(p=0.3), 
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )

        self.Encoder = models.resnet18(pretrained=False)
        #self.dim_reconstruct = nn.Linear(1000, 18432)
        self.dim_reconstruct = nn.Linear(1000, 6272)
        #self.dim_reconstruct = Linear_BBB(1000, 6272)
        
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim

        hidden_dim=300
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
        
        #self.nn_avg = Linear_BBB(24, 1)
        self.nn_avg = nn.Linear(24, 1)
        
        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)
        #self.CCT=cct_7_3x1_32_sine_c100(pretrained=True, progress=True,num_classes=4800)

        self.CCT=cct_14_7x2_224(pretrained=True, progress=True,num_classes=7200)
        #self.CCT=cct_14_7x2_384(pretrained=True, progress=True,num_classes=4800)
        #self.CCT=torch.load('model/CCT_only_0.80auc.pt')
    def mixdown(self, positive_map, negative_map):
        num_positives = positive_map.size(0)
        num_negatives = negative_map.size(0)
        selected_indices = torch.randint(num_negatives, size=(num_positives,))
        negative_map_selected = negative_map[selected_indices, :, :]

        # Generate mixup lambdas for each sample
        mixup_lambdas = torch.tensor(np.random.beta(5, 3, size=(num_positives, 1, 1)), device=positive_map.device)

        # Perform mixup
        mixed_map = mixup_lambdas * positive_map + (1 - mixup_lambdas) * negative_map_selected

        return mixed_map.float(), mixup_lambdas.squeeze()
    def mixup(self, positive_map, negative_map):
        num_positives = positive_map.size(0)
        num_negatives = negative_map.size(0)
        repeat_factor = num_negatives // num_positives
        remaining_samples = num_negatives % num_positives

        positive_map_repeated = positive_map.repeat(repeat_factor, 1, 1)
        positive_map_remaining = positive_map[:remaining_samples, :, :]
        positive_map_balanced = torch.cat([positive_map_repeated, positive_map_remaining], dim=0)

        # Shuffle positive samples
        shuffled_indices = torch.randperm(num_negatives)
        positive_map_balanced = positive_map_balanced[shuffled_indices, :, :]

        # Generate mixup lambdas for each sample
        mixup_lambdas = torch.tensor(np.random.beta(5, 3, size=(num_negatives, 1, 1)), device=positive_map.device)

        # Perform mixup
        mixed_map = mixup_lambdas * positive_map_balanced + (1 - mixup_lambdas) * negative_map

        return mixed_map.float(), mixup_lambdas.squeeze()
    def stage(self,transmat,batch_size):
        #Re-weighted convolution operation
        #transmat=nn.relu(transmat)
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
        
        #output= self.nn_avg(trans)

        output = torch.sigmoid(output)                                        # output size is batch_size by timestep ,  fk
        return output

    def forward(self, input, label, device):
        batch_size, time_step, feature_dim = input.size()

        transmat = torch.zeros(batch_size, 24, 6272).to(device)
        for t in range(24):
            encode_input = input[:, t, :].reshape(input[:, t, :].shape[0], 1, 33, 25)
            encode_input = self.first_conv(encode_input)
            encode = self.Encoder(encode_input)

            encode = self.dim_reconstruct(encode)
            transmat = torch.cat((transmat[:, 1:, :], encode.view(batch_size, 1, 6272)), 1)
        
        if self.training:
            assert label is not None, "Labels must be provided during training."
            positive_mask = (label == 1)
            negative_mask = (label == 0)
            if positive_mask.any():
                positive_map = transmat[positive_mask, :, :]
                negative_map = transmat[negative_mask, :, :]
                transmat= self.CCT(transmat.view(batch_size,3,224,224)).view(batch_size,24,300)
                output= self.stage(transmat,batch_size)
                mixed_map, mixup_lambdas = self.mixdown(positive_map, negative_map)
                mixmat  = self.CCT(mixed_map.view(mixed_map.shape[0],3,224,224)).view(mixed_map.shape[0],24,300)
                mixout = self.stage(mixmat,mixed_map.shape[0])
                return output, mixout,mixup_lambdas
            else:
                transmat= self.CCT(transmat.view(batch_size,3,224,224)).view(batch_size,24,300)
                output= self.stage(transmat,batch_size)
                return output, torch.zeros(1,1,device=device),torch.zeros(1,1,device=device)     
        else:
            transmat= self.CCT(transmat.view(batch_size,3,224,224)).view(batch_size,24,300)
            output= self.stage(transmat,batch_size)
            return output
    
