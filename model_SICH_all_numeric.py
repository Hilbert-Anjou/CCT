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

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

class StageNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.3):
        super(StageNet, self).__init__()
        
        '''
        self.Encoder = nn.Sequential(
            nn.Linear(825, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 128)
            #nn.ReLU()
        )
        '''
        '''
        self.Encoder = nn.Sequential( #input size [128,1,33,25]
            #torch.nn.Dropout2d(p=0.3),                                          # apply dropout before passing data to autoencoder, achieving the goal of adding noise
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            #torch.nn.Dropout2d(p=0.3),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        '''
        '''
        self.Encoder = nn.Sequential( #input size [128,1,33,25]
            torch.nn.Dropout2d(p=0.3),                                          # apply dropout before passing data to autoencoder, achieving the goal of adding noise
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            #torch.nn.Dropout2d(p=0.3),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        '''
        '''
        self.Encoder = nn.Sequential( #input size [128,1,33,25]
            torch.nn.Dropout2d(p=0.3),                                          # apply dropout before passing data to autoencoder, achieving the goal of adding noise
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            torch.nn.Dropout2d(p=0.3),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),

            #torch.nn.Dropout2d(p=0.3),
            #nn.Conv2d(16, 8, 3, stride=2, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(2, stride=1)
        )
        '''
        self.first_conv  = nn.Sequential(                       # apply ResNet to replace CNN backbone
            #torch.nn.Dropout2d(p=0.3), 
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        self.Encoder = models.resnet18(pretrained=False)
        self.dim_reduce = nn.Linear(1000, 280)
        
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
        
        self.nn_scale_discharge = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale_discharge = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv_discharge = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), 1)
        self.nn_output_discharge = nn.Linear(int(self.conv_dim), int(output_dim))
        
        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

        
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
    def step(self, inputs, c_last, h_last, interval):                         # in this function, the size of inputs is batch_size by 76 + 17
        x_in = inputs                                                         # which is why the NN is nothing but linear ones

        # Integrate inter-visit time intervals
        interval = interval.unsqueeze(-1)                                     # output of both kernel is output batch_size by hidden_dim*4+levels*2
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1))             # to process the input for this round
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1)) # to process the h matrix from last RNN output
        
        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2                                               # dimension is batch_size by hidden_dim*4+levels*2
        
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r')             # to compute forget gate ft in the paper, size batch_size by levels by 1
        f_master_gate = f_master_gate.unsqueeze(2)                            # remove the 3rd dimension index
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels*2], 'r2l')# to compute input gate it in the paper
        i_master_gate = i_master_gate.unsqueeze(2)
        
        x_out = x_out[:, self.levels*2:]                                      # dimension is batch_size by hidden_dim*4
        x_out = x_out.reshape(-1, self.levels*4, self.chunk_size)             # dimension is batch_size by levels*4 by hidden_dim/levels
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels*2])           # the p_f vector in the paper (without doing the cumulative sum)
        o_gate = torch.sigmoid(x_out[:, self.levels*2:self.levels*3])
        
        c_in = torch.tanh(x_out[:, self.levels*3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)             # reshape the last cell state matrix, since another reshape later
        
        overlap = f_master_gate * i_master_gate                               # update the cell state h and c
        
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (i_master_gate - overlap) * c_in # by definition
        h_out = o_gate * torch.tanh(c_out)
        
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1) # f_master_gate[..., 0] is batch_size by levels
        return out, c_out, h_out                                              # the dimension of out is batch_size by hidden_dim + levels*2
                                                                              # 64 * 390 in my case
    
    def forward(self, input, time, device):
        batch_size, time_step, feature_dim = input.size()
        c_out = torch.zeros(batch_size, self.hidden_dim).to(device) # for the 1st RNN input, h and c are all set to be 0
        h_out = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(self.conv_size, batch_size, self.hidden_dim).to(device)
        tmp_dis = torch.zeros((self.conv_size, batch_size)).to(device)
        h = []
        h_two = []
        origin_h = []
        distance = []
        
        for t in range(time_step):   # to process the input data based on each timestep (variable t)
            encode_input = input[:,t,:].reshape(input[:,t,:].shape[0],1,33,25)

            encode_input = self.first_conv(encode_input)                      # the next 3 lines are for resnet
            encode = self.Encoder(encode_input)
            encode = self.dim_reduce(encode)

            '''
            encode = self.Encoder(encode_input)                               # the next 2 lines are for CNN backbone
            encode = encode.reshape(encode.shape[0],-1)
            '''
            #out, c_out, h_out = self.step(input[:, t, :], c_out, h_out, time[:, t])
            out, c_out, h_out = self.step(encode, c_out, h_out, time[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim+self.levels], -1) # size is just batch_size  # f-mastergate
            cur_distance_in = torch.mean(out[..., self.hidden_dim+self.levels:], -1)
            origin_h.append(out[..., :self.hidden_dim])                       # out[..., :self.hidden_dim] is the last cell state info h, batch_size by hidden_dim

            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0) # tmp_h remain the size, conv_size by batch_size by hidden_dim
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)  # conv_size is the windows size
            distance.append(cur_distance)                                     # tmp_h and tmp_dis are both used to record the past data/value
            
        #Re-weighted convolution operation
        local_dis = tmp_dis.permute(1, 0)                                 # after permuatation, the size is batch_size by conv_size  #st
        local_dis = torch.cumsum(local_dis, dim=1)                        
        local_dis = torch.softmax(local_dis, dim=1)                       # delta st
        
        local_h = tmp_h.permute(1, 2, 0)                                  # after permutation, the size is batch_size by hidden_dim by conv_size
        S = local_dis.unsqueeze(1)
        H = local_h
        local_h = local_h * local_dis.unsqueeze(1)                        # to weight the past health state info based on the relavant time gap
        
        #Re-calibrate Progression patterns
        local_theme = torch.mean(local_h, dim=-1)                         # z_t in the paper, the average of past health state
        local_theme = self.nn_scale(local_theme)                          # input: hidden_dim, output: hidden_dim //6
        #local_theme = F.prelu(local_theme,weight=prelu)
        local_theme = torch.relu(local_theme) 
        local_theme = self.nn_rescale(local_theme)                        # input: hidden_dim//6, output: hidden_dim
        
        local_theme = torch.sigmoid(local_theme)                          # the size of local_h, this is the attention part
        # before conv, the size is batch_size by hidden_dim by conv_size
        local_h = self.nn_conv(local_h).squeeze(-1)                       # after conv, the size is batch_size by hidden_dim
        local_h = local_theme * local_h                                   # weight based on 
        h.append(local_h)  

        rnn_output = local_h
        
        origin_single_h = out[..., :self.hidden_dim]
        
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
        return output, H, S
