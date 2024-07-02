# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:22:17 2022

@author: CHEN YUHUA
"""
import os
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as models
from Compact_Transformers.src import cct_7_3x1_32_sine_c100,cct_7_3x1_32_sine,cct_14_7x2_224,cct_14_7x2_384
from Compact_Transformers.src.text import text_cct_6
import numpy as np
from simpnet import Linear_BBB
RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
class StageTranNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.):
        super(StageTranNet, self).__init__()
        self.first_conv  = nn.Sequential(                       # apply ResNet to replace CNN backbone
            #torch.nn.Dropout2d(p=0.3), 
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )

        self.Encoder = models.resnet18(pretrained=False)
        self.dim_reconstruct = nn.Linear(1000, 18432)
        #self.dim_reconstruct = nn.Linear(1000, 6272)
        #self.dim_reconstruct = Linear_BBB(1000, 6272)
        
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim

        hidden_dim=200
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

        #self.CCT=cct_14_7x2_224(pretrained=True, progress=True,num_classes=7200)
        self.CCT=cct_14_7x2_384(pretrained=True, progress=True,num_classes=4800)
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
    
    def forward(self, input, device=torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')):
        batch_size, time_step, feature_dim = input.size()

        #transmat = torch.zeros(batch_size,24, 6272).to(device)
        transmat = torch.zeros(batch_size,3,384,384).to(device)
        for t in range(24):   # to process the input data based on each timestep (variable t) 
            encode_input = input[:,t,:].reshape(input[:,t,:].shape[0],1,33,25)
            encode_input = self.first_conv(encode_input)                      # the next 3 lines are for resnet
            
            encode = self.Encoder(encode_input)

            encode = self.dim_reconstruct(encode)
            #transmat= torch.cat((transmat[:,1:,:], encode.view(batch_size,1,6272)), 1)

            transmat= torch.cat((transmat[:,:,16:,:], encode.view(batch_size,3,16,384)), 2)

        transmat = self.CCT(transmat.view(batch_size,3,384,384)).view(batch_size,24,200)
        #transmat= self.CCT(transmat.view(batch_size,3,224,224)).view(batch_size,24,300)



        local_h = transmat.permute(0, 2, 1)                                  # after permutation, the size is batch_size by hidden_dim by conv_size
        local_theme = self.nn_avg(local_h).view(batch_size,-1)                        # to weight the past health state info based on the relavant time gap
        local_theme = self.nn_scale(local_theme)                          # input: hidden_dim, output: hidden_dim //6
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
        output = torch.sigmoid(output)
        return output

class ManifoldMixupModel(nn.Module):
    def __init__(self, model, num_classes = 10, alpha = 1):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lam = None
        self.num_classes = num_classes
        ##选择需要操作的层，在ResNet中各block的层名为layer1,layer2...所以可以写成如下。其他网络请自行修改
        self. module_list = []
        for n,m in self.model.named_modules():
            #if 'conv' in n:
            if n == 'CCT':
                self.module_list.append(m)

    def forward(self, x, target=None):
        if target==None:
            out = self.model(x)
            return out
        else:
            if self.alpha <= 0:
                self.lam = 1
            else:
                self.lam = np.random.beta(self.alpha, self.alpha)
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            target_onehot = to_one_hot(target, self.num_classes)
            target_shuffled_onehot = target_onehot[self.indices]
            if k == -1:
                x = x * self.lam + x[self.indices] * (1 - self.lam)
                out = self.model(x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
            target_reweighted = target_onehot* self.lam + target_shuffled_onehot * (1 - self.lam)
            bce_loss = nn.BCELoss()
            loss = bce_loss(out, target_reweighted)
            return out, loss
        
    def hook_modify(self, module, input, output):
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        return output

class PatchUpModel(nn.Module):
    def __init__(self, model, num_classes=2, block_size=1, gamma=0.5, patchup_type='hard', keep_prob=0.7):
        super().__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.computed_lam = None
        
        self.model = model
        self.num_classes = num_classes
        self.module_list = []
        for n, m in self.model.named_modules():
            if n == 'CCT':
                self.module_list.append(m)

    def adjust_gamma(self, x):
        return self.gamma * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x, target=None):
        if target is None:
            out = self.model(x)
            return out
        else:
            self.lam = np.random.beta(2.0, 2.0)
            #self.lam = np.random.beta(0.2, 0.2)
            k = np.random.randint(0, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            self.target = target.to(dtype=torch.float)
            self.target_shuffled = self.target[self.indices]

            if k == -1:  # CutMix
                W, H = x.size(2), x.size(3)
                cut_rat = np.sqrt(1. - self.lam)
                cut_w = np.int(W * cut_rat)
                cut_h = np.int(H * cut_rat)
                cx = np.random.randint(W)
                cy = np.random.randint(H)
        
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)
                
                x[:, :, bbx1:bbx2, bby1:bby2] = x[self.indices, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                out = self.model(x)
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(out, self.target) * lam + loss_fn(out, self.target_shuffled) * (1. - lam)

            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
                
                loss_fn = nn.BCELoss()
                
                loss = loss_fn(out, self.target_a) * self.total_unchanged_portion + loss_fn(out, self.target_b) * (1. - self.total_unchanged_portion) + \
                        loss_fn(out, self.target_reweighted)
            return out, loss
        
    def hook_modify(self, module, input, output):
        output=output.view(output.shape[0],1,24,200)
        self.gamma_adj = self.adjust_gamma(output)
        p = torch.ones_like(output[0]) * self.gamma_adj
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)
        m_i_j = m_i_j.expand(output.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)
        mask = 1 - holes
        unchanged = mask * output
        if mask_shape == 1:
            total_feats = output.size(1)
        else:
            total_feats = output.size(1) * output.size(2)*output.size(3)
        total_changed_pixels = holes[0].sum()
        total_changed_portion = total_changed_pixels / total_feats
        self.total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        if self.patchup_type == 'hard':
            self.target_reweighted = self.total_unchanged_portion * self.target +total_changed_portion * self.target_shuffled
            patches = holes * output[self.indices]
            self.target_b = self.target[self.indices]
        elif self.patchup_type == 'soft':
            self.target_reweighted = self.total_unchanged_portion * self.target +\
                self.lam * total_changed_portion * self.target +\
                (1 - self.lam) * total_changed_portion * self.target_shuffled
            patches = holes * output
            patches = patches * self.lam + patches[self.indices] * (1 - self.lam)
            self.target_b = self.lam * self.target + (1 - self.lam) * self.target_shuffled
        else:
            raise ValueError("patchup_type must be 'hard' or 'soft'.")
        
        output = unchanged + patches
        self.target_a = self.target
        return output