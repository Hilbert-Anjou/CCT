# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 19:53:32 2022

@author: JyGuo
"""

import torch
from torch import nn

def IBLoss(masked_output, batch_y, for_gradient, batch_mask):
    num_live = 2134907                                                    # the total number of sampels from class 0 and class 1
    num_dead = 45253                                                      # use for the weight of Influenced-balanced loss
    class_0_weight = (1/num_live)/(1/num_live + 1/num_dead) * 2
    class_1_weight = (1/num_dead)/(1/num_live + 1/num_dead) * 2
    #class_0_weight = 1
    #class_1_weight = 1
    
    alpha = 0.01
    
    loss = class_1_weight * batch_y * torch.log(masked_output + 1e-12) + class_0_weight * (1 - batch_y) * torch.log(1 - masked_output + 1e-12)
    grads = torch.abs(masked_output - batch_y) * 2
    #for_gradient = torch.sum(torch.abs(for_gradient), 2).unsqueeze(2)
    for_gradient = grads * for_gradient
    for_gradient = alpha / (for_gradient + 1e-12)
    loss = loss * for_gradient
    loss = loss * batch_mask
    
    loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
    batch_size = batch_y.shape[0]
    loss = torch.neg(torch.sum(loss)) / batch_size
    
    return loss
    
'''
class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.05):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        #self.weight = weight
        self.num_live = 2134907                                                    # the total number of sampels from class 0 and class 1
        self.num_dead = 45253                                                      # use for the weight of Influenced-balanced loss
        self.class_0_weight = (1/self.num_live)/(1/self.num_live + 1/self.num_dead) * 2
        self.class_1_weight = (1/self.num_dead)/(1/self.num_live + 1/self.num_dead) * 2

    def forward(self, masked_output, batch_y, for_gradient, batch_mask):
        loss = self.class_1_weight * batch_y * torch.log(masked_output + 1e-12) + self.class_0_weight * (1 - batch_y) * torch.log(1 - masked_output + 1e-12)
        grads = torch.abs(masked_output - batch_y) * 2
        
        for_gradient = grads * for_gradient
        for_gradient = self.alpha / (for_gradient + 1e-12)
        
        loss = loss * for_gradient
        loss = loss * batch_mask
        
        loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
        loss = torch.neg(torch.sum(loss))
        
        return loss
'''