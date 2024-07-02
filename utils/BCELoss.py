# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:21:54 2022

@author: JyGuo
"""

import torch

def BCELoss(batch_y, masked_output, batch_mask):
    
    batch_size = batch_y.shape[0]
    
    loss = batch_y * torch.log(masked_output + 1e-7) +  (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
    loss = loss * batch_mask
    
    loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
    loss = torch.neg(torch.sum(loss)) / batch_size
    
    return loss