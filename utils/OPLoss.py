# -*- coding: utf-8 -*-
"""
Created on Sun May 29 20:56:41 2022

@author: JyGuo
"""

import torch
import torch.nn.functional as F


def OrthogonalProjectionLoss(features, batch_mask, device, labels=None):

    gamma = 1.0
    #device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    #  features are normalized
    #print('features')
    #print(features.shape)
    features = F.normalize(features, p=2, dim=1)
    #print(features.shape)
    
    labels = labels.reshape(-1)
    labels = labels[:, None]  # extend dim
    real_mask = batch_mask.reshape(-1)
    real_mask = real_mask[:, None]
    #print('labels & mask')
    #print(labels.shape)
    #print(real_mask.shape)

    #print('reality')
    reality = torch.mm(real_mask, real_mask.t())
    #print(reality.shape)
    mask = torch.eq(labels, labels.t()).bool()
    eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

    mask_pos = mask.masked_fill(eye, 0)
    mask_pos = mask_pos.float()
    mask_pos = mask_pos * reality
    
    mask_neg = (~mask)
    mask_neg = mask_neg.float()
    mask_neg = mask_neg * reality
    
    dot_prod = torch.matmul(features, features.t())
    #dot_prod = dot_prod * reality

    pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
    neg_pairs_mean = (abs(mask_neg * dot_prod)).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

    loss = (1.0 - pos_pairs_mean) + gamma * neg_pairs_mean

    return loss