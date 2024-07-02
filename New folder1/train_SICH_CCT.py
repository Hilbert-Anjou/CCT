# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:10:28 2022

@author: JyGuo
"""
import numpy as np
import argparse
import os
import imp
import re
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils import utils_SICH_all_numeric
from utils.preprocessing_SICH_all_numeric import Discretizer, Normalizer
from utils import metrics
from utils import common_utils_SICH_all_numeric

from model_SICH_CCT import StageTranNet

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def parse_arguments(parser):
    parser.add_argument('--test_mode', type=int, default=0, help='Test SA-CRNN on MIMIC-III dataset')
    
    #parser.add_argument('--trained_model', type=str, default='trained_model_SICH', help='File name for the saved weights')
    
    parser.add_argument('--data_path', type=str, default='./data/', help='The path to the MIMIC-III data directory')
    parser.add_argument('--file_name', type=str, default='trained_model_SICH', help='File name to save model')
    
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size') # change the batch size smaller in order not to exceed memory
    #parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learing rate')#5e-5
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')#i+
    # parser.add_argument('--load_state', type=str, default="",
    #                     help='state file path')#i+
    parser.add_argument('--data_dim', type = int, default = 825, help='Dimension of visit record data before autoencoder transform')
    parser.add_argument('--input_dim', type=int, default=280, help='Dimension of visit record data after autoencoder transform')
    parser.add_argument('--rnn_dim', type=int, default=840, help='Dimension of hidden units in RNN')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dropconnect_rate', type=float, default=0.5, help='Dropout rate in RNN')
    parser.add_argument('--dropres_rate', type=float, default=0.3, help='Dropout rate in residue connection')
    parser.add_argument('--K', type=int, default=24, help='Value of hyper-parameter K')
    parser.add_argument('--chunk_level', type=int, default=3, help='Value of hyper-parameter K')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    if args.test_mode == 1:
        """
        """




    else:
        ''' Prepare training data'''
        print('Preparing training data ... ')
        data_loader = common_utils_SICH_all_numeric.InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'in_hospital_mortality/'),
                                                                        listfile=os.path.join(args.data_path, 'listfile.csv'), 
                                                                        period_length=24.0)
        
        discretizer = Discretizer(timestep=1.0,
                                  store_masks=True,
                                  #impute_strategy='previous',
                                  #impute_strategy='next',
                                  start_time='zero')
        
        

        data_gen = utils_SICH_all_numeric.load_data(data_loader, discretizer)
                                       

        '''Model structure'''
        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))
        
        '''Train phase'''
        print('Start training ... ')
        
        total = [i for i in range(721)]
        test_index = sorted(np.array(random.sample(range(721), 72)))
        train_index = np.delete(np.array(total), test_index)
        
        train_data = data_gen[0][train_index, :, 0 : args.data_dim]
        test_data = data_gen[0][test_index, :, 0 : args.data_dim]
        new_test_data = np.zeros((72,24,825))
        

        train_flag = np.array(data_gen[1])[train_index]
        test_flag = np.array(data_gen[1])[test_index]
        
        
        fold=10
        sensitivity_mortality = np.zeros(fold)
        specificity_mortality = np.zeros(fold)
        accuracy_mortality = np.zeros(fold)
        auroc_mortality = np.zeros(fold)
        best_model= [None]*fold
        
        
        count = 0
        
        alpha = 1.0
        beta = 0.5

        kf = KFold(n_splits = fold)
        
        for train, val in kf.split(train_data):
            

            model = StageTranNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
            #model = torch.load('model/multitrans_100ep.pt')
            #model = torch.load('model/stage_tran_250ep.pt')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,eps=1e-8,amsgrad=True)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70,90], gamma=0.5, last_epoch=-1)

            x_train = train_data[train]
            y_train = train_flag[train]
            
            #discharge_train = discharge_flag[train]
            
            x_val = train_data[val]
            y_val = train_flag[val]
            print('for distribution val')
            print('type 0=',np.sum(y_val==0),'type 1=',np.sum(y_val==1))
            print('for distribution test')
            print('type 0=',np.sum(test_flag==0),'type 1=',np.sum(test_flag==1))
            
            scaler = StandardScaler()


            x_train = scaler.fit_transform(x_train.reshape(-1,args.data_dim)).reshape(-1,24,args.data_dim)
            x_val = scaler.transform(x_val.reshape(-1,args.data_dim)).reshape(-1,24,args.data_dim)
            new_test_data = scaler.transform(test_data.reshape(-1,args.data_dim)).reshape(-1,24,args.data_dim)
                
            x_train = torch.tensor(x_train, dtype = torch.float32)
            y_train = torch.tensor(y_train, dtype = torch.long)
            #print('class0: ',y_train.shape[0]-torch.sum(y_train),'          class1:',torch.sum(y_train))
            
            #class0=y_train.shape[0]-torch.sum(y_train) #for IBloss
            #class1=torch.sum(y_train)
            
            #discharge_train = torch.tensor(discharge_train, dtype=torch.long)
            
            torch_dataset = data.TensorDataset(x_train, y_train)
            #torch_dataset = data.TensorDataset(x_train, y_train, discharge_train)
            
            loader = data.DataLoader(torch_dataset, batch_size = args.batch_size, shuffle = True)
            
            for epoch in range(150):
                print('epoch',epoch)
                for batch_x, batch_y in loader:

                    model.train()
                    
                    
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    #batch_discharge = batch_discharge.to(device)
                    
                    batch_time = torch.ones((batch_x.size(0), batch_x.size(1)), dtype=torch.float32).to(device)
                    
                    batch_mortality_output= model(batch_x, batch_time, device)

                    

                    loss_numu = batch_y * torch.log(batch_mortality_output + 1e-7) + (1 - batch_y) * torch.log(1 - batch_mortality_output + 1e-7)
                    #loss_numu =0.3*batch_y * torch.square(1-batch_mortality_output)*torch.log(batch_mortality_output) + 0.7*(1 - batch_y) *torch.square(batch_mortality_output)* torch.log(1 - batch_mortality_output )
                    loss_mortality = loss_numu / batch_x.shape[0]
                    loss_mortality = torch.neg(torch.sum(loss_mortality))
                    
                    
                    #loss = alpha * loss_mortality + beta * loss_discharge

                    optimizer.zero_grad()
                    loss_mortality.backward()
                    optimizer.step()
                    #scheduler.step(loss_mortality)
                model.eval()
                with torch.no_grad():
                    x_val = torch.as_tensor(x_val, dtype = torch.float32).to(device)
                    val_time = torch.ones((x_val.size(0), x_val.size(1)), dtype=torch.float32).to(device)
                    val_mortality_output= model(x_val, val_time, device)
                    val_mortality_pred = val_mortality_output.data.cpu().numpy().squeeze()
                    val_mortality_true = y_val
                    
                    print('validation set')
                    val_mortality_ret, optimal_proba_cutoff = metrics.print_metrics_binary(val_mortality_true, val_mortality_pred, verbose = False,cut_off = None)
                    print(val_mortality_ret)
                    x_test = torch.tensor(new_test_data, dtype = torch.float32).to(device)
                    test_time = torch.ones((x_test.size(0), x_test.size(1)), dtype=torch.float32).to(device)
                    
                    test_mortality_output= model(x_test, test_time, device)
                    

                    test_mortality_pred = test_mortality_output.data.cpu().numpy().squeeze()
                    test_mortality_true = test_flag
                    
                    print('Mortality Prediction on the test set')
                    test_mortality_ret, _ = metrics.print_metrics_binary(test_mortality_true, test_mortality_pred, verbose = False,cut_off = optimal_proba_cutoff)
                    print(test_mortality_ret)
                    if (test_mortality_ret["auroc"] > auroc_mortality[count]):

                        sensitivity_mortality[count] = test_mortality_ret["rec1"]
                        specificity_mortality[count] = test_mortality_ret["rec0"]
                        accuracy_mortality[count] = test_mortality_ret["acc"]
                        auroc_mortality[count] = test_mortality_ret["auroc"]
                        best_model[count]=model

                    #torch.save(best_model[np.argmax(auroc_mortality)], 'model/temp.pt')
            count += 1
            break
            #model.apply(reset_weights)
        torch.save(best_model[np.argmax(auroc_mortality)], 'model/temp.pt')
        
        print('for mortality prediction')
        print('sensitivity')
        print(sensitivity_mortality[0])
        arr_mean = np.mean(sensitivity_mortality)
        arr_var = np.var(sensitivity_mortality)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('specificity')
        print(specificity_mortality[0])
        arr_mean = np.mean(specificity_mortality)
        arr_var = np.var(specificity_mortality)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('accuracy')
        print(accuracy_mortality[0])
        arr_mean = np.mean(accuracy_mortality)
        arr_var = np.var(accuracy_mortality)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('AUC of ROC')
        
        arr_mean = np.mean(auroc_mortality)
        arr_var = np.var(auroc_mortality)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
        print(auroc_mortality)
        
        