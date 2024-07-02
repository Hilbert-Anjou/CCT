# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:10:28 2022

@author: JyGuo
"""
import numpy as np
import argparse
import os
import math
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
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
from model_SICH_pretrain_CCT import PreTranNet


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
    parser.add_argument('--test_mode', type=int, default=1, help='Test SA-CRNN on MIMIC-III dataset')
    
    #parser.add_argument('--trained_model', type=str, default='trained_model_SICH', help='File name for the saved weights')
    
    parser.add_argument('--data_path', type=str, default='./data/', help='The path to the MIMIC-III data directory')
    parser.add_argument('--file_name', type=str, default='trained_model_SICH', help='File name to save model')
    
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=200, help='Training batch size') # change the batch size smaller in order not to exceed memory
    #parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learing rate')#5e-5
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
        ''' Prepare training data'''
        print('Preparing training data ... ')
        def read_data(ts_filename):
            tsdata=pd.read_csv('./data/'+ts_filename)
            tsdata=tsdata.fillna(-1)
            tsdata.replace({'male': 1, 'female': 0,'TRUE':1,'FALSE':0},inplace = True)
            assert tsdata.isnull().values.any()==False
            return tsdata[0:].to_numpy()
                                       
        trainX=read_data('mimic_train_X.csv')
        trainY=read_data('mimic_train_y.csv')
        testX=read_data('mimic_test_X.csv')
        testY=read_data('mimic_test_y.csv')
        valX=read_data('mimic_val_X.csv')
        valY=read_data('mimic_val_y.csv')

        '''Model structure'''
        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))
        
        '''Train phase'''
        print('Start training ... ')
        def force_cudnn_initialization():
            s = 32
            dev = torch.device('cuda')
            torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

        
        fold=10
        sensitivity_mortality = np.zeros(fold)
        specificity_mortality = np.zeros(fold)
        accuracy_mortality = np.zeros(fold)
        auroc_mortality = np.zeros(fold)
        
        best_model= [None]
        
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        results={}
        bestpoint=[0,0]
        bestacc=0
        auroc=0
        #model = PreTranNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
        #model = torch.load('model/multitrans_100ep.pt')
        model = torch.load('model/pretrain_200ep.pt')
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        
        
        
        scaler = StandardScaler()
        #print(to_normalize.shape)
        trainX = scaler.fit_transform(trainX)
        valX = scaler.transform(valX)
        testX = scaler.transform(testX)
            
        x_train = torch.tensor(trainX, dtype = torch.float32)
        y_train = torch.tensor(trainY, dtype = torch.long)
        torch_dataset = data.TensorDataset(x_train, y_train) 
        loader = data.DataLoader(torch_dataset, batch_size = args.batch_size, shuffle = True)

        x_val = torch.tensor(valX, dtype = torch.float32)

        val_dataset = data.TensorDataset(x_val) 
        val_loader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)

        x_test = torch.tensor(testX, dtype = torch.float32)

        test_dataset = data.TensorDataset(x_test) 
        test_loader = data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
        force_cudnn_initialization() 

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr*3.16*10,weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-7)
        train_loss=[]
        test_auc=[]
        val_auc=[]
        lr_list=[]
        for epoch in range(200):
            print(epoch)
            temp_train_loss=0

            for batch_x, batch_y in loader:
            
                model.train()
                
                
                
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                #batch_discharge = batch_discharge.to(device)

                batch_time = torch.ones((batch_x.size(0), batch_x.size(1)), dtype=torch.float32).to(device)
                
                batch_mortality_output= model(batch_x, batch_time, device)

                
                loss_numu = batch_y * torch.log(batch_mortality_output + 1e-7) + (1 - batch_y) * torch.log(1 - batch_mortality_output + 1e-7)
                loss_mortality = loss_numu / batch_x.shape[0]
                loss_mortality = torch.neg(torch.sum(loss_mortality))
                temp_train_loss+=loss_mortality.item()
                optimizer.zero_grad()
                loss_mortality.backward()
                optimizer.step()
            train_loss.append(temp_train_loss)
            scheduler.step(temp_train_loss)
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            with torch.no_grad():
                model.eval()
                val_mortality_pred=np.zeros((1,12))
                for bacth_xval in val_loader:

                    bacth_xval = bacth_xval[0].to(device)
                    #y_val = torch.tensor(valY, dtype= torch.long).to(device)
                
                    val_mortality_output= model(bacth_xval, None, device)
                    val_mortality_temp = val_mortality_output.data.cpu().numpy().squeeze()
                    val_mortality_pred = np.vstack((val_mortality_pred,val_mortality_temp))
                    #val_discharge_pred = test_discharge_output.data.cpu().numpy()

                    #val_discharge_true = discharge_test.data.cpu().numpy()
                    
                    
                print('validation set')
                val_mortality_ret = metrics.print_metrics_multilabel(valY, val_mortality_pred[1:,:], verbose=0)['auc_scores'].mean()
                print(val_mortality_ret)
                #scheduler.step(val_mortality_ret)
                val_auc.append(val_mortality_ret)
                test_mortality_pred=np.zeros((1,12))
                for bacth_xtest in test_loader:
                    bacth_xtest = bacth_xtest[0].to(device)
                    test_mortality_output= model(bacth_xtest, None, device)
                    

                    test_mortality_temp = test_mortality_output.data.cpu().numpy().squeeze()
                    test_mortality_pred = np.vstack((test_mortality_pred,test_mortality_temp))
                    
                
                print('Mortality Prediction on the test set')
                test_mortality_ret = metrics.print_metrics_multilabel(testY, test_mortality_pred[1:,:], verbose=0)['auc_scores'].mean()
                test_auc.append(test_mortality_ret)
                print(test_mortality_ret)
                if (test_mortality_ret > auroc):
                    auroc = test_mortality_ret
                    
                    best_model=model

        torch.save(best_model, 'model/temp.pt')

        print('AUC of ROC')
        print(auroc)
        
        e=np.array(range(len(train_loss)))
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('epoches')
        ax1.set_ylabel('train loss', color=color)
        ax1.plot(e,train_loss, label='train loss',color='green')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

        color = 'tab:red'
        ax2.set_ylabel('auc', color=color)
        ax2.plot(e,val_auc,label='val auc')
        ax2.plot(e,test_auc,label='test auc')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.legend( bbox_to_anchor=(0.5, 0.8))
        plt.title('avoid overfitting')
        plt.savefig('train_val_test3.png', bbox_inches='tight')

        plt.figure()
        plt.plot(e,lr_list)
        plt.savefig('learning_rate.png', bbox_inches='tight')

    else:
        ''' Prepare training data'''
        print('Preparing training data ... ')
        def read_data(ts_filename):
            tsdata=pd.read_csv('./data/'+ts_filename)
            tsdata=tsdata.fillna(-1)
            tsdata.replace({'male': 1, 'female': 0,'TRUE':1,'FALSE':0},inplace = True)
            assert tsdata.isnull().values.any()==False
            return tsdata[0:].to_numpy()
                                       
        trainX=read_data('mimic_train_X.csv')
        trainY=read_data('mimic_train_y.csv')
        testX=read_data('mimic_test_X.csv')
        testY=read_data('mimic_test_y.csv')
        valX=read_data('mimic_val_X.csv')
        valY=read_data('mimic_val_y.csv')

        '''Model structure'''
        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))
        
        '''Train phase'''
        print('Start training ... ')
        def force_cudnn_initialization():
            s = 32
            dev = torch.device('cuda')
            torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

        
        fold=10
        sensitivity_mortality = np.zeros(fold)
        specificity_mortality = np.zeros(fold)
        accuracy_mortality = np.zeros(fold)
        auroc_mortality = np.zeros(fold)
        auroc=0
        best_model= [None]
        
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()




        
        alpha = 1.0
        beta = 0.5


        lr_range=np.logspace(-0.5,2,num=5)
        reg_range=np.logspace(-4,-0.5,num=5)
        lr_mat,reg_mat=np.meshgrid(lr_range, reg_range)
        print(lr_range,reg_range)
        results={}
        bestpoint=[0,0]
        bestacc=0
        #model = PreTranNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
        #model = torch.load('model/multitrans_100ep.pt')
        model = torch.load('model/pretrain_20ep.pt')
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70,90], gamma=0.5, last_epoch=-1)
        
        scaler = StandardScaler()
        #print(to_normalize.shape)
        trainX = scaler.fit_transform(trainX)
        valX = scaler.transform(valX)
        testX = scaler.transform(testX)
            
        x_train = torch.tensor(trainX, dtype = torch.float32)
        y_train = torch.tensor(trainY, dtype = torch.long)
        torch_dataset = data.TensorDataset(x_train, y_train) 
        loader = data.DataLoader(torch_dataset, batch_size = args.batch_size, shuffle = True)

        x_val = torch.tensor(valX, dtype = torch.float32)

        val_dataset = data.TensorDataset(x_val) 
        val_loader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)

        x_test = torch.tensor(testX, dtype = torch.float32)

        test_dataset = data.TensorDataset(x_test) 
        test_loader = data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)
        force_cudnn_initialization()
        for i in range(25):
            lr=lr_mat[i//5,i%5]
            reg=reg_mat[i//5,i%5]
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr*lr,weight_decay=reg)
            for epoch in range(10):
                print(epoch)
                for batch_x, batch_y in loader:
                #for batch_x, batch_y, batch_discharge in loader:                               # for each batch
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

                    optimizer.zero_grad()
                    loss_mortality.backward()
                    optimizer.step()
                    #scheduler.step(loss_mortality)

                with torch.no_grad():
                    model.eval()
                    val_mortality_pred=np.zeros((1,12))
                    for bacth_xval in val_loader:

                        bacth_xval = bacth_xval[0].to(device)
                        #y_val = torch.tensor(valY, dtype= torch.long).to(device)
                  
                        val_mortality_output= model(bacth_xval, None, device)
                        val_mortality_temp = val_mortality_output.data.cpu().numpy().squeeze()
                        val_mortality_pred = np.vstack((val_mortality_pred,val_mortality_temp))
                        #val_discharge_pred = test_discharge_output.data.cpu().numpy()

                        #val_discharge_true = discharge_test.data.cpu().numpy()
                        
                        
                    print('validation set')
                    val_mortality_ret = metrics.print_metrics_multilabel(valY, val_mortality_pred[1:,:], verbose=0)['auc_scores'].mean()
                    test_mortality_pred=np.zeros((1,12))
                    for bacth_xtest in test_loader:
                        bacth_xtest = bacth_xtest[0].to(device)
                        test_mortality_output= model(bacth_xtest, None, device)
                        

                        test_mortality_temp = test_mortality_output.data.cpu().numpy().squeeze()
                        test_mortality_pred = np.vstack((test_mortality_pred,test_mortality_temp))
                        
                    
                    print('Mortality Prediction on the test set')
                    test_mortality_ret = metrics.print_metrics_multilabel(testY, test_mortality_pred[1:,:], verbose=0)['auc_scores'].mean()
                    
                    if (test_mortality_ret > auroc):
                        auroc = test_mortality_ret
                        
                        best_model=model
            if val_mortality_ret>bestacc:
                bestacc=val_mortality_ret
                bestpoint[0]=lr
                bestpoint[1]=reg
            results.update({(lr,reg):val_mortality_ret})
            model.apply(weight_reset)
        torch.save(best_model, 'model/temp.pt')

        print('AUC of ROC')
        print(auroc)
        
        x_scatter = [math.log10(x[0]) for x in results]
        y_scatter = [math.log10(x[1]) for x in results]
        marker_size = 100  # default: 20
        colors = [results[x] for x in results]  # depend color on val_acc
        plt.figure()

        plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
        plt.annotate('(%.2f,%.2f,%.2f%%)'% (bestpoint[0],bestpoint[1],bestacc*100), 
                    xy=bestpoint, xytext=(-30, 30), textcoords='offset pixels',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        plt.colorbar()
        plt.xlabel('10^lr')
        plt.ylabel('10^reg')
        plt.title('validation auc')
        plt.savefig('squares_plot3.png', bbox_inches='tight')