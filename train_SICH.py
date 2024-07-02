# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:15:14 2022

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

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils import utils_SICH
from utils.readers import DecompensationReader
from utils.preprocessing_SICH import Discretizer, Normalizer
from utils import metrics
from utils import common_utils_SICH
from model_SICH import StageNet

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
    
    #parser.add_argument('--train_listfile', type=str, metavar='<data_path>', help='File name for the training samples')
    #parser.add_argument('--trained_model', type=str, metavar='<data_path>', help='File name for the saved weights')
    
    #parser.add_argument('--data_path', type=str, metavar='./data', help='The path to the MIMIC-III data directory')
    #parser.add_argument('--file_name', type=str, metavar='saved_model_test', help='File name to save model')
    
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size') # change the batch size smaller in order not to exceed memory
    #parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learing rate')
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')#i+
    # parser.add_argument('--load_state', type=str, default="",
    #                     help='state file path')#i+

    parser.add_argument('--input_dim', type=int, default=27, help='Dimension of visit record data')
    parser.add_argument('--rnn_dim', type=int, default=90, help='Dimension of hidden units in RNN')
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
        print('Preparing test data ... ')

        train_data_loader = common_utils_SICH.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'train_listfile.csv'), small_part=True)
        
        
        '''
        print('train')
        train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'train_listfile.csv'), small_part=False)
        print('test')
        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data_path, 'test'),
                                                                        listfile=os.path.join(args.data_path, 'test_listfile.csv'), small_part=args.small_part)
        print('val')
        val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'val_listfile.csv'), small_part=args.small_part)
        '''
        
        discretizer = Discretizer(timestep=1.0, store_masks=True,
                                impute_strategy='previous', start_time='zero')

        discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    #
        print(cont_channels) # the non-categorical channels
    #
        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'decomp_normalizer'
        normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
        normalizer.load_params(normalizer_state)


        test_data_loader = common_utils_SICH.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data_path, 'test'),
                                                                        listfile=os.path.join(args.data_path, 'test_listfile.csv'), small_part=args.small_part)
        test_data_gen = utils_SICH.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                    normalizer, args.batch_size,
                                                    shuffle=False, return_names=True)

        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        # #
        # model = StageNet(76 + 17, 384, 10, 1, 3, 0.3, 0.3, 0.3)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model ,device_ids=[0,1])                # 2 gpu
        # model.to(device)
        # #
        print("available device: {}".format(device))

        model = StageNet(76+17, 384, 10, 1, 3, 0.3, 0.3, 0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # checkpoint = torch.load('./saved_weights/trained_model_Jingyuan_2',map_location ='cpu') #load the weights into the newly created model frame
        #checkpoint = torch.load('./saved_weights/trained_model_IBLoss')
        weight_path = './saved_weights/' + args.trained_model
        checkpoint = torch.load(weight_path)
        save_chunk = checkpoint['chunk']
        print("last saved model is in chunk {}".format(save_chunk))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        with torch.no_grad():
            cur_test_loss = []
            test_true = []
            test_pred = []
            names = []
            # test_names_all = []
            
            for each_batch in range(test_data_gen.steps):


                #
                # test_data = next(test_data_gen)
                # test_data = test_data['data']
                # test_name = test_data['names']

                test_data = next(test_data_gen)
                test_name = test_data['names']
                test_data = test_data['data']

                # test_name = np.array(test_name.repeat(test_data[0].shape[1], axis=-1))



                # print(len(test_data))
                # print(test_name)

                # test_ts = test_data['ts']      # me +
                # for single_ts in test_ts:      #
                #     ts += single_ts            #

                                                                                       # for each batch, the timesteps are padded with zeros
                test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device) # the transformed value of input, (batch_size, time_step, 76)
                test_mask = torch.tensor(test_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device) # record which hour/timestep is real data
                test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)    # the ground truth
                tmp = torch.zeros(test_x.size(0),17, dtype=torch.float32).to(device)   # batch_size by 17 matrix with all 0's
                test_interval = torch.zeros((test_x.size(0),test_x.size(1),17), dtype=torch.float32).to(device) # 


                # test_name = np.array(test_name).repeat(test_x.shape[1], axis=-1) # i +
                # print(test_name)


                for i in range(test_x.size(1)): # for every timestep
                    cur_ind = test_x[:,i,-17:]  # the masks in preprocessing, to indicate which data are fake/generated data
                    tmp+=(cur_ind == 0).float() # the size of tmp and cur_ind are all (64, 17)
                    test_interval[:, i, :] = cur_ind * tmp
                    tmp[cur_ind==1] = 0         # test_interval is used to check the interval timestep between 2 real data from the same category
                #
                # print(test_mask.size()[1])
                #

                if test_mask.size()[1] > 400: # if number of timesteps exceeds 400
                    test_x = test_x[:, :400, :]
                    test_mask = test_mask[:, :400, :] # include only the first 400 timestep data
                    test_y = test_y[:, :400, :]
                    test_interval = test_interval[:, :400, :]
                    # test_name = test_name[:400]# i +

                #
                # print(test_mask.size()[1])
                #
                
                test_x = torch.cat((test_x, test_interval), dim=-1) # here the input dimension becomes 76 + 17, batch_size X timestep X 76 + 17

                # #
                # print(test_x.shape)
                # #

                test_time = torch.ones((test_x.size(0), test_x.size(1)), dtype=torch.float32).to(device) # batch_size by timestep by 76 + 17
                
                test_output, test_dis = model(test_x, test_time, device)

                #
                # print(len(test_output))
                #

                masked_test_output = test_output * test_mask # only check those real lables 

                test_loss = test_y * torch.log(masked_test_output + 1e-7) + (1 - test_y) * torch.log(1 - masked_test_output + 1e-7)
                test_loss = torch.sum(test_loss, dim=1) / torch.sum(test_mask, dim=1)
                test_loss = torch.neg(torch.sum(test_loss))
                cur_test_loss.append(test_loss.cpu().detach().numpy())

                # print(test_x.shape)
                test_name = np.array(test_name).repeat(test_x.shape[1], axis=-1)  # i +

                # print(test_name.shape)
                
                # for m, t, p in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
                for m, t, p, name in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(),
                                   test_output.cpu().detach().numpy().flatten(), test_name.flatten()):        # me plus
                    if np.equal(m, 1):
                        test_true.append(t)
                        test_pred.append(p)
                        names.append(name)


            print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
            print('\n')
            test_pred = np.array(test_pred)
            # test_pred = np.stack([1 - test_pred, test_pred], axis=1)

            test_pred = np.stack([1 - test_pred, test_pred], axis=1)
            test_ret = metrics.print_metrics_binary(test_true, test_pred)

            #
            print(len(test_pred))
            print(len(test_true))
            print(len(test_name))
            print(len(test_x))
            print(len(names))
            #

            # # path = os.path.join(args.output_dir, 'test_predictions_my') + '.csv'
            # path = os.path.join(args.output_dir, 'test_predictions_Jingyuan') + '.csv'
            # utils.save_my_results(names, test_pred, test_true, path)





    else:
        ''' Prepare training data'''
        print('Preparing training data ... ')
        data_loader = common_utils_SICH.InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'in_hospital_mortality_10/'),
                                                                        listfile=os.path.join(args.data_path, 'listfile.csv'), 
                                                                        period_length=24.0)
        #val_data_loader = common_utils_SICH.DeepSupervisionDataLoader(dataset_dir=os.path.join(
        #    args.data_path, 'train'), listfile=os.path.join(args.data_path, 'val_listfile.csv'), small_part=args.small_part)
        discretizer = Discretizer(timestep=1.0,
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')

        discretizer_header = discretizer.transform(data_loader.read_example(0)["X"])[1].split(',')
        #print(discretizer_header)
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        #
        #print(cont_channels)
        #
        
        # normalizer 需要进一步处理，因为我们要对数据进行 K-Fold Cross Validation
        #normalizer = Normalizer(fields=cont_channels)
        #normalizer_state = 'decomp_normalizer'
        #normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
        #normalizer.load_params(normalizer_state)

        data_gen = utils_SICH.load_data(data_loader, discretizer)
                                       #(data_loader, discretizer,
                                       #                  args.batch_size, shuffle=True, return_names=True)
        #for i in range(721):
        #    print(data_gen[0].shape)
            #if (data_gen[0][i].shape[0] != 24):
            #    print('no 24')
        #val_data_gen = utils_SICH.BatchGenDeepSupervision(val_data_loader, discretizer,
        #                                            normalizer, args.batch_size, shuffle=False, return_names=True)

        '''Model structure'''
        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))

        

        '''Train phase'''
        print('Start training ... ')
        
        SICH_data = data_gen[0]
        flag = np.array(data_gen[1])
        
        sensitivity = np.zeros(10)
        specificity = np.zeros(10)
        accuracy = np.zeros(10)
        auroc = np.zeros(10)
        
        count = 0
        
        kf = KFold(n_splits = 10)
        for train, test in kf.split(SICH_data):
            
            model = StageNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
            model.apply(reset_weights)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            x_train = SICH_data[train]
            y_train = flag[train]
            x_test = SICH_data[test]
            y_test = flag[test]
            
            
            #y_train.reshape((length_train, ))
            #y_test.reshape((length_test, ))
            
            scaler = StandardScaler()
            to_normalize = x_train.reshape(-1,27)
            print(to_normalize.shape)
            x_train_normalized = scaler.fit_transform(to_normalize)
            for i in range(x_test.shape[0]):
                x_test[i] = scaler.transform(x_test[i])
            for i in range(x_train.shape[0]):
                x_train[i] = scaler.transform(x_train[i])
                
            x_train = torch.tensor(x_train, dtype = torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            
            torch_dataset = data.TensorDataset(x_train, y_train)
            
            loader = data.DataLoader(torch_dataset, batch_size = args.batch_size, shuffle = True)
            
            for epoch in range(200):
                for batch_x, batch_y in loader:
                    model.train()
                    optimizer.zero_grad()
                    
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    #print(batch_x[0])
                    #print(batch_y[0])
                    batch_time = torch.ones((batch_x.size(0), batch_x.size(1)), dtype=torch.float32).to(device)
                    
                    batch_output, _ = model(batch_x, batch_time, device)
                    
                    loss = batch_y * torch.log(batch_output + 1e-7) + (1 - batch_y) * torch.log(1 - batch_output + 1e-7)
                    loss = loss / batch_x.shape[0]
                    loss = torch.neg(torch.sum(loss))
                    #print('training loss is {}'.format(loss))
                    
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    model.eval()
                    
                    x_test = torch.tensor(x_test, dtype = torch.float32).to(device)
                    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
                    
                    #x_test = x_test.to(device)
                    #y_test = y_test.to(device)
                    
                    test_time = torch.ones((x_test.size(0), x_test.size(1)), dtype=torch.float32).to(device)
                    
                    test_output,_ = model(x_test, test_time, device)
                    #test_pred = np.array(test_output.cpu())
                    test_pred = test_output.data.cpu().numpy()
                    #print(test_pred)
                    
                    test_true = y_test.data.cpu().numpy()
                    test_ret = metrics.print_metrics_binary(test_true, test_pred)
                    
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                    train_time = torch.ones((x_train.size(0), x_train.size(1)), dtype=torch.float32).to(device)
                    train_output,_ = model(x_train, train_time, device)
                    train_pred = train_output.data.cpu().numpy()
                    train_ret = metrics.print_metrics_binary(y_train.data.cpu().numpy(), train_pred, verbose = False)
                    
                    
                
                    if (test_ret["auroc"] > auroc[count]):
                        sensitivity[count] = test_ret["rec1"]
                        specificity[count] = test_ret["rec0"]
                        accuracy[count] = test_ret["acc"]
                        auroc[count] = test_ret["auroc"]
                
            count += 1
            model.apply(reset_weights)
            
        print('sensitivity')
        arr_mean = np.mean(sensitivity)
        arr_var = np.var(sensitivity)
        #arr_std = np.std
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('specificity')
        arr_mean = np.mean(specificity)
        arr_var = np.var(specificity)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('accuracy')
        arr_mean = np.mean(accuracy)
        arr_var = np.var(accuracy)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('AUC of ROC')
        arr_mean = np.mean(auroc)
        arr_var = np.var(auroc)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
        print(auroc)
# 测试上方代码

'''
        train_loss = []
        val_loss = []
        batch_loss = []
        dead_batch_loss = []
        dead_train_loss = []

        max_auprc = 0

        file_name = './saved_weights/'+args.file_name
        for each_chunk in range(args.epochs):
        #for each_chunk in range(20):
            cur_batch_loss = []
            cur_dead_loss = []
            model.train()
            
            train_true = []
            train_pred = []
            
            for each_batch in range(train_data_gen.steps):
                batch_data = next(train_data_gen)
                batch_name = batch_data['names']
                batch_data = batch_data['data']

                batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
                tmp = torch.zeros(batch_x.size(0),17, dtype=torch.float32).to(device)
                batch_interval = torch.zeros((batch_x.size(0),batch_x.size(1),17), dtype=torch.float32).to(device)
                
                for i in range(batch_x.size(1)):
                    cur_ind = batch_x[:,i,-17:]
                    tmp+=(cur_ind == 0).float()
                    batch_interval[:, i, :] = cur_ind * tmp
                    tmp[cur_ind==1] = 0        
                
                if batch_mask.size()[1] > 400:
                    batch_x = batch_x[:, :400, :]
                    batch_mask = batch_mask[:, :400, :]
                    batch_y = batch_y[:, :400, :]
                    batch_interval = batch_interval[:, :400, :]

                batch_x = torch.cat((batch_x, batch_interval), dim=-1)
                batch_time = torch.ones((batch_x.size(0), batch_x.size(1)), dtype=torch.float32).to(device)

                optimizer.zero_grad()
                cur_output, _ = model(batch_x, batch_time, device)
                masked_output = cur_output * batch_mask 
                loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
                # loss = batch_y * 0.25*pow(1-masked_output,2) * torch.log(masked_output + 1e-7) + (1 - batch_y) * 0.75*pow(masked_output,2)*torch.log(1 - masked_output + 1e-7)
                  # focal loss, adopted by zixuan

                loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
                loss = torch.neg(torch.sum(loss))
                cur_batch_loss.append(loss.cpu().detach().numpy())

                # print(batch_y)

                # if batch_y == 1:
                #     dead_loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
                #     dead_loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
                #     dead_loss = torch.neg(torch.sum(loss))
                #     cur_dead_loss.append(dead_loss.cpu().detach().numpy())


                loss.backward()
                optimizer.step()

                
                if each_batch % 50 == 0:
                    print('Chunk %d, Batch %d: Loss = %.4f'%(each_chunk, each_batch, cur_batch_loss[-1]))
                    
                    
                # added by Jingyuan to check the '0's and '1's in training set
                for m, t, p in zip(batch_mask.cpu().numpy().flatten(), batch_y.cpu().numpy().flatten(), cur_output.cpu().detach().numpy().flatten()):
                    if np.equal(m, 1):
                        train_true.append(t)
                        train_pred.append(p)
            test_ret = metrics.print_metrics_binary(train_true, train_pred)

            batch_loss.append(cur_batch_loss)
            train_loss.append(np.mean(np.array(cur_batch_loss)))
            # dead_batch_loss.append(cur_dead_loss)
            # dead_train_loss.append(np.mean(np.array(cur_dead_loss)))
            
            print("\n==>Predicting on validation")
            with torch.no_grad():
                model.eval()
                cur_val_loss = []
                valid_true = []
                valid_pred = []
                for each_batch in range(val_data_gen.steps):
                    valid_data = next(val_data_gen)
                    valid_name = valid_data['names']
                    valid_data = valid_data['data']
                    
                    valid_x = torch.tensor(valid_data[0][0], dtype=torch.float32).to(device)
                    valid_mask = torch.tensor(valid_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                    valid_y = torch.tensor(valid_data[1], dtype=torch.float32).to(device)
                    tmp = torch.zeros(valid_x.size(0),17, dtype=torch.float32).to(device)
                    valid_interval = torch.zeros((valid_x.size(0),valid_x.size(1),17), dtype=torch.float32).to(device)
                    
                    for i in range(valid_x.size(1)):
                        cur_ind = valid_x[:,i,-17:]
                        tmp+=(cur_ind == 0).float()
                        valid_interval[:, i, :] = cur_ind * tmp
                        tmp[cur_ind==1] = 0  
                    
                    if valid_mask.size()[1] > 400:
                        valid_x = valid_x[:, :400, :]
                        valid_mask = valid_mask[:, :400, :]
                        valid_y = valid_y[:, :400, :]
                        valid_interval = valid_interval[:, :400, :]
                    
                    valid_x = torch.cat((valid_x, valid_interval), dim=-1)
                    valid_time = torch.ones((valid_x.size(0), valid_x.size(1)), dtype=torch.float32).to(device)
                    
                    valid_output, valid_dis = model(valid_x, valid_time, device)
                    masked_valid_output = valid_output * valid_mask

                    valid_loss = valid_y * torch.log(masked_valid_output + 1e-7) + (1 - valid_y) * torch.log(1 - masked_valid_output + 1e-7)
                    valid_loss = torch.sum(valid_loss, dim=1) / torch.sum(valid_mask, dim=1)
                    valid_loss = torch.neg(torch.sum(valid_loss))
                    cur_val_loss.append(valid_loss.cpu().detach().numpy())

                    for m, t, p in zip(valid_mask.cpu().numpy().flatten(), valid_y.cpu().numpy().flatten(), valid_output.cpu().detach().numpy().flatten()):
                        if np.equal(m, 1):
                            valid_true.append(t)
                            valid_pred.append(p)

                val_loss.append(np.mean(np.array(cur_val_loss)))
                print('Valid loss = %.4f'%(val_loss[-1]))
                print('\n')
                valid_pred = np.array(valid_pred)
                valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
                ret = metrics.print_metrics_binary(valid_true, valid_pred)
                print()

                cur_auprc = ret['auprc']
                if cur_auprc > max_auprc:
                    max_auprc = cur_auprc
                    state = {
                        'net': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'chunk': each_chunk
                    }
                    torch.save(state, file_name)
                    print('\n------------ Save best model ------------\n')


        # epolen = len(train_loss)
        # epo = [0] * epolen
        # for i in range(0,epolen):
        #     epo[i] = i
        #plt.plot( batch_loss, label='Training loss')            # my
        plt.plot( train_loss, label='Training loss')
        # plt.plot(dead_train_loss, label='dead loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig("D:\StageNet\Jingyuan\loss_2")
'''
        ## Evaluata Phase
'''
        print('Testing model ... ')
        checkpoint = torch.load(file_name)
        save_chunk = checkpoint['chunk']
        print("last saved model is in chunk {}".format(save_chunk))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()

        test_data_loader = common_utils_SICH.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data_path, 'test'),
                                                                        listfile=os.path.join(args.data_path, 'test_listfile.csv'), small_part=args.small_part)
        test_data_gen = utils_SICH.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                    normalizer, args.batch_size,
                                                    shuffle=False, return_names=True)

        with torch.no_grad():
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)
        
            cur_test_loss = []
            test_true = []
            test_pred = []
            
            for each_batch in range(test_data_gen.steps):
                test_data = next(test_data_gen)
                test_name = test_data['names']
                test_data = test_data['data']

                test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device)
                test_mask = torch.tensor(test_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)
                tmp = torch.zeros(test_x.size(0),17, dtype=torch.float32).to(device)
                test_interval = torch.zeros((test_x.size(0),test_x.size(1),17), dtype=torch.float32).to(device)

                for i in range(test_x.size(1)):
                    cur_ind = test_x[:,i,-17:]
                    tmp+=(cur_ind == 0).float()
                    test_interval[:, i, :] = cur_ind * tmp
                    tmp[cur_ind==1] = 0  
                
                if test_mask.size()[1] > 400:
                    test_x = test_x[:, :400, :]
                    test_mask = test_mask[:, :400, :]
                    test_y = test_y[:, :400, :]
                    test_interval = test_interval[:, :400, :]
                
                test_x = torch.cat((test_x, test_interval), dim=-1)
                test_time = torch.ones((test_x.size(0), test_x.size(1)), dtype=torch.float32).to(device)
                
                test_output, test_dis = model(test_x, test_time, device)
                masked_test_output = test_output * test_mask

                test_loss = test_y * torch.log(masked_test_output + 1e-7) + (1 - test_y) * torch.log(1 - masked_test_output + 1e-7)
                test_loss = torch.sum(test_loss, dim=1) / torch.sum(test_mask, dim=1)
                test_loss = torch.neg(torch.sum(test_loss))
                cur_test_loss.append(test_loss.cpu().detach().numpy()) 
                
                for m, t, p in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
                    if np.equal(m, 1):
                        test_true.append(t)
                        test_pred.append(p)
            
            print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
            print('\n')
            test_pred = np.array(test_pred)
            test_pred = np.stack([1 - test_pred, test_pred], axis=1)
            test_ret = metrics.print_metrics_binary(test_true, test_pred)
'''