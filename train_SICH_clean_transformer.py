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

from model_SICH_clean_transformer import StageTranNet
#from model_SICH_stage_transformer import StageTranNet
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
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size') # change the batch size smaller in order not to exceed memory
    #parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learing rate')#5e-5
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')#i+
    # parser.add_argument('--load_state', type=str, default="",
    #                     help='state file path')#i+
    parser.add_argument('--data_dim', type = int, default = 825, help='Dimension of visit record data before autoencoder transform')
    parser.add_argument('--input_dim', type=int, default=280, help='Dimension of visit record data after autoencoder transform')
    parser.add_argument('--rnn_dim', type=int, default=512, help='Dimension of hidden units in RNN')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dropconnect_rate', type=float, default=0.5, help='Dropout rate in RNN')
    parser.add_argument('--dropres_rate', type=float, default=0.3, help='Dropout rate in residue connection')
    parser.add_argument('--K', type=int, default=24, help='Value of hyper-parameter K')
    parser.add_argument('--chunk_level', type=int, default=2, help='Value of hyper-parameter K')

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
        auprc = np.zeros(fold)
        best_model= [None]*fold
        
        
        count = 0
        
        alpha = 1.0
        beta = 0.5
        CEsingle=50
        IBratio=0.5
        kf = KFold(n_splits = fold)
        
        for train, val in kf.split(train_data):
            

            #model = StageTranNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
            #model = torch.load('model/multitrans_100ep.pt')
            model = torch.load('model/vanilla_200ep.pt')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70,90], gamma=0.5, last_epoch=-1)

            x_train = train_data[train]
            y_train = train_flag[train]
            
            #discharge_train = discharge_flag[train]
            
            x_val = train_data[val]
            y_val = train_flag[val]
            #discharge_test = discharge_flag[test]
            
            #y_train.reshape((length_train, ))
            #y_test.reshape((length_test, ))
            
            scaler = StandardScaler()
            to_normalize = x_train.reshape(-1,args.data_dim)
            #print(to_normalize.shape)
            x_train_normalized = scaler.fit_transform(to_normalize)
            for i in range(x_val.shape[0]):
                x_val[i] = scaler.transform(x_val[i])
            for i in range(x_train.shape[0]):
                x_train[i] = scaler.transform(x_train[i])
            for i in range(test_data.shape[0]):
                new_test_data[i] = scaler.transform(test_data[i])
                
            x_train = torch.tensor(x_train, dtype = torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            #print('class0: ',y_train.shape[0]-torch.sum(y_train),'          class1:',torch.sum(y_train))
            
            #class0=y_train.shape[0]-torch.sum(y_train) #for IBloss
            #class1=torch.sum(y_train)
            
            #discharge_train = torch.tensor(discharge_train, dtype=torch.long)
            
            torch_dataset = data.TensorDataset(x_train, y_train)
            #torch_dataset = data.TensorDataset(x_train, y_train, discharge_train)
            
            loader = data.DataLoader(torch_dataset, batch_size = args.batch_size, shuffle = True)
            
            for epoch in range(200):
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
                    
                    
                    #loss_discharge = batch_discharge * torch.log(batch_discharge_output + 1e-7) + (1 - batch_discharge) * torch.log(1 - batch_discharge_output + 1e-7)
                    #loss_discharge = loss_discharge / batch_x.shape[0]
                    #loss_discharge = torch.neg(torch.sum(loss_discharge))
                    #print('training loss is {}'.format(loss))
                    
                    #loss = alpha * loss_mortality + beta * loss_discharge
                    """
                    if epoch <=CEsingle:
                        loss = loss_mortality
                    else:
                        lambda_k=0.5/((batch_y*class1+(1-batch_y)*class0)*(1/class0+1/class1))  #alpha
                        loss_denominator = torch.norm(batch_mortality_output-batch_y,p=1)*torch.norm(H,p=1)
                        loss_IB= torch.sum(lambda_k*torch.neg(loss_numu/loss_denominator))/args.batch_size
                        loss = loss_mortality+IBratio*loss_IB
                    """
                    optimizer.zero_grad()
                    loss_mortality.backward()
                    optimizer.step()
                    #scheduler.step(loss_mortality)
                model.eval()
                with torch.no_grad():
                    
                    """
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                    train_time = torch.ones((x_train.size(0), x_train.size(1)), dtype=torch.float32).to(device)
                    
                    #train_mortality_output, _ = model(x_train, train_time, device)
                    train_mortality_output_se, _ = model_se(x_train, train_time, device)
                    train_mortality_output_sp, _ = model_sp(x_train, train_time, device)
                    train_mortality_output_se = train_mortality_output_se.to(device)
                    train_mortality_output_sp = train_mortality_output_sp.to(device)
                    train_mortality_output=model(train_mortality_output_se,train_mortality_output_sp)

                    train_mortality_pred = train_mortality_output.data.cpu().numpy()
                    train_ret = metrics.print_metrics_binary(y_train.data.cpu().numpy(), train_mortality_pred, verbose = True, cut_off = None)
                    """
                    
                    x_val = torch.tensor(x_val, dtype = torch.float32).to(device)
                    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
                    #discharge_test = torch.tensor(discharge_test, dtype=torch.long).to(device)
                    
                   
                    
                    val_time = torch.ones((x_val.size(0), x_val.size(1)), dtype=torch.float32).to(device)
                    
                    val_mortality_output= model(x_val, val_time, device)
                    


                    #val_mortality_output, test_discharge_output, _ = model(x_test, test_time, device)
                    #val_pred = np.array(test_output.cpu())
                    
                    val_mortality_pred = val_mortality_output.data.cpu().numpy()
                    val_mortality_pred = val_mortality_pred.squeeze()
                    #val_discharge_pred = test_discharge_output.data.cpu().numpy()
                    
                    
                    
                    val_mortality_true = y_val.data.cpu().numpy()
                    #val_discharge_true = discharge_test.data.cpu().numpy()
                    print('validation set')
                    val_mortality_ret, optimal_proba_cutoff = metrics.print_metrics_binary(val_mortality_true, val_mortality_pred, verbose = True,
                                                                                           cut_off = None)
                    #print('Discharge Prediction')
                    #val_discharge_ret = metrics.print_metrics_binary(val_discharge_true, val_discharge_pred, verbose = True)
                    
                    #x_train = x_train.to(device)
                    #y_train = y_train.to(device)
                    #train_time = torch.ones((x_train.size(0), x_train.size(1)), dtype=torch.float32).to(device)
                    #train_mortality_output, train_discharge_output, _ = model(x_train, train_time, device)
                    #train_mortality_pred = train_mortality_output.data.cpu().numpy()
                    #train_discharge_pred = train_discharge_output.data.cpu().numpy()
                    #train_ret = metrics.print_metrics_binary(y_train.data.cpu().numpy(), train_mortality_pred, verbose = False)
                    #train_ret = metrics.print_metrics_binary(discharge_train.data.cpu().numpy(), train_discharge_pred, verbose = False)
                    
                    x_test = torch.tensor(new_test_data, dtype = torch.float32).to(device)
                    y_test = torch.tensor(test_flag, dtype=torch.long).to(device)
                    test_time = torch.ones((x_test.size(0), x_test.size(1)), dtype=torch.float32).to(device)
                    
                    test_mortality_output= model(x_test, test_time, device)
                    

                    test_mortality_pred = test_mortality_output.data.cpu().numpy()
                    test_mortality_pred = test_mortality_pred.squeeze()
                    test_mortality_true = y_test.data.cpu().numpy()
                    
                    print('Mortality Prediction on the test set')
                    test_mortality_ret, _ = metrics.print_metrics_binary(test_mortality_true, test_mortality_pred, verbose = True,
                                                                         cut_off = optimal_proba_cutoff)
                
                    if (test_mortality_ret["auroc"] > auroc_mortality[count]):
                    #if (test_mortality_ret["rec1"] > sensitivity_mortality[count] and test_mortality_ret["rec0"]>specificity_mortality[count]):
                    #if (test_mortality_ret["rec1"] > 0.7 and test_mortality_ret["rec0"]>specificity_mortality[count]):
                        sensitivity_mortality[count] = test_mortality_ret["rec1"]
                        specificity_mortality[count] = test_mortality_ret["rec0"]
                        accuracy_mortality[count] = test_mortality_ret["acc"]
                        auroc_mortality[count] = test_mortality_ret["auroc"]
                        auprc[count]=test_mortality_ret["auprc"]
                        best_model[count]=model
                        
                    """
                    if epoch % CEsingle==0:
                        with open("epoches"+str(CEsingle)+".txt","a") as f:
                            
                            f.write('for fold'+str(count)+'for mortality prediction:\n')
                            f.write('sensitivity\n')
                            arr_mean = np.mean(sensitivity_mortality)
                            arr_var = np.var(sensitivity_mortality)
                            #arr_std = np.std
                            f.write('mean is {} and variance is {}'.format(arr_mean, arr_var))
                                
                            f.write('\nspecificity')
                            arr_mean = np.mean(specificity_mortality)
                            arr_var = np.var(specificity_mortality)
                            f.write('mean is {} and variance is {}'.format(arr_mean, arr_var))
                                
                            f.write('\naccuracy')
                            arr_mean = np.mean(accuracy_mortality)
                            arr_var = np.var(accuracy_mortality)
                            f.write('\nmean is {} and variance is {}'.format(arr_mean, arr_var))
                                
                            f.write('\nAUC of ROC')
                            arr_mean = np.mean(auroc_mortality)
                            arr_var = np.var(auroc_mortality)
                            f.write('mean is {} and variance is {}'.format(arr_mean, arr_var))
                            #f.write(auroc_mortality)
                        """
                    #if (test_discharge_ret["auroc"] > auroc_discharge[count]):
                    #    sensitivity_discharge[count] = test_discharge_ret["rec1"]
                    #    specificity_discharge[count] = test_discharge_ret["rec0"]
                    #    accuracy_discharge[count] = test_discharge_ret["acc"]
                    #    auroc_discharge[count] = test_discharge_ret["auroc"]
                    torch.save(best_model[np.argmax(auroc_mortality)], 'model/temp.pt')
            count += 1
            model.apply(reset_weights)
        torch.save(best_model[np.argmax(auroc_mortality)], 'model/temp.pt')
        print('for mortality prediction')
        print('sensitivity')
        arr_mean = np.mean(sensitivity_mortality)
        arr_var = np.var(sensitivity_mortality)
        #arr_std = np.std
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('specificity')
        arr_mean = np.mean(specificity_mortality)
        arr_var = np.var(specificity_mortality)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('accuracy')
        arr_mean = np.mean(accuracy_mortality)
        arr_var = np.var(accuracy_mortality)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('AUC of ROC')
        arr_mean = np.mean(auroc_mortality)
        arr_var = np.var(auroc_mortality)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
        print(auroc_mortality)
        
        print('AUPRC')
        arr_mean = np.mean(auprc)
        arr_var = np.var(auprc)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
        print(auprc)
        """
        print('\nfor discharge prediction')
        print('sensitivity')
        arr_mean = np.mean(sensitivity_discharge)
        arr_var = np.var(sensitivity_discharge)
        #arr_std = np.std
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('specificity')
        arr_mean = np.mean(specificity_discharge)
        arr_var = np.var(specificity_discharge)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('accuracy')
        arr_mean = np.mean(accuracy_discharge)
        arr_var = np.var(accuracy_discharge)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
            
        print('AUC of ROC')
        arr_mean = np.mean(auroc_discharge)
        arr_var = np.var(auroc_discharge)
        print('mean is {} and variance is {}'.format(arr_mean, arr_var))
        print(auroc_mortality)
        """
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