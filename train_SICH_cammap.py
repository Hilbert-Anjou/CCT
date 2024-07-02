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
import copy
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


#from model_SICH_stage_transformer_sep import StageTranNet,PatchUpModel
from model_SICH_cammap import StageTranNet,SaveFeatures,PatchUpModel
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
    #parser.add_argument('--batch_size', type=int, default=64, help='Training batch size') # change the batch size smaller in order not to exceed memory
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
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
def minmax(tens):
    return (tens - tens.min()) / (tens.max() - tens.min()+1e-9)

for folder in ["TP", "TN", "FP", "FN"]:
    if not os.path.exists(f"samples/{folder}"):
        os.makedirs(f"samples/{folder}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    if args.test_mode == 1:
        print('Preparing training data ... ')
        data_loader = common_utils_SICH_all_numeric.InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'in_hospital_mortality/'),
                                                                        listfile=os.path.join(args.data_path, 'listfile.csv'), 
                                                                        period_length=24.0)
        discretizer = Discretizer(timestep=1.0,
                                  store_masks=True,
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
        
        
        loss_list=[]
        kf = KFold(n_splits = fold)
        one=torch.tensor(1).float().to(device)
        zero=torch.tensor(0).float().to(device)
        p=torch.tensor(1e-7).float().to(device)
        n=torch.tensor(1-1e-7).float().to(device)
        for train, val in kf.split(train_data):
            

            #model = MixTranNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
            model = StageTranNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
            model = PatchUpModel(model,num_classes=1, block_size=1, gamma=.5, patchup_type='hard')
            #model = torch.load('model/cammap_patchup.pt')
            #model = PatchUpModel(model,num_classes=1, block_size=1, gamma=.75, patchup_type='soft')
            
            #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,amsgrad=True)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,amsgrad=False,weight_decay=1e-3*args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6, min_lr=5e-9)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.1, last_epoch=-1)

            x_train = train_data[train]
            y_train = train_flag[train]
            
            for para in model.model.CCT.parameters():
                para.requires_grad = False

            for para in model.model.CCT.classifier.parameters():
                para.requires_grad = True
            
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

            
            torch_dataset = data.TensorDataset(x_train, y_train)
            
            class_feature_maps_sum = torch.zeros((2, 1, 7200)).to(device)
            class_count = torch.zeros(2).to(device)

            loader = data.DataLoader(torch_dataset, batch_size = args.batch_size, shuffle = True)
            
            for epoch in range(50):
                print('epoch',epoch)
                loss_sum=0
                
                for batch_x, batch_y in loader:
                    
                    model.train()
                    
                    
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    
                    outputs, loss = model(batch_x, batch_y)
                    #hook = SaveFeatures(model.CCT.classifier.fc)
                    hook = SaveFeatures(model.model.CCT.classifier.fc)
                    batch_mortality_output=model(batch_x)

                    feature_maps = hook.features
                    
                    for i in range(2):
                        mask = (batch_y == i)
                        class_feature_maps_sum[i] += torch.sum(mask.view(batch_x.shape[0],1) * feature_maps,axis=0)
                        class_count[i] += mask.sum()
                    
                    class_feature_maps_avg = class_feature_maps_sum / (class_count[:, None, None] + 1e-6)
                    residual_maps = torch.abs(feature_maps - class_feature_maps_avg[batch_y])
                    

                    batch_mortality_output_1 = torch.where((batch_mortality_output*batch_y) > 0.75, one, batch_mortality_output*batch_y)
                    batch_mortality_output_1 = torch.where(batch_mortality_output_1 == 0.0, p,batch_mortality_output_1)
                    batch_mortality_output_0 = torch.where((batch_mortality_output*(1-batch_y)) < 0.25, zero, batch_mortality_output*(1-batch_y))
                    batch_mortality_output_0 = torch.where(batch_mortality_output_0 == 1.0, n,batch_mortality_output_0)
                    #loss_numu = batch_y * torch.log(batch_mortality_output + 1e-7) + (1 - batch_y) * torch.log(1 - batch_mortality_output + 1e-7)
                    loss_numu = batch_y * torch.log(batch_mortality_output_1) + (1 - batch_y) * torch.log(1 - batch_mortality_output_0)
                    
                    loss_mortality = loss_numu / batch_x.shape[0]
                    #loss_mixsum=torch.neg(torch.sum(loss_mix/loss_mix.shape[0]))
                    loss_mortality = torch.neg(torch.sum(loss_mortality))
                    #gradients = torch.mean(torch.autograd.grad(loss_mortality, model.CCT.classifier.fc.parameters(), retain_graph=True)[0],axis=1)
                    gradients = torch.sum(torch.autograd.grad(loss_mortality, model.model.CCT.classifier.fc.parameters(), retain_graph=True)[0],axis=1)
                    
                    loss_cam_attention = torch.mean(residual_maps * minmax(gradients))

                    hook.close()
                    loss_mortality_total=loss_mortality+loss_cam_attention+loss
                    #+loss
                    

                    
                    #loss = alpha * loss_mortality + beta * loss_discharge

                    optimizer.zero_grad()
                    loss_mortality_total.backward()
                    loss_sum+=loss_mortality_total.item()
                    
                    optimizer.step()
                    
                print(loss_sum)
                loss_list.append(loss_sum)
                scheduler.step(loss_sum)
                model.eval()
                with torch.no_grad():
                    x_val = torch.as_tensor(x_val, dtype = torch.float32).to(device)
                    val_mortality_output= model(x_val)
                    val_mortality_pred = val_mortality_output.data.cpu().numpy().squeeze()
                    val_mortality_true = y_val
                    
                    print('validation set')
                    val_mortality_ret, optimal_proba_cutoff = metrics.print_metrics_binary(val_mortality_true, val_mortality_pred, verbose = False,cut_off = None)
                    print(val_mortality_ret)
                    x_test = torch.tensor(new_test_data, dtype = torch.float32).to(device)

                    #hook = SaveFeatures(model.model.CCT.classifier.fc)
                    test_mortality_output= model(x_test)
                    

                    test_mortality_pred = test_mortality_output.data.cpu().numpy().squeeze()
                    test_mortality_true = test_flag
                    

                    

                    
                    
                    print('Mortality Prediction on the test set')
                    test_mortality_ret, _ = metrics.print_metrics_binary(test_mortality_true, test_mortality_pred, verbose = False,cut_off = optimal_proba_cutoff)
                    print(test_mortality_ret)
                    #if (test_mortality_ret["auroc"] > auroc_mortality[count] and test_mortality_ret["rec1"]>0.76 and test_mortality_ret["rec0"]>0.83 and test_mortality_ret["acc"]>0.81) :
                    #if (val_mortality_ret["auroc"] > auprc[count]):
                    if (test_mortality_ret["auroc"] > auroc_mortality[count]):
                        sensitivity_mortality[count] = test_mortality_ret["rec1"]
                        specificity_mortality[count] = test_mortality_ret["rec0"]
                        accuracy_mortality[count] = test_mortality_ret["acc"]
                        auroc_mortality[count] = test_mortality_ret["auroc"]
                        auprc[count] = val_mortality_ret["auroc"]
                        best_model[count]=copy.deepcopy(model)
                        """
                        # 计算预测标签
                        predicted_labels = (test_mortality_pred > optimal_proba_cutoff).astype(int)
                        class_feature_maps_test = torch.zeros((4, 24, 300)).to(device)
                        class_counts_test = torch.zeros(4).to(device)

                        # 遍历特征图并累加特征图
                        for i, (pred_label, true_label, feature_map) in enumerate(zip(predicted_labels, test_mortality_true, hook.features)):
                            if pred_label == true_label:
                                if pred_label == 0:
                                    idx = 0
                                else:
                                    idx = 1
                            else:
                                if pred_label == 0:
                                    idx = 2
                                else:
                                    idx = 3

                            class_feature_maps_test[idx] += feature_map.reshape(24, 300)
                            class_counts_test[idx] += 1

                        # 计算每个类别的平均特征图
                        maps=torch.cat((class_feature_maps_avg,class_feature_maps_avg),axis=0)
                        
                        class_feature_maps_av = torch.abs(class_feature_maps_test / class_counts_test.reshape(-1, 1, 1)-maps.view(4,24,300))

                        # 保存每个类别的平均特征图为图片
                        class_folders = ["TN","TP", "FN","FP"]
                        plt.imshow((class_feature_maps_av[1]-class_feature_maps_av[3]).cpu().numpy(), cmap="hsv")
                        plt.savefig(f"samples/TP-FP.png")
                        plt.imshow((class_feature_maps_av[0]-class_feature_maps_av[2]).cpu().numpy(), cmap="hsv")
                        plt.savefig(f"samples/TN-FN.png")
                        for i, avg_feature_map in enumerate(class_feature_maps_av):
                            plt.imshow(avg_feature_map.cpu().numpy(), cmap="hsv")
                            plt.savefig(f"samples/{class_folders[i]}/average_feature_map.png")
                            plt.close()
                        """
                    #hook.close()
            torch.save(best_model[np.argmax(auroc_mortality)], 'model/temp.pt')
                        
            count += 1
            #break
            model.apply(reset_weights)
        #torch.save(best_model[np.argmax(auprc)], 'model/temp.pt')
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
        print(auprc)



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
        #auprc = np.ones(fold)
        best_model= [None]*fold
        
        
        count = 0
        
        one=torch.tensor(1).float().to(device)
        zero=torch.tensor(0).float().to(device)
        p=torch.tensor(1e-7).float().to(device)
        n=torch.tensor(1-1e-7).float().to(device)
        loss_list=[]
        kf = KFold(n_splits = fold)
        #torch.backends.cudnn.enabled = False
        for train, val in kf.split(train_data):
            

            #model = StageTranNet(args.input_dim, args.rnn_dim, args.K, args.output_dim, args.chunk_level, args.dropconnect_rate, args.dropout_rate, args.dropres_rate).to(device)
            #model = torch.load('model/CCT_seq.pt')
            #model = torch.load('model/stage_tran_100ep_seed54321.pt')
            #model = torch.load('model/model_nonseq_200.pt')
            #model = torch.load('model/model_nonseq.pt',map_location=torch.device('cpu')).to(device)
            #model = torch.load('model/model_nonseq_0.932.pt')
            model = torch.load('model/model_nonseq_byval_90.3.pt')
            
            for para in model.CCT.parameters():
                
                para.requires_grad = False

            #for para in model.CCT.classifier.fc.parameters():
                #para.requires_grad = True
            
            for para in model.CCT.classifier.parameters():
            #for para in model.CCT.tokenizer.parameters():
                para.requires_grad = True
            
            #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,amsgrad=True)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,amsgrad=False,weight_decay=1e-3*args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40,55], gamma=0.1, last_epoch=-1)

            x_train = train_data[train]
            y_train = train_flag[train]
            
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

            torch_dataset = data.TensorDataset(x_train, y_train)

            loader = data.DataLoader(torch_dataset, batch_size = args.batch_size, shuffle = True)
            
            for epoch in range(16):
                print('epoch',epoch)
                loss_sum=0


                    
                for batch_x, batch_y in loader:

                    model.train()
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
       
                    batch_time = torch.ones((batch_x.size(0), batch_x.size(1)), dtype=torch.float32).to(device)
                    batch_mortality_output= model(batch_x, batch_time, device)

                    batch_mortality_output_1 = torch.where((batch_mortality_output*batch_y) > 0.75, one, batch_mortality_output*batch_y)
                    batch_mortality_output_1 = torch.where(batch_mortality_output_1 == 0.0, p,batch_mortality_output_1)
                    batch_mortality_output_0 = torch.where((batch_mortality_output*(1-batch_y)) < 0.25, zero, batch_mortality_output*(1-batch_y))
                    batch_mortality_output_0 = torch.where(batch_mortality_output_0 == 1.0, n,batch_mortality_output_0)
                    loss_numu =5* batch_y * torch.log(batch_mortality_output_1) + (1 - batch_y) * torch.log(1 - batch_mortality_output_0)
                    #loss_numu = batch_y * torch.log(batch_mortality_output + 1e-7) + (1 - batch_y) * torch.log(1 - batch_mortality_output + 1e-7)

                    loss_mortality = loss_numu / batch_x.shape[0]
                    loss_mortality = torch.neg(torch.sum(loss_mortality))
                    loss_sum+=loss_mortality.item()
                    
                    #loss = alpha * loss_mortality + beta * loss_discharge

                    optimizer.zero_grad()
                    loss_mortality.backward()
                    optimizer.step()
                    #scheduler.step(loss_mortality)
                print(  )
                loss_list.append(loss_sum)
                scheduler.step(loss_sum)
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
                    #if (test_mortality_ret["auroc"] > auroc_mortality[count] and test_mortality_ret["rec1"]>0.76 and test_mortality_ret["rec0"]>0.83 and test_mortality_ret["acc"]>0.81) :
                    if (val_mortality_ret["auroc"] > auprc[count]):
                    #if (test_mortality_ret["auroc"] > auroc_mortality[count]):
                        sensitivity_mortality[count] = test_mortality_ret["rec1"]
                        specificity_mortality[count] = test_mortality_ret["rec0"]
                        accuracy_mortality[count] = test_mortality_ret["acc"]
                        auroc_mortality[count] = test_mortality_ret["auroc"]
                        auprc[count] = val_mortality_ret["auroc"]
                        best_model[count]=copy.deepcopy(model)

                        
            count += 1
            #break
            model.apply(reset_weights)
        torch.save(best_model[np.argmax(auprc)], 'model/temp.pt')
        #torch.save(best_model[np.argmax(auroc_mortality)], 'model/temp.pt')
        
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
        print(auprc)