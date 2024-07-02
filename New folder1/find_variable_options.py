# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:42:49 2022

@author: JyGuo
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils import utils_SICH_all_numeric
from utils.readers import DecompensationReader
from utils.preprocessing_SICH_all_numeric import Discretizer, Normalizer
from utils import metrics
from utils import common_utils_SICH_all_numeric

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
    parser.add_argument('--lr', type=float, default=0.00001, help='Learing rate')
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

def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    var_map = dataframe_from_csv(path=os.path.join(args.data_path, 'listfile.csv'), index_col=None)
    
    file_path = os.path.join(args.data_path, 'in_hospital_mortality/')
    
    GCS_eye_opening = []
    GCS_motor_response = [] 
    GCS_scale_total = []
    GCS_verbal_response = []
    
    #Glascow coma scale eye opening
    #Glascow coma scale motor response
    #Glascow coma scale total
    #Glascow coma scale verbal response
    count = 0
    for index, row in var_map.iterrows():
        each_file_path = os.path.join(file_path, row['stay'])
        tmp_file = dataframe_from_csv(path=each_file_path, index_col=None)
            #for line in tsfile:
            #    mas = line.strip().split(',')
            #    ret.append(np.array(mas))
        #return (np.stack(ret), header)
        
        #list(events['ITEMID'].unique())
        GCS_eye_opening += list(tmp_file['Glascow coma scale eye opening'].unique())
        GCS_motor_response += list(tmp_file['Glascow coma scale motor response'].unique())
        GCS_scale_total += list(tmp_file['Glascow coma scale total'].unique())
        GCS_verbal_response += list(tmp_file['Glascow coma scale verbal response'].unique())
        
        GCS_eye_opening = list(set(GCS_eye_opening))
        GCS_motor_response = list(set(GCS_motor_response))
        GCS_scale_total = list(set(GCS_scale_total))
        GCS_verbal_response = list(set(GCS_verbal_response))
        
        if count == 0:
            print(each_file_path)
            with open(each_file_path, "r") as tsfile:
                header = tsfile.readline().strip().split(',')
                #print(header)
                assert header[0] == "Hours"
                
                for i in range(813):
                    if (header[i] == 'Glascow coma scale eye opening'):
                        print('the loc of Glascow coma scale eye opening')
                        print(i)
                

        count += 1
    print(header)
    
    file = open('variable_name.txt', 'w')
    for fp in header:
        file.write(fp)
        file.write('\n')
    file.close()

    GCS_eye_opening = [str(x) for x in GCS_eye_opening]
    GCS_motor_response = [str(x) for x in GCS_motor_response]
    GCS_scale_total = [str(x) for x in GCS_scale_total]
    GCS_verbal_response = [str(x) for x in GCS_verbal_response]
    GCS_eye_opening = [x for x in GCS_eye_opening if x != 'nan']
    GCS_motor_response = [x for x in GCS_motor_response if x != 'nan']
    GCS_scale_total = [x for x in GCS_scale_total if x != 'nan']
    GCS_verbal_response = [x for x in GCS_verbal_response if x != 'nan']
    print('eye opening')
    print(GCS_eye_opening)
    print('motor response')
    print(GCS_motor_response)
    print('scale total')
    print(GCS_scale_total)
    print('verbal response')
    print(GCS_verbal_response)
    
    #print(GCS_eye_opening[0])
    #print(type(GCS_eye_opening[0]))
    '''
    data_loader = common_utils_SICH_all_numeric.InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'in_hospital_mortality/'),
                                                                    listfile=os.path.join(args.data_path, 'listfile.csv'), 
                                                                    period_length=24.0)
    discretizer = Discretizer(timestep=1.0,
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')
    
    discretizer_header = discretizer.transform(data_loader.read_example(0)["X"])[1].split(',')
    print(discretizer_header)
    
    a b c d e f g h i j(10) k l m n o p q r s t u v w x y z
    '''