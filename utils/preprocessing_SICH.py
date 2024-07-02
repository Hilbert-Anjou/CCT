# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 00:35:25 2022

@author: JyGuo
"""
import numpy as np
import platform
import pickle
import json
import os

GCS_eye_map = {'Spontaneously': 3,
               '4 Spontaneously': 3,
               'To speech': 2,
               'To Speech': 2,
               '3 To speech': 2,
               'To pain': 1,
               'To Pain': 1,
               '2 To pain': 1,
               '1 No Response': 0,
               'None': 0,
}

class Discretizer:
    def __init__(self, timestep=0.8, store_masks=False, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), 'resources/discretizer_config_SICH.json')):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            #self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel) - 2)))
            self._channel_to_id = {"Chloride (serum)": 0,
                                   "Creatinine": 1,
                                   "Glascow coma scale eye opening": 2,
                                   "Glucose (serum)": 3,
                                   "Heart Rate": 4, 
                                   "Hematocrit": 5, 
                                   "Magnesium": 6,
                                   "Sodium (serum)": 7,
                                   "White blood cell count (blood)": 8, 
                                   "White blood cell count (urine)": 9,
                                   "pH": 10, 
                                   "pH (urine)": 11
                }
            self._is_categorical_channel = config['is_categorical_channel'] # categorical channel needs to be transformed into one-hot format
            self._possible_values = config['possible_values']
            # for x in self._possible_values['Glascow coma scale eye opening']:
            #     print(x)
            #     print(self._possible_values['Glascow coma scale eye opening'].index(x))
                
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None, name = None):
        if header is None:
            header = self._header
        #if name is not None:
        #    print(name)
        assert header[0] == "Hours"
        eps = 1e-6
        
        N_channels = len(self._id_to_channel) - 2 # which should be 17
        ts = [float(row[0]) for row in X] # the time of every row (bio-information)
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps  # check the time is increasing/logical/real

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps) # number of "1 hour"s in the total time slot

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            if i >= 2:
                loc = i + 2
            else:
                loc = i
            channel = self._id_to_channel[loc]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + 4                                 # since there is only 4 categories for GCS - eye opening
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]
        data = np.zeros(shape=(N_bins, cur_len), dtype=float)   # cur_len should be 59, fixed (categorical + non-catogorical) for transformation
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            #print(channel)
            #print(channel_id)
            #print(self._is_categorical_channel[channel])
            if self._is_categorical_channel[channel]:                     # transform categorical channel into one-hot format information
                category_id = GCS_eye_map[value]
                #category_id = self._possible_values[channel].index(value) # decide the loaction of '1' in one-hot
                N_values = 4                                              # decide the total length of one-hot
                one_hot = np.zeros((N_values,))                           # create one-hot
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos] # copy the one-hot information into data
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)        # copy the value into data
        previous_value = ["" for i in range(N_channels)]
        for row in X:                                                     # fill out all the existing data
            if float(row[0]) < 0:
                for j in range(1, len(row)):
                    if ((j == 3) or (j == 4)):
                        continue
                    if row[j] == "":
                        continue
                    channel = header[j]
                    channel_id = self._channel_to_id[channel]
                    previous_value[channel_id] = row[j]
        #print(previous_value)
        #print(original_value)
        original_value[0] = previous_value
        for j in range(1, 15):
            if ((j == 3) or (j == 4)):
                continue
            channel = header[j]
            channel_id = self._channel_to_id[channel]
            if (previous_value[channel_id] == ""):
                continue
            mask[0][channel_id] = 1
            write(data, 0, channel, previous_value[channel_id], begin_pos)
        for row in X:
            if float(row[0]) >= 0:
                t = float(row[0]) - first_time
                if t > max_hours + eps:                                       # exclude the information that exceeds the max time limit
                    continue
                bin_id = int(t / self._timestep - eps)                        # the information acquired at t(th) hour 
                if bin_id < 0:
                    continue
                assert bin_id < N_bins                                   # normal check

                for j in range(1, len(row)):
                    if ((j == 3) or (j == 4)):
                        continue
                    if row[j] == "":
                        continue                                              # leave the null (no information category) untouched
                    channel = header[j]
                    channel_id = self._channel_to_id[channel]

                    total_data += 1
                    if mask[bin_id][channel_id] == 1:                         # note : some question here
                        unused_data += 1
                    mask[bin_id][channel_id] = 1
                    #print(channel)
                    write(data, bin_id, channel, row[j], begin_pos)           # use the write function to create data from texts and some values
                    original_value[bin_id][channel_id] = row[j]               # record the original value

        # impute missing values

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")
            
        if self._impute_strategy in ['normal_value', 'previous']:         # fill out the missing part, by copying previous record or a normal value
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    if ((channel == 'Creatinine (urine)') or (channel == 'Creatinine clearance')):
                        continue
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]  # take a normal value, since no previous records
                        else:
                            imputed_value = prev_values[channel_id][-1]   # take the latest previous value
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)]) # number of bins/time slots that have no original data at all
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)
        #if (name == '45009_episode2_timeseries.csv'):
        #    print(data)
        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])
        #print(data)
 
        # create new header
        new_header = []
        for channel in self._id_to_channel:    # record the all possible values that can be taken for categorical channels
            if ((channel == 'Creatinine (urine)') or (channel == 'Creatinine clearance')):
                continue
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:                              # else, only record the non-categorical channel
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)): # the id's for each channel
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)
        
        #if (name == '72627_episode1_timeseries.csv'):
        #    print(data)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(100.0 * self._empty_bins_sum / self._done_count))


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']
            ##
            # print('dct=',dct)
            # print(self._means)
            # print(self._stds)

    def transform(self, X): # to normalize non-categorical data, minus its mean, devided by its standard deviation
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret
