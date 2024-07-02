# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:14:16 2022

@author: JyGuo
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import json
import random

from .feature_extractor import extract_features


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if len(channel_info[channel]['possible_values']) != 0:
            ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
        ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
    return ret


def extract_features_from_rawdata(chunk, header, period, features):
    with open(os.path.join(os.path.dirname(__file__), "resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def sort_and_shuffle(data, batch_size):
    """ Sort data by the length and then make batches and shuffle them.
        data is tuple (X1, X2, ..., Xn) all of them have the same length.
        Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i: i+batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data


def add_common_arguments(parser):
    """ Add all the parameters which are common across the tasks
    """
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--dim', type=int, default=256,
                        help='number of hidden units')
    parser.add_argument('--depth', type=int, default=1,
                        help='number of bi-LSTMs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
    parser.add_argument('--load_state', type=str, default="",
                        help='state file path')
    parser.add_argument('--mode', type=str, default="train",
                        help='mode: train or test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
    parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save state every x epoch')
    parser.add_argument('--prefix', type=str, default="",
                        help='optional prefix of network name')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rec_dropout', type=float, default=0.0,
                        help="dropout rate for recurrent connections")
    parser.add_argument('--batch_norm', type=bool, default=False,
                        help='batch normalization')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--small_part', dest='small_part', action='store_true')
    parser.add_argument('--whole_data', dest='small_part', action='store_false')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta_1 param for Adam optimizer')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--size_coef', type=float, default=4.0)
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                             'use one of the provided ones.')
    parser.set_defaults(small_part=False)

class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]                                 # timeseries name
        self._data = self._data[1:]                                           # length of stay and in-hospital mortality

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)

class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=24.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}

class DeepSupervisionDataLoader:
    r"""
    Data loader for decompensation and length of stay task.
    Reads all the data for one patient at once.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listfile : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """
    def __init__(self, dataset_dir, listfile=None, small_part=False):

        self._dataset_dir = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()[1:]  # skip the header, and take in all the data

        self._data = [line.split(',') for line in self._data] # [the id_timeseries_episodex, hours, lable/ground truth]
        self._data = [(x, float(t), y) for (x, t, y) in self._data]
        self._data = sorted(self._data)                       # this is why the training and validation set can be random

        mas = {"X": [],
               "ts": [],
               "ys": [],
               "name": []}
        i = 0
        # died = 0
        while i < len(self._data):
            j = i
            cur_stay = self._data[i][0]
            cur_ts = []
            cur_labels = []
            # die = False
            while j < len(self._data) and self._data[j][0] == cur_stay: # take the records belonging to one episode_timeseries.csv file in one
                cur_ts.append(self._data[j][1])
                cur_labels.append(self._data[j][2])
                #print(self._data[j][2].type())
                # if int(self._data[j][2]) == 1:
                #     die = True
                j += 1
                
            # if die == True:
            #     died += 1
            
            
            cur_X, header = self._read_timeseries(cur_stay)
            mas["X"].append(cur_X) # the input/bio_information
            mas["ts"].append(cur_ts) # the time that the information is taken
            mas["ys"].append(cur_labels) # the ground truth
            mas["name"].append(cur_stay) # the corresponding csv file

            i = j
            if small_part and len(mas["name"]) == 256:
                break

        self._data = mas
        # print('true died people')
        # print(died)
        # print('total number')
        # print(len(mas["name"]))
              
    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile: # take the bio_information (input data) in
            header = tsfile.readline().strip().split(',') # the 1st line is the header
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas)) # every row is one bio_information
        return (np.stack(ret), header)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr]) # max_len is the maximum total time_step of x/ single data
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) # concatenate 0's to the latter rows of data
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)
