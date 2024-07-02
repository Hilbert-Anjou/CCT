# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:22:51 2022

@author: JyGuo
"""

from __future__ import absolute_import
from __future__ import print_function

from . import common_utils_SICH_all_numeric
import threading
import os
import numpy as np
import random


def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    return data


def load_data(reader, discretizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils_SICH_all_numeric.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    #discharge_location = ret["z"]
    names = ret["name"]
    #print(names)
    #for (X, t, name) in zip(data, ts, names):
    #    print(name)
    #    print(discretizer.transform(X, end=t)[0])
    data = [discretizer.transform(X, end=t, name=name)[0]  for (X, t, name) in zip(data, ts, names)]
    #if normalizer is not None:
    #    data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    #whole_data = (np.array(data), labels, discharge_location)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, ts, pred, y_true, path):
    common_utils_SICH_all_numeric.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,period_length,prediction,y_true\n")
        for (name, t, x, y) in zip(names, ts, pred, y_true):
            f.write("{},{:.6f},{:.6f},{}\n".format(name, t, x, y))


def save_my_results(names, pred, y_true, path):
    common_utils_SICH_all_numeric.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:                                                  #i+
        f.write("stay, 1 - prediction, prediction,y_true\n")
        for (name,  x, y) in zip(names,  pred, y_true):
            # print(name.shape)
            # print(x.shape)
            # print(y.shape)
            # f.write("{},{},{},{}\n".format(name, x[0],x[1], y))
            #if int(y) == 1: # only write down those that are true '1' labels
            f.write("{},{:.6f},{:.6f},{}\n".format(name, x[0], x[1], y))

# def save_results_pr(precisions, recalls, path):
#     common_utils.create_directory(os.path.dirname(path))
#     with open(path, 'w') as f:
#         f.write("stay,period_length,prediction,y_true\n")
#         for (name, t, x, y) in zip(names, ts, pred, y_true):
#             f.write("{},{:.6f},{:.6f},{}\n".format(name, t, x, y))