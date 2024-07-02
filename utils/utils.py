from __future__ import absolute_import
from __future__ import print_function

from . import common_utils
import threading
import os
import numpy as np
import random


def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    return data


class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer,
                 batch_size, steps, shuffle, return_names=False):
        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        if steps is None:
            self.n_examples = reader.get_number_of_examples()
            self.steps = (self.n_examples + batch_size - 1) // batch_size
        else:
            self.n_examples = steps * batch_size
            self.steps = steps

        self.chunk_size = min(1024, self.steps) * batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                remaining -= current_size

                ret = common_utils.read_chunk(self.reader, current_size)
                Xs = ret["X"]
                ts = ret["t"]
                ys = ret["y"]
                names = ret["name"]

                Xs = preprocess_chunk(Xs, ts, self.discretizer, self.normalizer)
                (Xs, ys, ts, names) = common_utils.sort_and_shuffle([Xs, ys, ts, names], B)

                for i in range(0, current_size, B):
                    X = common_utils.pad_zeros(Xs[i:i + B])
                    y = np.array(ys[i:i + B])
                    batch_names = names[i:i+B]
                    batch_ts = ts[i:i+B]
                    batch_data = (X, y)
                    if not self.return_names:
                        yield batch_data
                    else:
                        yield {"data": batch_data, "names": batch_names, "ts": batch_ts}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


class BatchGenDeepSupervision(object):

    def __init__(self, dataloader, discretizer, normalizer,
                 batch_size, shuffle, return_names=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_per_patient_data(dataloader, discretizer, normalizer)

        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_per_patient_data(self, dataloader, discretizer, normalizer):
        timestep = discretizer._timestep # to decide the timestep that the process is based on, for example every 1 hour

        def get_bin(t):
            eps = 1e-6
            return int(t / timestep - eps) # get the current timestep/hour of time t

        N = len(dataloader._data["X"]) # number of all data, based on different id_episodex_timesiries.csv files
        Xs = []
        ts = []
        masks = []
        ys = []
        names = []

        for i in range(N):
            X = dataloader._data["X"][i]    # the i(th) input/bio-information
            cur_ts = dataloader._data["ts"][i] # time's of the information taken
            cur_ys = dataloader._data["ys"][i] # ground truth
            name = dataloader._data["name"][i] # the name of corresponding csv file

            cur_ys = [int(x) for x in cur_ys]

            T = max(cur_ts)
            nsteps = get_bin(T) + 1 # number of timesteps for this "patient"
            mask = [0] * nsteps     # to record the real information
            y = [0] * nsteps

            for pos, z in zip(cur_ts, cur_ys):
                mask[get_bin(pos)] = 1 # record the actual/real information, pos is computed based on cur_ts
                y[get_bin(pos)] = z    # record the lable

            X = discretizer.transform(X, end=T)[0] # transform input data, from texts to arrays
            if normalizer is not None:
                X = normalizer.transform(X)        # normalize the non-categorical data, (value - mean)/standard deviation

            Xs.append(X)                           # put the processed data into new storage, the size is (time_step, 76)
            masks.append(np.array(mask))           # record the location of real/existing data
            ys.append(np.array(y))                 # record the ground truth
            names.append(name)                     # record the name of the csv files
            ts.append(cur_ts)                      # record the discrete time, e.g. (xx.xxx)

            assert np.sum(mask) > 0                # check there is some original data
            assert len(X) == len(mask) and len(X) == len(y) # check there there is no error, element missing

        self.data = [[Xs, masks], ys]              # concatenate the transformed data with the mask/ this mask is different from the discretizer
        self.names = names
        self.ts = ts

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.data[1])             # the number of all data
                order = list(range(N))
                random.shuffle(order)             # shuffle the order of data to be extracted
                tmp_data = [[[None]*N, [None]*N], [None]*N]
                tmp_names = [None] * N
                tmp_ts = [None] * N
                for i in range(N):               # to shuffle
                    tmp_data[0][0][i] = self.data[0][0][order[i]] # the data/value part
                    tmp_data[0][1][i] = self.data[0][1][order[i]] # the mask part
                    tmp_data[1][i] = self.data[1][order[i]]       # the lables
                    tmp_names[i] = self.names[order[i]]           # the name
                    tmp_ts[i] = self.ts[order[i]]                 # the time
                self.data = tmp_data
                self.names = tmp_names
                self.ts = tmp_ts
            else:
                # sort entirely
                Xs = self.data[0][0]
                masks = self.data[0][1]
                ys = self.data[1]
                (Xs, masks, ys, self.names, self.ts) = common_utils.sort_and_shuffle([Xs, masks, ys,
                                                                                      self.names, self.ts], B)
                self.data = [[Xs, masks], ys]

            for i in range(0, len(self.data[1]), B): # take out the data batch by batch
                X = self.data[0][0][i:i + B]
                mask = self.data[0][1][i:i + B]
                y = self.data[1][i:i + B]
                names = self.names[i:i + B]
                ts = self.ts[i:i + B]

                X = common_utils.pad_zeros(X)  # (B, T, D) (batch_size, time_step, 59)
                mask = common_utils.pad_zeros(mask)  # (B, T)
                y = common_utils.pad_zeros(y)
                y = np.expand_dims(y, axis=-1)  # (B, T, 1)
                batch_data = ([X, mask], y)
                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names, "ts": ts}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


def save_results(names, ts, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,period_length,prediction,y_true\n")
        for (name, t, x, y) in zip(names, ts, pred, y_true):
            f.write("{},{:.6f},{:.6f},{}\n".format(name, t, x, y))


def save_my_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
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