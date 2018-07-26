# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
import random
import flags


## configuration of the network architecture of rnn
IN_TIMESTEPS = flags.FLAGS.in_timesteps
OUT_TIMESTEPS_MIN = flags.FLAGS.out_timesteps_min
OUT_TIMESTEPS_MAX = flags.FLAGS.out_timesteps_max
IN_DIM = flags.FLAGS.in_dim
OUT_DIM = flags.FLAGS.out_dim

# robot (0:3) , hand (3:6) + elbow (6:9)

# robot (0:7), hand(7:10)+elbow(10:13)

def rnn_data_X(data):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> chose input from sequence, [[1, 2], [2, 3], [3, 4]], 
        -> chose output from sequence, [3, 4, 5]
    """
    
    # chose input from sequence considering the maximal timestep of output layer
    rnn_df = []
    for i in range(data.shape[0]  - IN_TIMESTEPS):
        # X = data[i: i+in_timesteps, 3:9]  #hand (3:6) + elbow (6:9)
        X = data[i: i + IN_TIMESTEPS, 7:13]  # hand (3:6) + elbow (6:9)
        rnn_df.append(X if len(X.shape) > 1 else [[item] for item in X])
    # print('X shape:', rnn_df[0].shape)
    
    return rnn_df


def rnn_data_Y(data):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> chose input from sequence, [[1, 2], [2, 3], [3, 4]],
        -> chose output from sequence, [1,2]
    """

    ## chose multiple outputs from sequence for inputs
    rnn_df = []
    for i in range(data.shape[0] - IN_TIMESTEPS):

        Y_list = []  # multiple outputs for one input timestep
        for out_timesteps in range(OUT_TIMESTEPS_MIN, OUT_TIMESTEPS_MAX+1):
            # Y = data[i: i + out_timesteps, 0:3]  # robot (0:3) human_hand (3:6)
            Y = data[i: i + out_timesteps, 0:7]  # robot (0:3) human_hand (3:6)
            Y = Y.reshape((out_timesteps, OUT_DIM))
            Y_list.append(Y)
            # print('input_timestep[{0}] and output_timestep[{1}] Y shape: {2}'.format(i, out_timesteps, Y.shape))

        Y_list = np.concatenate(Y_list, axis=0)
        # print("Y_list shape:", Y_list.shape)

        rnn_df.append(Y_list)

    return rnn_df


def prepare_seqs_data(sets):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    seqs_x = []
    seqs_y = []
    for seqs in sets:
        print('length of seqs:', len(seqs))

        for i, seq in enumerate(seqs):
            print('shape of seq:', seq.shape)

            seq_x = rnn_data_X(seq)
            seq_y = rnn_data_Y(seq)

            seqs_x += seq_x
            seqs_y += seq_y

    c = list(zip(seqs_x,seqs_y))
    random.shuffle(c)
    seqs_x, seqs_y = zip(*c)

    seqs_x = np.array(seqs_x, dtype=np.float32)
    seqs_y = np.array(seqs_y, dtype=np.float32)
    print("shape of seqs_x and seqs_y:", seqs_x.shape, seqs_y.shape)
    return seqs_x, seqs_y


def prepare_test_seqs_data(sets):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    # print('length of seqs:', len(seqs))

    seqs_x = []
    seqs_y = []
    for seqs in sets:
        # print('shape of seq:', sets.shape)
        seq_xs = []
        seq_ys = []
        for i, seq in enumerate(seqs):
            seq_x = rnn_data_X(seq)
            seq_y = rnn_data_Y(seq)
            seq_x = np.array(seq_x, dtype=np.float32)
            seq_y = np.array(seq_y, dtype=np.float32)

            seq_xs.append(seq_x)
            seq_ys.append(seq_y)

        seqs_x.append(seq_xs)
        seqs_y.append(seq_ys)
    print("length of seqs_x and seqs_y:", len(seqs_x), len(seqs_y))

    return seqs_x, seqs_y


# def split_data_bc(data, train_pos=0.7, val_pos=0.9, test_pos=1.0):
#     """
#     splits data to training, validation and testing parts
#     """
#     random.shuffle(data)
#
#     num = len(data)
#     train_pos = int(num * train_pos)
#     val_pos = int(num * val_pos)
#
#     train_data = data[:train_pos]
#     val_data = data[train_pos:val_pos]
#
#     test_data = data[val_pos:]
#
#     return train_data, val_data, test_data


def split_data(datasets, train_pos=0.7, val_pos=0.8, test_pos=1.0):
    """
    splits data to training, validation and testing parts
    """
    train_sets = []
    val_sets = []
    test_sets = []

    for i, task in enumerate(datasets):
        print('\nTask {0} has {1} seqs'.format(str(i), len(task)))
        train_sets.append(task[:int(len(task) * train_pos)])
        val_sets.append(task[int(len(task) * train_pos):int(len(task) * val_pos)])
        test_sets.append(task[int(len(task) * val_pos):])


    return train_sets,val_sets,test_sets


def generate_data(file_name):
    """generates data with based on a function func"""

    pkl_file = open(file_name,'rb')
    datasets = pickle.load(pkl_file)
    print('length of tasks:', len(datasets))

    train_seqs, val_seqs, test_seqs = split_data(datasets)

    
    print('\ntrain_seqs info:')
    train_x, train_y = prepare_seqs_data(train_seqs)
    
    print('\nval_seqs info:')
    val_x, val_y = prepare_seqs_data(val_seqs)

    #todo: if want to seperate class, edit prepare_test_data
    print('\ntest_seqs info:')
    test_x, test_y = prepare_seqs_data(test_seqs)

    print('\ntest_seqs info:')
    test_class_x, test_class_y = prepare_test_seqs_data(test_seqs)

    return dict(train=train_x, val=val_x, test=test_x, test_class=test_class_x), dict(train=train_y, val=val_y, test=test_y,test_class=test_class_y)


def main():
    # pkl_file = open('./reg_fmt_datasets.pkl','rb')
    # datasets = pickle.load(pkl_file)
    # print('length of tasks:', len(datasets))

    generate_data('./reg_fmt_datasets.pkl')

if __name__ == '__main__':
    main()


