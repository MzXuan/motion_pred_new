# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os, logging
import numpy as np
import tensorflow as tf
import flags
import pickle

from model import RNN_MODEL
from data_processing import generate_data

from scipy.ndimage.filters import gaussian_filter1d

from utils import restore_data
from utils import traj_gen


FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
  os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
  os.makedirs(FLAGS.sample_dir)


# the current file path
FILE_PATH = os.path.dirname(__file__)
TASK_NAME_LIST = []


data_path = os.path.join(FILE_PATH, './model', FLAGS.model_name)

if FLAGS.run_mode is 0:
    try:
        ## load model and dataset
        X = pickle.load(open(data_path+"/x_set_train"+".pkl", "rb"))
        Y = pickle.load(open(data_path+"/y_set_train"+".pkl", "rb"))
    except:
        ## generate train/val/test datasets based on raw data
        X, Y = generate_data('./reg_fmt_datasets_1spd.pkl')
        # save dataset
        # os.mkdir(data_path)
        pickle.dump(X,open(data_path+"/x_set_train"+".pkl", "wb"))
        pickle.dump(Y,open(data_path+"/y_set_train"+".pkl", "wb"))
        print ("Save data successfully!")
else:
    try:
        ## load model and dataset
        X = pickle.load(open(data_path+"/x_set_test"+".pkl", "rb"))
        Y = pickle.load(open(data_path+"/y_set_test"+".pkl", "rb"))
    except:
        ## generate train/val/test datasets based on raw data
        X, Y = generate_data('./reg_fmt_datasets_6spd.pkl')
        # save dataset
        # os.mkdir(data_path)
        pickle.dump(X,open(data_path+"/x_set_test"+".pkl", "wb"))
        pickle.dump(Y,open(data_path+"/y_set_test"+".pkl", "wb"))
        print ("Save data successfully!")




def predict_test_set(X,Y):
    print('testing trajs')
    global rnn_model

    true_all_result = []
    pred_all_result = []
    x_all_result = []

    if rnn_model.load() is not None:
        for test_x,test_y in zip(X,Y):
            true_class_result = []
            pred_class_result = []
            x_class_result = []

            for traj_x,traj_y in zip(test_x,test_y):
                true_y, pred_y = rnn_model.test_onetraj(traj_x,traj_y)

                true_y = true_y[0]
                pred_y = pred_y[0]

                # restore to origin data (0-1 to original range)
                y_true_restore = restore_data.restore_dataset(true_y)
                pred_restore = restore_data.restore_dataset(pred_y)
                x_true_restore = restore_data.restore_dataset(traj_x, human_flag=True)

                # save data
                true_class_result.append(y_true_restore)
                pred_class_result.append(pred_restore)
                x_class_result.append(x_true_restore)

            true_all_result.append(true_class_result)
            pred_all_result.append(pred_class_result)
            x_all_result.append(x_class_result)

        # for future use
        pickle.dump(true_all_result, open(data_path + "/true_all_result.pkl", "wb"))
        pickle.dump(pred_all_result, open(data_path + "/pred_all_result.pkl", "wb"))
        pickle.dump(x_all_result, open(data_path + "/x_all_result.pkl", "wb"))
        print("save successfully!")


def online_test(human_motion):

    # max_min_transform
    human_origin_motion = restore_data.scale_hdata(human_motion)

    # gaussian
    human_origin_motion_filtered = gaussian_filter1d(human_origin_motion.T, sigma=5).T

    # predict (out time steps)


    # max_min_re-trans
    robot_origin_motion = restore_data.restore_data()
    return robot_origin_motion


def load_data():
    print('-1')


def main():
    global rnn_model
    global DATASETS
    with tf.Session() as sess:
        DATASETS = [X, Y]
        ## new rnn model
        rnn_model = RNN_MODEL(sess, FLAGS, DATASETS)

        ## train or test
        if FLAGS.run_mode is 0:
            rnn_model.train()
            rnn_model.test()

        if FLAGS.run_mode is 1:
            if rnn_model.load() is not None:
              rnn_model.test()

        if FLAGS.run_mode is 2:
          print('testing trajs, this may take about 1 minute...')
          #predict and save
          predict_test_set(X['test_class'], Y['test_class'])


        if FLAGS.run_mode is 3:
            if rnn_model.load() is not None:
              test_x = X['test']
              test_y = Y['test']
              for i in range(0,2):
                rnn_model.test_online(test_x[0], test_y[0])


if __name__ == '__main__':
    main()
