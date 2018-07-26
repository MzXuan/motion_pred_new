# -*- coding: utf-8 -*-
import os
from datetime import datetime
import tensorflow as tf

flags = tf.app.flags


##model name
flags.DEFINE_string('model_name','r_js_1speed_10_10_new_new','name of the model')


## model hyper-parameters
flags.DEFINE_integer('num_units', 50, 'number of units of a rnn cell')
flags.DEFINE_integer('num_stacks', 1, 'number of stacked rnn cells')
flags.DEFINE_integer('num_dense_units', 32, 'number of hidden dense layer')

flags.DEFINE_integer('in_dim', 6, 'dimensionality of each timestep input')
flags.DEFINE_integer('out_dim', 7, 'dimensionality of each timestep output')
# add weights for each dimensionality of output
flags.DEFINE_integer("loss_mode", 1, "0: calculate loss as a whole; 1: calculate loss one by one")
flags.DEFINE_integer('out_dim_wgt1', 1, 'The 1th weight for each dimensionality of output')
flags.DEFINE_integer('out_dim_wgt2', 1, 'The 2th weight for each dimensionality of output')
flags.DEFINE_integer('out_dim_wgt3', 1, 'The 3th weight for each dimensionality of output')
#
flags.DEFINE_integer('in_timesteps', 10, 'input timesteps')
flags.DEFINE_integer('out_timesteps_min', 10, 'minimal output timesteps')
flags.DEFINE_integer('out_timesteps_max', 10, 'maximal output timesteps')

## optimization hyper-parameters
flags.DEFINE_integer('max_iteration', 11000, 'max iteration of training model')
flags.DEFINE_integer('batch_size', 64, 'batch size of datapoints')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate of optimization')

## todo & can be adjusted
flags.DEFINE_integer("run_mode", 2, "0 for training, 1 for testing, 2 for class, 3 for online")
flags.DEFINE_integer("test_type", 0, "0: testing as a whole; 1: testing one by one")

## log hyper-parameters
flags.DEFINE_integer('validation_interval', 1000, 'interval of performing validation')
flags.DEFINE_integer('checkpoint_interval', 1000, 'interval of saving checkpoint')
flags.DEFINE_integer('sample_interval', 100, 'interval of sampling datapoints')
flags.DEFINE_integer('display_interval', 100, 'interval of displaying information')

## log directory 
num_units = 'units' + str(flags.FLAGS.num_units)
num_stacks = 'stacks' + str(flags.FLAGS.num_stacks)
num_dense_units = 'dense' + str(flags.FLAGS.num_dense_units)
in_dim = 'in' + str(flags.FLAGS.in_dim)
out_dim = 'out' + str(flags.FLAGS.out_dim)
outmin = 'min' + str(flags.FLAGS.out_timesteps_min)
outmax = 'max' + str(flags.FLAGS.out_timesteps_max)
batch_size = 'batch' + str(flags.FLAGS.batch_size)


if flags.FLAGS.run_mode is 0:
  stamp = 'stamp' + datetime.now().strftime("%Y%m%d-%H%M-%S")
  # postfix = '_'.join([stamp, num_units, num_stacks, in_dim, out_dim, outmin, outmax, batch_size])
  postfix = '_'.join([num_units, num_stacks, in_dim, out_dim, outmin, outmax, batch_size])
else:
  postfix = '_'.join([num_units, num_stacks, in_dim, out_dim, outmin, outmax, batch_size])


model_name = flags.FLAGS.model_name
# checkpoint_dir = './model/'+model_name+'/checkpoint_' + postfix
# sample_dir = './model/'+model_name+'/sample_' + postfix


checkpoint_dir = '/home/xuan/Dropbox/code_using/human-robot_interaction_api-updated/model/'+model_name+'/checkpoint_' + postfix
sample_dir = '/home/xuan/Dropbox/code_using/human-robot_interaction_api-updated/model/'+model_name+'/sample_' + postfix
# print(checkpoint_dir, sample_dir)


flags.DEFINE_string('checkpoint_dir', checkpoint_dir, 'Directory name to save checkpoints')
flags.DEFINE_string('sample_dir', sample_dir, 'Directory name to save datapoints')

FLAGS = flags.FLAGS