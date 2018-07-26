# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pickle as pkl
import tensorflow as tf
import time

import csv

class RNN_MODEL(object):
  def __init__(self, sess, FLAGS, DATASETS):
    ## extract FLAGS
    self.sess = sess
    
    self.num_units = FLAGS.num_units
    self.num_stacks = FLAGS.num_stacks
    self.num_dense_units = FLAGS.num_dense_units
    
    self.in_dim = FLAGS.in_dim
    self.out_dim = FLAGS.out_dim
    
    self.in_timesteps = FLAGS.in_timesteps
    
    self.out_timesteps_min = FLAGS.out_timesteps_min
    self.out_timesteps_max = FLAGS.out_timesteps_max
    self.out_timesteps_range = list(range(self.out_timesteps_min, self.out_timesteps_max+1))  # output timesteps range
    self.out_timesteps_sum = sum(self.out_timesteps_range)  # total output timesteps

    self.validation_interval = FLAGS.validation_interval
    self.checkpoint_interval = FLAGS.checkpoint_interval
    self.sample_interval = FLAGS.sample_interval
    self.display_interval = FLAGS.display_interval
    self.checkpoint_dir = FLAGS.checkpoint_dir
    self.sample_dir = FLAGS.sample_dir
    
    self.max_iteration = FLAGS.max_iteration
    self.batch_size = FLAGS.batch_size
    self.learning_rate = FLAGS.learning_rate
    self.test_type = FLAGS.test_type
    
    self.loss_mode = FLAGS.loss_mode
    self.out_dim_wgts = [FLAGS.out_dim_wgt1, FLAGS.out_dim_wgt2, FLAGS.out_dim_wgt3]

    self.validate_data = []


    ## split datasets
    X = DATASETS[0]
    Y = DATASETS[1]
    self.train_x = X['train']
    self.train_y = Y['train']
    self.val_x = X['val']
    self.val_y = Y['val']
    self.test_x = X['test']
    self.test_y = Y['test']
    
    self.idx = 0  ## index for iterator during training
    
    ## build model
    self.build()


  def calculate_loss(self):
    ## setup optimization
    self.loss_list = []
    for i, (y, pred) in enumerate(zip(self.y_list, self.pred_list)):
      print('shapes of y[{0}] and pred[{1}]: {2} == {3}'.format(self.out_timesteps_range[i], self.out_timesteps_range[i], y.get_shape(), pred.get_shape()))

      ## calculate loss as a whole based on all dimensionalities
      if self.loss_mode is 0:
        loss = tf.losses.mean_squared_error(y, pred)
      ## calculate loss one by one based on each dimensionality
      
      if self.loss_mode is 1:
        ## split the data based on each dimensionality
        y_list = tf.split(y, self.out_dim, axis=2)
        pred_list = tf.split(pred, self.out_dim, axis=2)
        ## sum the loss of each dimensionality
        loss = []
        for i in range(self.out_dim):
          # loss.append(tf.losses.mean_squared_error(y_list[i], pred_list[i], weights=self.out_dim_wgts[i]))
          loss.append(tf.losses.mean_squared_error(y_list[i], pred_list[i]))
        loss = tf.add_n(loss)

      ## average each output timesteps
      weight = 1.0 
      self.loss_list.append(loss * weight)
    
    self.loss_sum = tf.reduce_mean(tf.stack(self.loss_list))


  def build(self):
    ## define input and output
    self.x = tf.placeholder(tf.float32, shape=[None, self.in_timesteps, self.in_dim], name='in_timesteps')
    self.y = tf.placeholder(tf.float32, shape=[None, self.out_timesteps_sum, self.out_dim], name='in_timesteps')
    
    ## split the true y based on each output timesteps
    self.y_list = tf.split(self.y, self.out_timesteps_range, axis=1)
    for i in range(len(self.y_list)):
      print('shape of y_list[{0}]: {1}\n'.format(i, self.y_list[i].get_shape()))

    ## stack multiple rnn cells
    cells = [tf.nn.rnn_cell.BasicLSTMCell(self.num_units) for i in range(self.num_stacks)]
    multi_rnn = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    x2seq = tf.unstack(self.x, axis=1) # split x into seq based on timestep
    print('x2seq length:{0}, x2seq[0] shape:{1}'.format(len(x2seq), x2seq[0].get_shape()))
    outputs, state = tf.nn.static_rnn(multi_rnn, x2seq, dtype=tf.float32)
    
    ## slice the output at the latest timestep.
    output_latest = outputs[-1]
    print('shape of the latest output: ', output_latest.get_shape())

    ## FLAG: remove this hidden layer
    ## add one hidden layer
    hidden = tf.layers.dense(output_latest, self.num_dense_units, activation=tf.nn.relu, name='hidden')
    # hidden = tf.layers.dense(output_latest, self.out_timesteps_sum * self.out_dim, activation=tf.nn.relu, name='hidden')
    
    ## perform prediction for each output timesteps with one dense layer
    self.pred_list = []
    for out_timesteps in self.out_timesteps_range:
      out_timesteps_dim = out_timesteps * self.out_dim  # dimensionality of dense layer [out_timesteps * self.out_dim]
      pred = tf.layers.dense(hidden, out_timesteps_dim, activation=tf.nn.sigmoid, name='pred' + str(out_timesteps)) # activation can be set as None

      # pred = tf.layers.dense(output_latest, out_timesteps_dim, activation=tf.nn.sigmoid,
      #                        name='pred' + str(out_timesteps))  # activation can be set as None
      pred = tf.reshape(pred, [-1, out_timesteps, self.out_dim])  # reshape to [out_timesteps, self.out_dim]
      self.pred_list.append(pred)

    ## setup optimization
    self.calculate_loss()
    self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_sum, var_list=tf.trainable_variables())
    print('var_list:\n', tf.trainable_variables())
    
    ## save summary
    tf.summary.scalar('loss_sum', self.loss_sum)
    for i in range(len(self.loss_list)):
      tf.summary.scalar('loss_list'+str(i), self.loss_list[i])
      
    self.merged_summary = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

    ## new a saver for save and load model
    self.saver = tf.train.Saver()


  def feed_data(self):
    idx_max = self.train_x.shape[0]

    ## fetch a batch of samples iteratively
    idx = self.idx
    end = idx + self.batch_size
    
    if end < idx_max:
      self.idx += self.batch_size

      x = self.train_x[idx:end]
      y = self.train_y[idx:end]
    else:
      end = self.batch_size - (idx_max - idx)
      self.idx = end
      
      x1 = self.train_x[idx:idx_max]
      x2 = self.train_x[0:end]
      y1 = self.train_y[idx:idx_max]
      y2 = self.train_y[0:end]
      
      x = np.concatenate((x1, x2))
      y = np.concatenate((y1, y2))

    return x, y


  def train(self):
    ## preload the previous saved model or initialize model from scratch
    path_name = self.load()
    if path_name is None:
      ## initialize global variables
      self.sess.run(tf.global_variables_initializer())
      print("initialize model training first")
      
    ## train model
    for iteration in range(self.max_iteration):
      ## feed data
      x, y = self.feed_data()
      
      ## run training
      fetches  = [self.train_op, self.merged_summary]
      fetches += [self.loss_list, self.loss_sum, self.y_list, self.pred_list]
      feed_dict = {self.x:x, self.y:y}

      train_op, merged_summary, \
      loss_list, loss_sum, y_list, pred_list = self.sess.run(fetches, feed_dict)

      self.file_writer.add_summary(merged_summary, iteration)


      ## validate model
      if (iteration % self.validation_interval) is 0:
        self.validate()
      
      ## save model
      if (iteration % self.checkpoint_interval) is 0:
        self.save(iteration)

      ## sample datapoints
      if (iteration % self.sample_interval) is 0:
        postfix = str(iteration) + '.pkl'
        pkl.dump(y_list, open(self.sample_dir + '/y_list_' + postfix, 'wb'))
        pkl.dump(pred_list, open(self.sample_dir + '/pred_list_' + postfix, 'wb'))

      ## display information
      if (iteration % self.display_interval) is 0:
        print('\n')
        print('iteration {0}: loss_sum of out_timesteps = {1}'.format(iteration, loss_sum))
        for i, loss in enumerate(loss_list):
          print('iteration {0}: loss of out_timesteps[{1}] = {2}'.format(iteration, self.out_timesteps_range[i], loss))

    self.write_csv()
    self.file_writer.close()


  def validate(self):
    print('validating ...')
    fetches  = [self.y_list, self.pred_list]
    feed_dict = {self.x:self.val_x, self.y:self.val_y}
    y_list, pred_list = self.sess.run(fetches, feed_dict)
    
    rmse_sum = 0
    for i, (y, pred) in enumerate(zip(y_list, pred_list)):
      rmse = np.sqrt(np.square(y - pred).mean())
      rmse_sum += rmse
      print('validation out_timesteps[{0}] rmse={1}'.format(self.out_timesteps_range[i], rmse))
    print('validation all_timesteps rmse_sum=', rmse_sum)
    self.validate_data.append(rmse_sum)



  def test(self):
    if self.test_type is 0:
      self.test_asawhole()
    else:
      self.test_onebyone()


  def test_asawhole(self):
    print('testing as a whole...')
    
    fetches  = [self.y_list, self.pred_list]
    feed_dict = {self.x:self.test_x, self.y:self.test_y}
    y_list, pred_list = self.sess.run(fetches, feed_dict)
       
    rmse_sum = 0
    for i, (y, pred) in enumerate(zip(y_list, pred_list)):
      rmse = np.sqrt(np.square(y - pred).mean())
      rmse_sum += rmse
      print('out_timesteps[{0}] rmse={1}'.format(self.out_timesteps_range[i], rmse))
    print('all_timesteps rmse_sum=', rmse_sum)
    
    for i, (y, pred) in enumerate(zip(y_list, pred_list)):
      print('out_timesteps[{0}] y={1}'.format(self.out_timesteps_range[i], y[-1]))
      print('out_timesteps[{0}] pred={1}'.format(self.out_timesteps_range[i], pred[-1]))
      print('\n')


  def test_onetraj(self, traj_x, traj_y):
      # print('testing one traj')
      fetches = [self.y_list, self.pred_list]
      feed_dict = {self.x: traj_x,
                   self.y: traj_y}
      y_list, pred_list = self.sess.run(fetches, feed_dict)

      return y_list,pred_list

  def test_onebyone(self):
    print('testing one by one ...')
    idx_max = self.test_x.shape[0]
    
    for idx in range(idx_max):
      # ticks = time.time()
      
      ## get single sample
      fetches  = [self.y_list, self.pred_list]
      feed_dict = {self.x:np.expand_dims(self.test_x[idx], axis=0), self.y:np.expand_dims(self.test_y[idx], axis=0)}
      y_list, pred_list = self.sess.run(fetches, feed_dict)
      
      # print("time used for predicting one sample: ", time.time() - ticks)
      
      ## print rmse
      rmse_sum = 0
      for i, (y, pred) in enumerate(zip(y_list, pred_list)):
        rmse = np.sqrt(np.square(y - pred).mean())
        rmse_sum += rmse
        print('samples[{0}] out_timesteps[{1}] rmse={2}'.format(idx, self.out_timesteps_range[i], rmse))
      print('all_timesteps rmse_sum=', rmse_sum)
      print('\n')
      
    for i, (y, pred) in enumerate(zip(y_list, pred_list)):
      print('out_timennsteps[{0}] y={1}'.format(self.out_timesteps_range[i], y))
      print('out_timesteps[{0}] pred={1}'.format(self.out_timesteps_range[i], pred))
      print('\n')


  def test_online(self, x, y):
    print('testing online one by one ...')
    ticks = time.time()
    
    fetches  = [self.y_list, self.pred_list]
    feed_dict = {self.x:np.expand_dims(x, axis=0), self.y:np.expand_dims(y, axis=0)}
    y_list, pred_list = self.sess.run(fetches, feed_dict)
    
    print("time used for predicting one sample: ", time.time() - ticks)
    
    ## print rmse
    rmse_sum = 0
    for i, (y, pred) in enumerate(zip(y_list, pred_list)):
      rmse = np.sqrt(np.square(y - pred).mean())
      rmse_sum += rmse
      print('out_timesteps[{0}] rmse={1}'.format(self.out_timesteps_range[i], rmse))
    print('all_timesteps rmse_sum=', rmse_sum)
    print('\n')
      
    for i, (y, pred) in enumerate(zip(y_list, pred_list)):
      print('out_timennsteps[{0}] y={1}'.format(self.out_timesteps_range[i], y))
      print('out_timesteps[{0}] pred={1}'.format(self.out_timesteps_range[i], pred))
      print('\n')

  def test_x_online(self, x):
    # print('testing online one by one ...')
    ticks = time.time()

    fetches = [self.pred_list]
    feed_dict = {self.x: np.expand_dims(x, axis=0)}
    pred_list = self.sess.run(fetches, feed_dict)

    # print("time used for predicting one sample: ", time.time() - ticks)
    # print(pred_list)

    return pred_list


  def save(self, iteration):
    ## save the model iteratively
    print('iteration {0}: save model to {1}'.format(iteration, self.checkpoint_dir))
    self.saver.save(self.sess, self.checkpoint_dir + '/model.ckpt', global_step=iteration)


  def load(self):
    ## load the previous saved model
    path_name = tf.train.latest_checkpoint(self.checkpoint_dir)
    if path_name is not None:
      self.saver.restore(self.sess, path_name)
      print('restore model from checkpoint path: ', path_name)
    else:
      print('no checkpoints are existed in ', self.checkpoint_dir)
      
    return path_name

  def write_csv(self):
      with open('validate.csv', 'a') as csvfile:
          spamwriter = csv.writer(csvfile, delimiter=',',
                                  quotechar=',', quoting=csv.QUOTE_MINIMAL)
          spamwriter.writerow(self.validate_data)