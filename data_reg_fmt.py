'''
This script only works in python2 environment
'''
from __future__ import print_function
import pickle
import numpy as np
import pre_process_data

from scipy.interpolate import griddata

def adjust_speed(traj, factor):
    # equalize length of human_traj and robot_traj
    traj_len =len(traj)

    xp=np.linspace(0,traj_len,traj_len)
    intrpl_x = np.linspace(0,traj_len,int(traj_len/factor))

    intrpl_traj = griddata(xp, traj, intrpl_x, method='linear')
    return intrpl_traj


def gen_speed_var_trajs(dataset, factor):
    new_dataset = []
    for traj in dataset:
        try:
            new_traj = adjust_speed(traj,factor)
            new_dataset.append(new_traj)
        except:
            pass
    return new_dataset



def main():
    #todo: read new dataset
    pre_process_data.main()

    # load previous generate data
    pkl_file = open('./pkl/datasets_reg.pkl', 'rb')
    datasets = pickle.load(pkl_file)

    print('length of dataset:', len(datasets))
    print('\n')


    # combine different features
    reg_fmt_datasets = []
    for i, dataset in enumerate(datasets):
      print('length of dataset:', str(i), len(dataset))

      reg_fmt_dataset = []
      for j, sample in enumerate(dataset):
        left_hand = sample['left_hand']
        left_joints = sample['left_joints']
        reg_fmt_dataset.append(np.hstack([left_joints,left_hand]))

      print('length of reg_fmt_dataset:', len(reg_fmt_dataset))

      reg_fmt_datasets.append(reg_fmt_dataset)


    # add various speed of the trajectory

    print('generating different speed of trajectories')
    speed_list = [0.75,1.25]
    new_reg_fmt_datasets = list(reg_fmt_datasets)
    for i,dataset in enumerate(reg_fmt_datasets):
        for speed in speed_list:
            new_trajs = gen_speed_var_trajs(dataset,speed)
            new_reg_fmt_datasets.append(new_trajs)

    print('\n')
    print('length of reg_fmt_datasets:', len(new_reg_fmt_datasets))


    pickle.dump(new_reg_fmt_datasets, open("reg_fmt_datasets.pkl", "wb"))


if __name__ == '__main__':
    main()
