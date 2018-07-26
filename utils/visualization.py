from __future__ import print_function
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

import os
import ConfigParser

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

# the current file path
FILE_PATH = os.path.dirname(__file__)
TASK_NAME_LIST = []

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(FILE_PATH, '../cfg/models.cfg'))
# read models params
MODEL_NAME = cp_models.get('model', 'model_name')

def mse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())

def dist(pred,true):
    '''
    calculate Cartesian distance error along a trajectory
    '''
    dist = []
    for p1, p2 in zip(pred, true):
        dist.append(np.sqrt(np.sum((p1-p2)**2)))
    dist = np.asarray(dist)
    return dist.mean(),dist

def plot_3D_human(human, num = 0):
    fig = plt.figure(num)
    ax = fig.gca(projection='3d')
    # print robot trajectory
    ax.plot(human[:, 0], human[:, 1], human[:, 2], '-o', linewidth=5, color='blue',
            label='human trajectory')


def plot_3D_result(pred_traj, true_traj, num = 0, save_flag = False):
    # fig = plt.figure(num)
    my_dpi = 100
    fig = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi, num=num)

    ax = fig.gca(projection='3d')
    fig.savefig("/home/xuan/Dropbox/PhD/Humanoids/result/example_trajs.eps")
    #print robot trajectory
    ax.plot(pred_traj[:,0],pred_traj[:,1],pred_traj[:,2], '-o', linewidth=5, color='yellow', label='generated robot trajectory')
    ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:, 2], '--.', linewidth=5, color='red', label='pre-recorded robot trajectory')

    # ax.legend(bbox_to_anchor=(1,1), loc='best', mode="expand", borderaxespad=0., fontsize=15)
    ax.legend(['human trajectory', 'generated robot trajectory','pre-recorded robot trajectory'], loc='best', fontsize =12)
    plt.tight_layout()



    ax.view_init(30, 45)
    ax.set_xlabel('x (m)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    ax.set_zlabel('z (m)', fontsize=14)
    # ax.xaxis.set_label_coords(1.05, -0.025, 1)
    # ax.yaxis.set_label_coords(1.3, -0.025, 1)
    return fig


def plot_dof_result(pred_traj, true_traj, num = 0 , save_flag = False):
    my_dpi = 100
    fig = plt.figure(figsize=(1250/my_dpi, 750/my_dpi),dpi=my_dpi, num=num)

    fontP = FontProperties()
    fontP.set_size('xx-large')

    # color = ['red','blue','green','m','k','']
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    joint_name = ['e0','e1','s0','s1','w0','w1','w2']
    dim = pred_traj.shape[1]
    for i in range(0,dim):
        # plt.plot(range(0,len(pred_traj)),pred_traj[:,i],'-',color='red')
        # plt.plot(range(0,len(true_traj)), true_traj[:,i], '--',color='blue')

        plt.plot(range(0,len(pred_traj)),pred_traj[:,i],'-o',color=color[i], label='generated '+joint_name[i])
        # plt.plot(range(0,len(true_traj)), true_traj[:,i], '--.',color=color[i], label='true result')

    for i in range(0,dim):
        plt.plot(range(0,len(true_traj)), true_traj[:,i], '--.',color=color[i], label='true '+joint_name[i])

    # plt.legend(bbox_to_anchor=(1, 1), loc=2, mode="expand", borderaxespad=0., fontsize = 12)

    plt.legend(bbox_to_anchor=(0.05, 0.85,1,0), mode = 'expand', loc=4, ncol=4,fontsize = 20, prop=fontP)
    # plt.legend(mode='expand', loc='best', ncol=4, fontsize=20, prop=fontP)
    # fig.set_figwidth(20)
    # fig.suptitle('Example of the predicted result and ground truth of a test trajectory in joint space', fontsize=20)
    plt.xlabel('Time steps', fontsize=22)
    plt.ylabel('Joint angle (radian)', fontsize=22)
    plt.axis(xmin=0)
    plt.grid()
    # fig.savefig("/home/xuan/Dropbox/PhD/Humanoids/result/example_in_joint_space.eps")
    if save_flag == True:
        fig.savefig("/home/xuan/Dropbox/PhD/Humanoids/result/example_in_joint_space.eps")


def visual_dof():
    ## plot step vs simple trajectory
    y_true = pickle.load(open("test_y_true.pkl", "rb"))
    predicted = pickle.load(open("test_y_predicted.pkl","rb"))


    for pred,true in zip(predicted,y_true):
        try:
            test_pred
        except:
            test_pred=np.copy(pred)
            test_true=np.copy(true)
        else:
            test_pred = np.vstack([test_pred,pred])
            test_true = np.vstack([test_true,true])

    plot_dof_result(test_pred, test_true)

    plt.show()


def main():
    # data_path = os.path.join(FILE_PATH, '../model', MODEL_NAME)
    # y_true = pickle.load(open(data_path+"/raw_true_trajs.pkl", "rb"))
    # predicted = pickle.load(open(data_path+"/raw_pred_trajs.pkl","rb"))
    #
    # #for robot trajectory only lists
    # for true_traj,pred_traj in zip(y_true,predicted):
    #     plot_3D_result(pred_traj, true_traj)
    #     plt.show()

    visual_dof()

    # #for trajectory lists
    # for true_traj,pred_traj in zip(y_true,predicted):
    #     plot_3D_result(pred_traj,true_traj,human_flag=True)
    #     plot_3D_result(pred_traj, true_traj)
    #     plt.show()
    # #plot result for single dof
    # for true_traj, pred_traj in zip(y_true, predicted):
    #     plot_single_dof_result(pred_traj,true_traj)


if __name__ == '__main__':
     main()
