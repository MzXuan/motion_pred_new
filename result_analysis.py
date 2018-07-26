from __future__ import print_function
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import os
import ConfigParser
import csv

from utils import traj_gen
from utils import visualization
from utils import calculate_mse

from scipy.interpolate import griddata
import flags

FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
  os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
  os.makedirs(FLAGS.sample_dir)


# the current file path
FILE_PATH = os.path.dirname(__file__)
TASK_NAME_LIST = []


data_path = os.path.join(FILE_PATH, './model', FLAGS.model_name)

## load result
true_all_result = pickle.load(open(data_path+"/true_all_result.pkl","rb"))
pred_all_result = pickle.load(open(data_path+"/pred_all_result.pkl", "rb"))
x_all_result = pickle.load(open(data_path+"/x_all_result.pkl", "rb"))


print(data_path+"/true_all_result.pkl")
print(data_path+"/pred_all_result.pkl")

print("load successfully!")

def gen_trajs():
    # generate trajectories
    true_all_trajs = []
    pred_all_trajs = []
    x_all_trajs = []
    cls = 0
    for task_true, task_pred, x_true in zip(true_all_result,pred_all_result,x_all_result):
        print("now generating class: ", cls)
        cls+=1
        temp_t =[]
        temp_p = []
        temp_x = []
        for traj_true, traj_pred, traj_x in zip(task_true,task_pred,x_true):
            temp_t.append(traj_gen.traj_generation(traj_true, step = 1))
            temp_p.append(traj_gen.traj_generation(traj_pred, step = 1))
            temp_x.append(traj_gen.traj_generation(traj_x, step = 1))

        true_all_trajs.append(temp_t)
        pred_all_trajs.append(temp_p)
        x_all_trajs.append(temp_x)

    print("generate trajectories successfully!")
    pickle.dump(true_all_trajs,open(data_path+"/true_all_trajs.pkl","wb"))
    pickle.dump(pred_all_trajs,open(data_path+"/pred_all_trajs.pkl","wb"))
    pickle.dump(x_all_trajs, open(data_path + "/x_all_trajs.pkl", "wb"))

    return true_all_trajs,pred_all_trajs, x_all_trajs



def visual_dof(true_all_trajs,pred_all_trajs, class_list):
    #visualize for single dof
    j=0
    for cls_num in class_list:
        for traj_t,traj_p in zip(true_all_trajs[cls_num],pred_all_trajs[cls_num]):
            visualization.plot_dof_result(traj_p,traj_t,j,save_flag = True)
            plt.show()
            j+=1


def visual_3d(true_all_ee_trajs, pred_all_ee_trajs, x_all_trajs, class_list = [0]):
    #visualize for single dof
    for i in range(0,3):
        visualization.plot_3D_human(x_all_trajs[i][4])
        fig = visualization.plot_3D_result(pred_all_ee_trajs[i][4], true_all_ee_trajs[i][4], save_flag=False)

    fig.savefig("/home/xuan/Dropbox/PhD/Humanoids/result/example_trajs.eps")
    plt.grid()
    plt.show()


    # j=0
    # for cls_num in class_list:
    #     for traj_t,traj_p, x_traj in zip(true_all_ee_trajs[cls_num],pred_all_ee_trajs[cls_num], x_all_trajs[cls_num]):
    #         visualization.plot_3D_human(x_traj, num =j)
    #         visualization.plot_3D_result(traj_p,traj_t,num = j,save_flag = False)
    #         plt.show()
    #         j+=1


def normal_ratio_err(err_list):
    traj_len = len(err_list)

    xp = np.linspace(0,traj_len, traj_len)
    intrpl_x = np.linspace(0, traj_len, 100)

    intrpl_err = griddata(xp, err_list, intrpl_x, method='linear')

    return intrpl_err


def error_traj(true_all,pred_all,class_list = [0], js = True):
    global DOF_FLAG
    # calculate mse for different class
    all_err_list = []
    j = 0
    for cls_num in class_list:
        mean_err_list = []
        for traj_t,traj_p in zip(true_all[cls_num],pred_all[cls_num]):
            #call the trajectory error calculation in mse file
            if DOF_FLAG == -1: # NORMAL CHECK
                if js==True:
                    mean_err, err_list = calculate_mse.dist(traj_p, traj_t)
                else:
                    mean_err, err_list = calculate_mse.dist(traj_p[:,0:3], traj_t[:,0:3])
            else:
                mean_err, err_list = calculate_mse.dist(traj_p[:, DOF_FLAG], traj_t[:, DOF_FLAG])

            intrpl_err = normal_ratio_err(err_list)
            intrpl_err = np.expand_dims(intrpl_err,axis=1)

            # for error along a trajectorypython
            try:
                sum_err_list
            except:

                sum_err_list = intrpl_err
            else:

                sum_err_list = np.concatenate((sum_err_list,intrpl_err),axis=1)

            # for average error of a whole trajectory
            all_err_list.append(mean_err)
            mean_err_list.append(mean_err)

            j+=1

        # calculate the error of one class
        mean_err_list=np.asarray(mean_err_list)

        avg_err = mean_err_list.mean()
        std_err = np.std(mean_err_list)
        # print(mean_err_list,std_err)
        # print("class: ", cls_num, "has average trajectory error: ", avg_err)

    avg_sum_err_list = np.average(sum_err_list, axis=1)

    std_sum_err_list = np.std(sum_err_list, axis=1)
    print("write average error to csv file")


    all_err_list = np.asarray(all_err_list)
    print("data_path: ", data_path)
    print("average err of all lists: ", all_err_list.mean(),"std: ", np.std(all_err_list))
    return avg_sum_err_list, std_sum_err_list



def pipeline(ipromp, js):
    global data_path
    global DOF_FLAG

    class_whole_list =[range(0, 21),[0, 3, 4, 5, 6, 7, 8],[1, 9, 10, 11, 12, 13, 14],[2, 15, 16, 17, 18, 19, 20]]
    # class_whole_list = [[0,1,2],[0],[1],[2]]
    # class_list = range(0, 21)
    # class_list = [0, 3, 4, 5, 6, 7, 8] #task 1
    # class_list = [1, 9, 10, 11, 12, 13, 14] #task2
    # class_whole_list = [[2, 15, 16, 17, 18, 19, 20]] #task3



    if ipromp == True:
        class_list = [0, 1, 2]
        data_path = "./ipromp_result"
        try:
            true_all_trajs = pickle.load(open(data_path+"/true_all_trajs.pkl","rb"))
            pred_all_trajs = pickle.load(open(data_path+"/pred_all_trajs.pkl","rb"))
            # x_all_trajs = pickle.load(open(data_path + "/x_all_trajs.pkl", "rb"))
        except:
            print("loading error")
            return

    else:
        class_list = range(0, 21)
        try:
            true_all_trajs = pickle.load(open(data_path+"/true_all_trajs.pkl","rb"))
            pred_all_trajs = pickle.load(open(data_path+"/pred_all_trajs.pkl","rb"))
            x_all_trajs = pickle.load(open(data_path + "/x_all_trajs.pkl", "rb"))
        except:
            print("generating trajectories, this may take several minutes..")
            true_all_trajs, pred_all_trajs, x_all_trajs = gen_trajs()

    for class_list in class_whole_list:
        if (js == True):
            ## if using Cartesian space, use the code below
            avg_sum_err_list, std_sum_err_list = error_traj(true_all_trajs, pred_all_trajs, class_list, js = True)

            if DOF_FLAG == -1:

                with open(data_path + '/js_error_list.csv', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(avg_sum_err_list)
                    spamwriter.writerow(std_sum_err_list)

            else:
                with open(data_path + '/dof_error_list.csv', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(avg_sum_err_list)
                    spamwriter.writerow(std_sum_err_list)


            visual_dof(true_all_trajs, pred_all_trajs, class_list)
            # visual_3d(true_all_trajs,pred_all_trajs, x_all_trajs, class_list=class_list)

        else:
            ## if not using joint space, use the code below
            true_all_ee_trajs = pickle.load(open(data_path + "/true_all_ee_trajs.pkl", "rb"))
            pred_all_ee_trajs = pickle.load(open(data_path + "/pred_all_ee_trajs.pkl", "rb"))

            # visual_dof(true_all_trajs, pred_all_trajs, class_list)
            avg_sum_err_list, std_sum_err_list = error_traj(true_all_ee_trajs, pred_all_ee_trajs, class_list, js = False)

            with open(data_path + '/pos_error_list.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar=',', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(avg_sum_err_list)
                spamwriter.writerow(std_sum_err_list)
            visual_3d(true_all_ee_trajs, pred_all_ee_trajs, x_all_trajs, class_list=class_list)

    print("finish")


def main():
    global DOF_FLAG
    global data_path
    DOF_FLAG = -1 #-1: normal test

    ipromp = False
    js = True

    pipeline(ipromp,js)

    # for DOF_FLAG in range(0,7):
    #     pipeline(ipromp, js)



if __name__=='__main__':
    main()



