from __future__ import print_function
import numpy as np
import pickle

def mse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())

def dist(pred,true):
    '''
    calculate square root distance error along a trajectory
    '''
    dist = []
    for p1, p2 in zip(pred, true):
        dist.append(np.sqrt(np.sum((p1-p2)**2)))
    dist = np.asarray(dist)
    return dist.mean(),dist


def step_dist_err(pred,true):
    '''

    :param pred: single prediction traj
    :param true: single true traj
    :return:
    dist_list: error of each timestep of prediction
    mean_error_list: mean error of every step
    '''
    dist_list=[]
    out_len = len(pred[0])
    for d1, d2 in zip(pred, true):
        temp = []
        for p1, p2 in zip(d1, d2):
            distance = np.sqrt(np.sum((p1-p2)**2))
            temp.append(distance)
        temp=np.asarray(temp)
        dist_list.append(temp)

    dist_list=np.asarray(dist_list)

    mean_error_list = []
    for i in range(0,out_len):
        test = dist_list[:,i]
        mean_error_list.append(test.mean())

    return mean_error_list,dist_list

def main():
    # y_true = pickle.load(open("../results/raw_true_trajs.pkl", "rb"))
    # predicted = pickle.load(open("../results/raw_pred_trajs.pkl","rb"))
    #
    # for true_traj,pred_traj in zip(y_true,predicted):
    #     mean_dist,dist_list = dist(pred_traj,true_traj)

    y_true = pickle.load(open("test_y_true.pkl", "rb"))
    predicted = pickle.load(open("test_y_predicted.pkl","rb"))

    mean_error_list, dist_list = step_dist_err(predicted, y_true)

    print("1")



if __name__ == '__main__':
     main()
