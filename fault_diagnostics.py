"""


"""

## Import libraries in python
import argparse
import time
import json
import logging as log
import sys

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from math import sqrt
from scipy.stats import randint, expon, uniform

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pyts.approximation import SymbolicFourierApproximation
from sklearn import preprocessing

import matplotlib.pyplot as plt

from cwru_custom import CWRU
from fd_network import network_fit




def pca(train_vec_samples, test_vec_samples, n_components=100):
    '''
    Apply PCA to reduce dimensionality of input vector.
    :param train_vec_samples:
    :param test_vec_samples:
    :param n_components:
    :return:
    '''

    pca = PCA(n_components=n_components)
    pca.fit(train_vec_samples)

    pca_train_samples = pca.transform(train_vec_samples)
    # print("rp_pca_train_samples.shape: ", pca_train_samples.shape)

    pca_test_samples = pca.transform(test_vec_samples)
    # print("rp_pca_test_samples.shape: ", pca_test_samples.shape)

    return pca_train_samples, pca_test_samples


def sfa(train_vec_samples, test_vec_samples, n_components=100, n_bins=10, alphabet='ordinal'):
    '''
    Apply SFA to reduce dimensionality of input vector.
    :param train_vec_samples:
    :param test_vec_samples:
    :param n_components:
    :param n_bins:
    :param alphabet:
    :return:
    '''

    sfa = SymbolicFourierApproximation(n_coefs=n_components, n_bins=n_bins, alphabet=alphabet)
    sfa.fit(train_vec_samples)

    sfa_train_samples = sfa.transform(train_vec_samples)
    # print("sfa_train_samples.shape: ", sfa_train_samples.shape)

    sfa_test_samples = sfa.transform(test_vec_samples)
    # print("sfa_test_samples.shape: ", sfa_test_samples.shape)

    return sfa_train_samples, sfa_test_samples


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'temp_net.h5')

    parser = argparse.ArgumentParser(description='fault diagnostics CWRU')
    parser.add_argument('-i', type=int, help='Input sources', required=True)
    parser.add_argument('-l', type=int, default=100, help='sequence length')
    parser.add_argument('--cross', type=str, default='no', help='cross val')
    parser.add_argument('--dim_method', type=str, default='non', help='dim reduction method')
    parser.add_argument('--n_comp', type=int, default=100, help='number of components of dim reduction method')
    # parser.add_argument('--thres_type', type=str, default='distance', required=False,
    #                     help='threshold type for RPs: distance or point ')
    # parser.add_argument('--thres_value', type=int, default=50, required=False,
    #                     help='percentage of maximum distance or black points for threshold')
    parser.add_argument('--n_hidden1', type=int, default=200, required=False,
                        help='number of neurons in the first hidden layer')
    parser.add_argument('--n_hidden2', type=int, default=100, required=False,
                        help='number of neurons in the second hidden layer')
    parser.add_argument('--epochs', type=int, default=1000, required=False, help='number epochs for network training')
    parser.add_argument('--batch', type=int, default=500, required=False, help='batch size of BPTT training')
    parser.add_argument('--verbose', type=int, default=2, required=False, help='Verbose TF training')
    # parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    # parser.add_argument('--gen', type=int, default=100, required=False, help='generations of evolution')
    parser.add_argument('--plotting', type=str, default='yes', help='plotting network training histroy')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on cpu or cuda.')

    args = parser.parse_args()
    cross = args.cross
    plotting = args.plotting

    if args.i == 48:
        frq = "48DriveEndFault"
    elif args.i == 12:
        frq = "12DriveEndFault"

    if cross == 'yes':
        cross = True
    elif cross == 'no':
        cross = False


    if plotting == 'yes':
        plotting = True
    elif plotting == 'no':
        plotting = False


    seq_length = args.l
    dim_method = args.dim_method
    n_components = args.n_comp
    n_hidden1 = args.n_hidden1
    n_hidden2 = args.n_hidden2
    epochs = args.epochs
    batch = args.batch
    verbose = args.verbose

    if cross == False:
        data_hp1 = CWRU(frq, "1772", seq_length, 1, cross)
        data_hp2 = CWRU(frq, "1750", seq_length, 1, cross)
        data_hp3 = CWRU(frq, "1730", seq_length, 1, cross)
        data_lst = [data_hp1, data_hp2, data_hp3]
    elif cross == True:
        data_hp1_lst = []
        data_hp2_lst = []
        data_hp3_lst = []
        for split in range(10):
            data_hp1_lst.append(CWRU(frq, "1772", seq_length, split+1, cross))
            data_hp2_lst.append(CWRU(frq, "1750", seq_length, split+1, cross))
            data_hp3_lst.append(CWRU(frq, "1730", seq_length, split+1, cross))
        data_lst_lst = [data_hp1_lst, data_hp2_lst, data_hp3_lst]


    # print ("data_hp1.X_train.shape", data_hp1.X_train.shape)
    # print ("len(data_hp1.y_train)", len(data_hp1.y_train))
    # print ("data_hp1.X_test.shape", data_hp1.X_test.shape)
    # # print ("data.y_test", data.y_test)
    # print ("len(data_hp1.y_test)", len(data_hp1.y_test))
    # print ("len(data_hp1.labels)", len(data_hp1.labels))
    # print ("data_hp1.labels", data_hp1.labels)
    # print ("data_hp1.nclasses", data_hp1.nclasses)

    if cross == False:
        acc_lst = []
        for idx, data in enumerate(data_lst):
            train_samples = data.X_train
            test_samples = data.X_test
            label_array_train = np.asarray(data.y_train)
            label_array_test = np.asarray(data.y_test)
            label_array_train = np.reshape(label_array_train, (label_array_train.shape[0],1))
            # label_array_test = np.reshape(label_array_test, (label_array_test.shape[0], 1))
            ohe = preprocessing.OneHotEncoder()
            ohe.fit(label_array_train)
            label_array_train = ohe.transform(label_array_train).toarray()
            # label_array_test = ohe.transform(label_array_test).toarray()


            if dim_method == 'non':
                pass
            elif dim_method =='sfa':
                train_samples, test_samples = sfa(train_samples, test_samples, n_components=n_components, n_bins=10)
            elif dim_method == 'pca':
                train_samples, test_samples = pca(train_samples, test_samples, n_components=n_components)


            # #preprocessing
            # min_max_scaler = preprocessing.MinMaxScaler()
            # train_samples = min_max_scaler.fit_transform(train_samples)
            # test_samples = min_max_scaler.transform(test_samples)

            mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                                   model_path=model_path, n_hidden1=n_hidden1, n_hidden2=n_hidden2, verbose=verbose)

            trained_net = mlps_net.train_net(epochs=epochs, batch_size=batch, plotting=plotting)
            acc = mlps_net.test_net(trained_net)

            print ("accuracy of data_hp_%s: " %(idx+1), acc )
            acc_lst.append(acc)

        print("accuracies: ", acc_lst)
        print("avg accuracy: ", sum(acc_lst) / len(acc_lst))


    elif cross == True:
        acc_lst_lst = []
        for lst in data_lst_lst:
            acc_lst = []
            for idx, data in enumerate(lst):
                train_samples = data.X_train
                test_samples = data.X_test
                label_array_train = np.asarray(data.y_train)
                label_array_test = np.asarray(data.y_test)
                label_array_train = np.reshape(label_array_train, (label_array_train.shape[0], 1))
                # label_array_test = np.reshape(label_array_test, (label_array_test.shape[0], 1))
                ohe = preprocessing.OneHotEncoder()
                ohe.fit(label_array_train)
                label_array_train = ohe.transform(label_array_train).toarray()
                # label_array_test = ohe.transform(label_array_test).toarray()

                if dim_method == 'non':
                    pass
                elif dim_method == 'sfa':
                    train_samples, test_samples = sfa(train_samples, test_samples, n_components=n_components, n_bins=10)
                elif dim_method == 'pca':
                    train_samples, test_samples = pca(train_samples, test_samples, n_components=n_components)

                mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                                       model_path=model_path, n_hidden1=n_hidden1, n_hidden2=n_hidden2, verbose=verbose)

                trained_net = mlps_net.train_net(epochs=epochs, batch_size=batch, plotting= plotting)
                acc = mlps_net.test_net(trained_net)

                print("accuracy of data_hp_%s: " %(idx+1), acc)
                acc_lst.append(acc)

            acc_lst_lst.append(acc_lst)

        results_lst = []
        for idx, acc_hp in enumerate(acc_lst_lst):
            print("HP_%s all accuracies per split: " %(idx+1), acc_hp)
            avg_hp = sum(acc_hp) / len(acc_hp)
            results_lst.append(avg_hp)

        print("per HP avg accuracy: ", results_lst)
        print("overall avg accuracy: ", sum(results_lst) / len(results_lst))






if __name__ == '__main__':
    main()
