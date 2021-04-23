#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual
Author:
Date:
"""
import pandas as pd
import numpy as np
from abc import abstractmethod
from sklearn import preprocessing

from sklearn.decomposition import PCA
from pyts.approximation import SymbolicFourierApproximation

from cwru_custom import CWRU
from fd_network import network_fit


class dim:
    @staticmethod
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

    @staticmethod
    def sfa(train_vec_samples, test_vec_samples, n_components=100, n_bins=25, alphabet='ordinal'):
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




class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass


class SimpleNeuroEvolutionTask(Task):
    def __init__(self, frq, hp, seq_length, dim_method, model_path, epochs, batch):
        self.frq = frq
        self.hp = hp
        self.seq_length = seq_length,
        self.dim_method = dim_method,
        self.model_path = model_path
        self.epochs = epochs
        self.batch = batch



    def get_n_parameters(self):
        return 3

    def get_parameters_bounds(self):

        bounds = [
            (1, 30),
            (1, 50),
            (1, 50),
        ]

        return bounds

    def evaluate(self, genotype):
        print ("genotype", genotype)
        # print ("len(genotype)", len(genotype))

        """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
        # print ("self.seq_length[0]",self.seq_length[0])
        data = CWRU(self.frq, self.hp, length = self.seq_length[0], split=1)

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

        if self.dim_method == 'non':
            pass
        elif self.dim_method == 'sfa':
            train_samples, test_samples = dim.sfa(train_vec_samples=train_samples, test_vec_samples=test_samples,
                                                  n_components=genotype[0]*10, n_bins=25, alphabet='ordinal')
        elif self.dim_method == 'pca':
            train_samples, test_samples = dim.pca(train_vec_samples=train_samples, test_vec_samples=test_samples,
                                                  n_components=genotype[0]*10)


        n_1 = genotype[1]*10
        n_2 = genotype[2]*10

        mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                               self.model_path,
                               n_hidden1=n_1 ,
                               n_hidden2=n_2 if n_2 < n_1  else n_1 )

        trained_net, fitness = mlps_net.train_net(epochs=self.epochs, batch_size=self.batch)


        return fitness

