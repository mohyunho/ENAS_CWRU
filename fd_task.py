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

from cwru_custom import CWRU
from fd_network import network_fit

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
    def __init__(self, frq, hp, model_path, epochs, batch):
        self.frq = frq
        self.hp = hp
        self.model_path = model_path
        self.epochs = epochs
        self.batch = batch


    def get_n_parameters(self):
        return 3

    def get_parameters_bounds(self):

        bounds = [
            (10,50),
            (1, 50),
            (1, 50),
        ]

        return bounds

    def evaluate(self, genotype):
        print ("genotype", genotype)
        # print ("len(genotype)", len(genotype))

        """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
        data = CWRU(self.frq, self.hp, length = genotype[0]*10, split=1)

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

        n_1 = genotype[1]*10
        n_2 = genotype[2]*10

        mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                               self.model_path,
                               n_hidden1=n_1 ,
                               n_hidden2=n_2 if n_2 < n_1  else n_1 )

        trained_net, fitness = mlps_net.train_net(epochs=self.epochs, batch_size=self.batch)


        return fitness

