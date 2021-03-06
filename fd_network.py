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
from scipy.stats import randint, expon, uniform

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from math import sqrt
# import keras
import tensorflow as tf

print(tf.__version__)

# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

np.random.seed(0)
tf.random.set_seed(0)


def scheduler(epoch, lr):
    if epoch == 200:
        return lr * 0.1

    else:
        return lr

# def scheduler(epoch, lr):
#     return lr



def gen_net(vec_len, num_hidden1, num_hidden2):
    '''
    TODO: Generate and evaluate any CNN instead of MLPs
    :param vec_len:
    :param num_hidden1:
    :param num_hidden2:
    :return:
    '''

    model = Sequential()
    model.add(Dense(num_hidden1, activation='relu', input_shape=(vec_len,)))
    model.add(Dense(num_hidden2, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # model.add(Dense(10, activation='sigmoid'))

    return model


class network_fit(object):
    '''
    class for network
    '''

    def __init__(self, train_samples, label_array_train, test_samples, label_array_test,
                 model_path, n_hidden1=100, n_hidden2=10, verbose=2):
        '''
        Constructor
        Generate a NN and train
        @param none
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.train_samples = train_samples
        self.label_array_train = label_array_train
        self.test_samples = test_samples
        self.label_array_test = label_array_test
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.model_path = model_path
        self.verbose = verbose

        self.mlps = gen_net(self.train_samples.shape[1], self.n_hidden1, self.n_hidden2)

    def train_net(self, epochs=500, batch_size=500, lr=1e-05, plotting=True):
        '''
        specify the optimizers and train the network
        :param epochs:
        :param batch_size:
        :param lr:
        :return:
        '''
        print("Initializing network...")
        # compile the model
        rp = optimizers.RMSprop(learning_rate=lr, rho=0.9, centered=True)
        adm = optimizers.Adam(learning_rate=lr, epsilon=1)
        sgd_m = optimizers.SGD(learning_rate=lr)
        adam = optimizers.Adam(lr=0.0001)

        lr_scheduler = LearningRateScheduler(scheduler)

        keras_rmse = tf.keras.metrics.RootMeanSquaredError()
        self.mlps.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=["accuracy"])
        # print(self.mlps.summary())
        # print ("self.train_samples.shape", self.train_samples.shape)
        # print ("self.label_array_train.shape", self.label_array_train.shape)

        # Train the model
        history = self.mlps.fit(self.train_samples, self.label_array_train, epochs=epochs, batch_size=batch_size,
                                validation_split=0.3, verbose=0,
                                callbacks=[lr_scheduler,
                                    EarlyStopping(monitor='val_loss', min_delta=0, patience=100,
                                                  verbose=0, mode='min'),
                                    ModelCheckpoint(self.model_path, monitor='val_loss',
                                                    save_best_only=True, mode='min',
                                                    verbose=0)])

        # print(history.history.keys())
        val_rmse_k = history.history['val_loss']
        val_rmse_min = min(val_rmse_k)
        min_val_rmse_idx = val_rmse_k.index(min(val_rmse_k))
        stop_epoch = min_val_rmse_idx + 1
        val_loss_min = round(val_rmse_min, 4)
        print("val_loss_min: ", val_loss_min)

        val_acc_k = history.history['val_accuracy']
        val_acc_max = val_acc_k[min_val_rmse_idx]
        val_acc_max = round(val_acc_max, 4)
        print("val_acc_max: ", val_acc_max)

        fitness_net = (val_loss_min,)

        trained_net = self.mlps

        ## Plot training & validation loss about epochs
        # if plotting == True:
        #     # summarize history for Loss
        #     fig_acc = plt.figure(figsize=(10, 10))
        #     plt.plot(history.history['loss'])
        #     plt.plot(history.history['val_loss'])
        #     plt.title('model loss')
        #     plt.ylabel('loss')
        #     # plt.ylim(0, 2000)
        #     plt.xlabel('epoch')
        #     plt.legend(['train', 'test'], loc='upper left')
        #     plt.show()

        fitness_net

        return trained_net, fitness_net

    def test_net(self, trained_net=None, best_model=True, plotting=True):
        '''
        Evalute the trained network on test set
        :param trained_net:
        :param best_model:
        :param plotting:
        :return:
        '''
        # Load the trained model
        if best_model:
            estimator = load_model(self.model_path)
        else:
            estimator = load_model(trained_net)

        # predict the RUL
        output = estimator.predict(self.test_samples)
        y_true_test = self.label_array_test  # ground truth of test samples

        output_classes = np.argmax(output, axis=1)

        print ("output_classes", output_classes)
        print("y_true_test", y_true_test)
        print("output_classes.shape", output_classes.shape)
        print ("y_true_test.shape", y_true_test.shape)

        y_pred_test = output_classes

        pd.set_option('display.max_rows', 1000)
        test_print = pd.DataFrame()
        test_print['y_pred'] = y_pred_test.flatten()
        test_print['y_truth'] = y_true_test.flatten()

        y_predicted = test_print['y_pred']
        y_actual = test_print['y_truth']
        acc = accuracy_score(y_actual, y_predicted)




        return acc