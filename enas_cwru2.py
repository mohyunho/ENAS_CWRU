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
from fd_task import dim
from fd_task import SimpleNeuroEvolutionTask
from ea import GeneticAlgorithm


jobs = 1


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'temp_net2.h5')

    ealog_folder = os.path.join(current_dir, 'EA_log')
    if not os.path.exists(ealog_folder):
        os.makedirs(ealog_folder)



    parser = argparse.ArgumentParser(description='fault diagnostics CWRU')
    parser.add_argument('-i', type=int, help='Input sources', required=True)
    parser.add_argument('--hp', type=int, default=1, help='motor load for EA')
    parser.add_argument('-l', type=int, default=400, help='sequence length')
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
    parser.add_argument('--epochs', type=int, default=500, required=False, help='number epochs for network training')
    parser.add_argument('--batch', type=int, default=500, required=False, help='batch size of BPTT training')
    parser.add_argument('--verbose', type=int, default=2, required=False, help='Verbose TF training')
    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')
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

    hp_list = ['', "1772", "1750", "1730"]
    hp_idx = args.hp
    hp = hp_list[hp_idx]



    ## Parameters for the GA
    pop_size = args.pop  # toy example
    n_generations = args.gen  # toy example
    cx_prob = 0.5  # 0.25
    mut_prob = 0.5  # 0.7
    cx_op = "one_point"
    mut_op = "uniform"
    sel_op = "best"
    other_args = {
        'mut_gene_probability': 0.3  # 0.1
    }




    # if cross == False:
    #     data_hp1 = CWRU(frq, "1772", seq_length, 1, cross)
    #     data_hp2 = CWRU(frq, "1750", seq_length, 1, cross)
    #     data_hp3 = CWRU(frq, "1730", seq_length, 1, cross)
    #     data_lst = [data_hp1, data_hp2, data_hp3]
    # elif cross == True:
    #     data_hp1_lst = []
    #     data_hp2_lst = []
    #     data_hp3_lst = []
    #     for split in range(10):
    #         data_hp1_lst.append(CWRU(frq, "1772", seq_length, split+1, cross))
    #         data_hp2_lst.append(CWRU(frq, "1750", seq_length, split+1, cross))
    #         data_hp3_lst.append(CWRU(frq, "1730", seq_length, split+1, cross))
    #     data_lst_lst = [data_hp1_lst, data_hp2_lst, data_hp3_lst]





    mutate_log_path = os.path.join(ealog_folder, 'mute_log_%s_%s_%s.csv' % (str(dim_method), pop_size, n_generations))


    mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'fitness', 'gen']
    mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
    mutate_log_df.to_csv(mutate_log_path, index=False)

    def log_function(population, gen, mutate_log_path=mutate_log_path):
        for i in range(len(population)):
            if population[i] == []:
                "non_mutated empty"
                pass
            else:
                # print ("i: ", i)
                population[i].append(population[i].fitness.values[0])
                population[i].append(gen)

        temp_df = pd.DataFrame(np.array(population), index=None)
        temp_df.to_csv(mutate_log_path, mode='a', header=None)
        print("population saved")
        return

    start = time.time()

    # Assign & run EA
    task = SimpleNeuroEvolutionTask(
        frq = frq,
        hp = hp,
        seq_length = seq_length,
        dim_method = dim_method,
        model_path=model_path,
        epochs=epochs,
        batch=batch
    )

    # aic = task.evaluate(individual_seed)

    ga = GeneticAlgorithm(
        task=task,
        population_size=pop_size,
        n_generations=n_generations,
        cx_probability=cx_prob,
        mut_probability=mut_prob,
        crossover_operator=cx_op,
        mutation_operator=mut_op,
        selection_operator=sel_op,
        jobs=jobs,
        log_function=log_function,
        **other_args
    )

    pop, log, hof = ga.run()

    print("Best individual:")
    print(hof[0])

    # Save to the txt file
    # hof_filepath = tmp_path + "hof/best_params_fn-%s_ps-%s_ng-%s.txt" % (csv_filename, pop_size, n_generations)
    # with open(hof_filepath, 'w') as f:
    #     f.write(json.dumps(hof[0]))

    print("Best individual is saved")
    end = time.time()
    print("EA time: ", end - start)

    """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
    print("Evaluate the best individual")

    ## Test the best individual
    data_hp1 = CWRU(frq, "1772", seq_length, 1, cross)
    data_hp2 = CWRU(frq, "1750", seq_length, 1, cross)
    data_hp3 = CWRU(frq, "1730", seq_length, 1, cross)
    data_lst = [data_hp1, data_hp2, data_hp3]

    acc_lst = []
    for idx, data in enumerate(data_lst):
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
            train_samples, test_samples = dim.sfa(train_vec_samples=train_samples, test_vec_samples=test_samples,
                                                  n_components=hof[0][0], n_bins=25, alphabet='ordinal')
        elif dim_method == 'pca':
            train_samples, test_samples = dim.pca(train_vec_samples=train_samples, test_vec_samples=test_samples,
                                                  n_components=hof[0][0])


        mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                               model_path=model_path, n_hidden1=hof[0][1], n_hidden2=hof[0][2], verbose=verbose)

        trained_net = mlps_net.train_net(epochs=epochs, batch_size=batch, plotting=plotting)
        acc = mlps_net.test_net(trained_net)

        print("accuracy of data_hp_%s: " % (idx + 1), acc)
        acc_lst.append(acc)

    print("accuracies: ", acc_lst)
    print("avg accuracy: ", sum(acc_lst) / len(acc_lst))




if __name__ == '__main__':
    main()