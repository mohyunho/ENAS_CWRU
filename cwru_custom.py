import os
import glob
import sys
import errno
import random
import urllib.request


import numpy as np
from scipy.io import loadmat


class CWRU:

    def __init__(self, exp, rpm, length, split, cross=False):
        if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
            print ("wrong experiment name: {}".format(exp))
            exit(1)
        if rpm not in ('1797', '1772', '1750', '1730'):
            print ("wrong rpm value: {}".format(rpm))
            exit(1)
        # root directory of all data
        # rdir = os.path.join(os.path.expanduser('~'), 'Datasets/CWRU')
        rdir = os.path.join(os.path.dirname(__file__), 'Datasets/CWRU')

        fmeta = os.path.join(os.path.dirname(__file__), 'metadata.txt')
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            # print ("l[2]", l[2])
            # if (l[0] == exp or l[0] == 'NormalBaseline') and l[1] == rpm and l[2] != 0.028 :
            if (l[0] == exp or l[0] == 'NormalBaseline') and l[1] == rpm and not l[2].startswith('0.028') \
                    and not l[2].endswith('OuterRace12') and not l[2].endswith('OuterRace3'):
                lines.append(l)
        # print ("lines", lines)

        self.cross = cross
        self.split = split
        self.length = length  # sequence length
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        self._shuffle()
        self.labels = tuple(line[2] for line in lines)
        self.nclasses = len(self.labels)  # number of classes


    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print ("can't create directory '{}''".format(path))
                exit(1)

    def _download(self, fpath, link):
        print ("Downloading to: '{}'".format(fpath))
        print (link)
        print (fpath)
        urllib.request.urlretrieve(link, fpath)

    def _load_and_slice_data(self, rdir, infos):
        self.X_train = np.zeros((0, self.length))
        self.X_test = np.zeros((0, self.length))
        self.y_train = []
        self.y_test = []
        for idx, info in enumerate(infos):
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))

            mat_dict = loadmat(fpath)
            # key = filter(lambda x: 'DE_time' in x, mat_dict.keys())[0]
            key = list(filter(lambda x: 'DE_time' in x, mat_dict.keys()))[0]
            time_series = mat_dict[key][:, 0]

            idx_last = -(time_series.shape[0] % self.length)
            clips = time_series[:idx_last].reshape(-1, self.length)
            # print ("clips.shape", clips.shape)

            if self.cross:
                n = clips.shape[0]
                n_split = int(n * 9/10)
                # print ("n_split", n_split)
                if self.split == 1:
                    self.X_train = np.vstack((self.X_train, clips[int(n * 1/10)+1:]))
                elif self.split == 10:
                    self.X_train = np.vstack((self.X_train, clips[:int(n * 9 / 10)]))
                else:
                    self.X_train = np.vstack((self.X_train, clips[:int(n * (self.split-1)/10)]))
                    self.X_train = np.vstack((self.X_train, clips[int(n * (self.split)/10):]))

                self.X_test = np.vstack(
                        (self.X_test, clips[int(n * (self.split - 1) / 10): int(n * (self.split) / 10)]))

                # print ("self.X_test.shape", self.X_test.shape)
                if self.split == 2:
                    self.y_train += [idx] * (n_split +1)
                    self.y_test += [idx] * (clips.shape[0] - (n_split) )
                elif self.split == 4 or 6:
                    self.y_train += [idx] * (n_split +2)
                    self.y_test += [idx] * (clips.shape[0] - (n_split) )
                else:
                    self.y_train += [idx] * n_split
                    self.y_test += [idx] * (clips.shape[0] - n_split)

                # print("len(self.y_test)", len(self.y_test))


            else:
                n = clips.shape[0]
                n_split = int(n * 9 / 10)
                # print ("n_split", n_split)
                self.X_train = np.vstack((self.X_train, clips[:n_split]))
                self.X_test = np.vstack((self.X_test, clips[n_split:]))
                self.y_train += [idx] * n_split
                self.y_test += [idx] * (clips.shape[0] - n_split)


    def _shuffle(self):
        # shuffle training samples
        # print ("len(self.y_train)", len(self.y_train))

        index = list(range(self.X_train.shape[0]))

        # print ("len(index)", len(index))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = tuple(self.y_train[i] for i in index)

        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = tuple(self.y_test[i] for i in index)
