import os
import glob
import sys
import errno
import random
import urllib.request

import numpy as np
# a = np.array([[1,2],[3,4],[5,6]])
# print (a.shape)



import numpy as np
from scipy.io import loadmat
rdir = os.path.join(os.path.dirname(__file__), 'Datasets/CWRU')

#1750
normal_mat = os.path.join(rdir, 'NormalBaseline/1750', 'Normal.mat')
fault_dir = os.path.join(rdir, '48DriveEndFault/1750')
fault_7b_mat = os.path.join(fault_dir, '0.007-Ball.mat')
fault_7i_mat = os.path.join(fault_dir, '0.007-InnerRace.mat')
fault_7o_mat = os.path.join(fault_dir, '0.007-OuterRace6.mat')

mat_list = [normal_mat, fault_7b_mat, fault_7i_mat, fault_7o_mat]
mat_array_lst = []

for mat_paht in mat_list :
    print (normal_mat)
    mat_dict = loadmat(fault_7o_mat)
    key = list(filter(lambda x: 'DE_time' in x, mat_dict.keys()))[0]
    time_series = mat_dict[key][:400000, 0]
    print (time_series)
    print (time_series.shape)
    mat_array_lst.append (time_series)


# full_array =  np.concatenate(mat_array_lst, axis=1)

full_array =  np.stack(mat_array_lst, axis=-1)

print (full_array)
print (full_array.shape)
first_low  = "Normal, Fault_Ball, Fault_InnerRace, Fault_OuterRace"

np.savetxt("CWRU_bearing_1750.csv", full_array, delimiter=',', header=first_low, comments="")