#========================================================================================================
# Joint Source, Channel and Space-Time Coding Based on Deep Learning in MIMO (rician fading)
#
# main.py
#
# 0. import packages
# 1. Settings: DNN structure & MIMO system parameters
# 2. Train, test data
# 3. DNN object
# 4. train
# 5. test
#
#========================================================================================================




#========================================================================================================
#
# 0. import packages
#
#========================================================================================================
import os
import glob
import sys
import csv
import numpy as np
from dnn import Dnn



#========================================================================================================
#
# 1. Settings: DNN structure & MIMO system parameters
#
#========================================================================================================
num_epoch = 150

num_sample_train = 800
num_sample_test = 100

snr_dB_train = np.arange(0, 65, 5)
snr_dB_test = np.arange(0, 65, 5)

K_train = np.arange(0, 13, 4)
K_test = np.arange(0, 13, 4)

lr = np.array([3e-4])

Batch_Size = 20

# antenna (2x2)
Nt = 2
Nr = 2

# R: spectral efficiency, C: spatial multiplexing rate (1: OSTBC, 4/3: DBLAST, 2: VBLAST)
R = np.array([1,2,3,4]).reshape([1,-1])     #1x4
C = np.array([1,4/3,2]).reshape([1,-1])     #1x3

no_pkt = 128

# DNN structure
Layer_dim_list = [16, R.shape[1]*no_pkt+C.shape[1]*no_pkt]

# bits
image_size = 512 * 512 * 1  # 512x512 pixels, 1bpp
maximum_pkt_size = image_size / no_pkt  # max num of bits per packet: 2048

# distortion step
D_STEP = 2**12
STEP = np.arange(0, int(np.ceil(image_size / 64)) + 1, int(D_STEP / 64))    # D_STEP=4096 -> STEP=[0, 4096, 8192, ...]




#========================================================================================================
#
# 2. Train, test data
#
#   data: Distortion-Rate curve
#
#========================================================================================================
i = 0
num_total_sample = 900
input = np.zeros([num_total_sample, STEP.shape[0]])     # input.shape: (num_sample)x65

input_path = sys.argv[0]
input_path = input_path[:-7]

# load train and test data from files
for input_file in glob.glob(os.path.join(input_path, 'distortion_all\distort_0*')):
    with open(input_file, 'r') as f:
        rdr = csv.reader(f, delimiter='\t')
        temp = np.array(list(rdr), dtype=np.float64)
        input[i, :] = temp[STEP, 1]
        i = i + 1

    if i == num_total_sample:
        break

# split data into train and test data
input_train, input_test = np.vsplit(input, [num_sample_train])
# input_train, input_test, input_valid = np.vsplit(input, (num_sample_train, num_sample_train+num_sample_test))




#========================================================================================================
#
# 3. DNN object
#
#========================================================================================================
dnnObj = Dnn(batch_size=Batch_Size, n_epoch=num_epoch, layer_dim_list=Layer_dim_list, num_pkt=no_pkt, D_step = D_STEP, r=R, c=C)




#========================================================================================================
#
# 4. Train
#
#========================================================================================================
dnnObj.train_dnn(input_train, snr_dB_train, K_train, lr[0])



# Warning message: the number of training samples are not a multiple of Batch_Size
if num_sample_train % Batch_Size != 0:
    print('===========================================================\n')
    print('Warning: num_sample_train is not a multiple of Batch_Size!!\n')
    print('===========================================================\n')




#========================================================================================================
#
# 5. Test
#
#========================================================================================================
for k in range(K_test.shape[0]):
    for i in range(snr_dB_test.shape[0]):
        dnnObj.test_dnn(input_test, snr_dB_test[i], K_test[k])