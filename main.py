#========================================================================================================
# Joint Source, Channel and Space-Time Coding Based on Deep Learning in MIMO (rician fading)
#
# main.py
#
# 0. import packages
# 1. Settings: DNN structure & MIMO system parameters
# 2. Outage probability
# 3. Train, test data
# 4. DNN object
# 5. Train
# 6. Test
#
#========================================================================================================





#========================================================================================================
#
# 0. import packages
#
#========================================================================================================
import os
import glob
import csv
import numpy as np
from .dnn_github_new import Dnn





#========================================================================================================
#
# 1. Settings: DNN structure & MIMO system parameters
#
#========================================================================================================
num_epoch = 1000
epoch_step = 500

num_sample_train = 800
num_sample_test = 100

snr_dB_train = np.arange(0, 65, 5)
snr_dB_test = np.arange(0, 65, 5)

K_train = np.arange(0, 13, 4)
K_test = K_train

lr = np.array([3e-4])
Batch_Size = 20
mode_input_scaling = 2

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
D_STEP = 2 ** 12
STEP = np.arange(0, int(np.ceil(image_size / 64)) + 1, int(D_STEP / 64))    # D_STEP=4096 -> STEP=[0, 4096, 8192, ...]





#========================================================================================================
#
# 2. Outage Probability
#
#========================================================================================================
Pout_all = np.zeros([K_train.shape[0], C.shape[1], 60-(-10)+1, R.shape[1]+1])
Pout_all[:,:,:,0] = np.arange(-10,61,1)

file_list = ['OSTBC', 'DBLAST', 'VBLAST']

for k in range(K_train.shape[0]):       # rician factor
    for i in range(len(file_list)):     # spatial multiplexing rate
        for r in range(R.shape[1]):     # data rate
            filename = '.\data\outage_prob\Pout/' + file_list[i] + '_K' + str(K_train[k]).zfill(2) + '_R' + str(R[0, r]) + '_2x2.dat'
            print(filename)
            with open(filename, 'r') as f:
                rdr = csv.reader(f, delimiter='\t')
                temp = np.array(list(rdr), dtype=np.float64)
                data = temp.reshape([1, 71, 2])
                Pout_all[k,i,:,r+1] = data[0,:,1]





#========================================================================================================
#
# 3. Train, test data
#
#   data: Distortion-Rate curve
#
#========================================================================================================
i = 0
num_total_sample = 900
input = np.zeros([num_total_sample, STEP.shape[0]]) # input.shape: (num_sample)x65

for input_file in glob.glob(os.path.join('data\distortion_DIV2K\distort_0*')):
    with open(input_file, 'r') as f:
        rdr = csv.reader(f, delimiter='\t')
        temp = np.array(list(rdr), dtype=np.float64)
        input[i, :] = temp[STEP, 1]
        i = i + 1

    if i == num_total_sample:
        break

input_train = input[:num_sample_train, :]
input_test = input[-num_sample_test:, :]





#========================================================================================================
#
# 4. DNN object
#
#========================================================================================================
mode_lr_shift = 'disable'

dnnObj = Dnn(batch_size=Batch_Size, mode_input_scale = mode_input_scaling, mode_shuffle = 'disable',
             n_epoch=num_epoch, layer_dim_list=Layer_dim_list, max_pkt_size=maximum_pkt_size, num_pkt=no_pkt, D_step = D_STEP, nr=Nr, nt=Nt, r=R, c=C, mode_lr_shift = mode_lr_shift)





# ========================================================================================================
#
# 5. Train
#
# ========================================================================================================
for j in range(lr.shape[0]):
    file_path = './' + str(num_sample_train) + 'img train(new_sampling)/snr_dB_train=[0,5,...,60]/batch_size=' + str(Batch_Size) + '/hidden_node=16/lr_shift=' + str(mode_lr_shift) + '/lr=' + str(lr[j]) + ',' + str(num_epoch) + 'epoch/'

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(file_path + 'loss_per_epoch.dat', 'w') as f7:
        pass
    with open(file_path + 'D_dnn_train.dat', 'w') as f:
        pass
    with open(file_path + 'R_dnn_train.dat', 'w') as f, open(file_path + 'Cx3_dnn_train.dat', 'w') as f3:
        pass

    dnnObj.train_dnn(input_train, snr_dB_train, K_train, lr[j], Pout_all, file_path, epoch_step)


if num_sample_train % Batch_Size != 0:
    print('===========================================================\n')
    print('Warning: num_sample_train is not a multiple of Batch_Size!!\n')
    print('===========================================================\n')





#========================================================================================================
#
# 6. Test
#
#========================================================================================================
for j in range(lr.shape[0]):
    file_path = './' + str(num_sample_train) + 'img train(new_sampling)/snr_dB_train=[0,5,...,60]/batch_size=' + str(Batch_Size) + '/hidden_node=16/lr_shift=' + str(mode_lr_shift) + '/lr=' + str(lr[j]) + ',' + str(num_epoch) + 'epoch/'

    with open(file_path + "/" + 'D_dnn_test.dat', 'w') as f:
        pass
    with open(file_path + "/" + 'R_dnn_test.dat', 'w') as f, open(file_path + "/" + 'Cx3_dnn_test.dat', 'w') as f3:
        pass

    for i in range(snr_dB_test.shape[0]):
        for k in range(K_test.shape[0]):
            snr_idx = np.where(Pout_all[0, 0, :, 0] == snr_dB_test[i])
            Pout_all_snr = np.squeeze(Pout_all[k, :, snr_idx[0], 1:])
            dnnObj.test_dnn(input_test, snr_dB_test[i], K_test[k], lr[j], Pout_all_snr, file_path)
