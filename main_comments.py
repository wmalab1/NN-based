#========================================================================================================
# Joint Source, Channel and Space-Time Coding Based on Deep Learning in MIMO (rician fading)
#--------------------------------------------------------------------------------------------------------
# 설명
#   지도 학습을 통해 최적 spectral efficiency와 spatial multiplexing rate을 찾는다.
#--------------------------------------------------------------------------------------------------------
#
# main.py
#
# 0. import packages
# 1. Settings: DNN structure & MIMO system parameters
# 2. Outage Probability
# 3. Train, test data
# 4. DNN 객체
# 5. train 관련 파일 초기화
# 6. Train
# 7. test 관련 파일 초기화
# 8. Test
#
#========================================================================================================




#========================================================================================================
#
# 0. import packages
#
#========================================================================================================
# import ctypes
# hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")
# hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cublas64_100.dll")
import os
import glob
import sys
import csv
import argparse
import time

import numpy as np
import matplotlib.pylab as plt

from dnn import Dnn

# 파이참에서 plot할 경우, 아래와 같이 옵션 설정 가능
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14,
                     'legend.fontsize': 11, 'lines.linewidth': 2,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'mathtext.fontset': 'cm',
                     'axes.unicode_minus': False
                     })
plt.rcParams['axes.unicode_minus'] = False   # 한글 사용 시, 축의 음수값 display 깨짐 현상(unicode 문제) 개선

# 그림에 한글 깨지지 않게 하려면 아래 수행
from matplotlib import font_manager, rc
path = 'C://Windows/Fonts/malgun.ttf'  # 맑은 고딕
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)
plt.rcParams["figure.figsize"] = (10,6.5)

import pprint
pp = pprint.PrettyPrinter(indent=4)




#========================================================================================================
#
# 1. Settings: DNN structure & MIMO system parameters
#
#========================================================================================================
# Define Argument Parser
parser = argparse.ArgumentParser()  # 배치 파일로 실행할 때, 인자값을 받을 수 있는 인스턴스 생성

# 입력 받을 인자 추가
parser.add_argument(
    '--pycharm_run',
    type=str,
    default='enable',
    help='pycharm_run'
)
parser.add_argument(
    '--num_epoch',
    type=int,
    default=500,
    help='Epoch'
)
parser.add_argument(
    '--num_sample_train',
    type=int,
    default=800,
    help='Number of training data'
)
parser.add_argument(
    '--num_sample_test',
    type=int,
    default=100,
    help='Number of testing data'
)
parser.add_argument(
    '--snr_dB_train',
    nargs='+', type=int,
    default=[],
    help='SNR train (dB scale)'
)
parser.add_argument(
    '--snr_dB_test',
    nargs='+', type=int,
    default=[],
    help='SNR test (dB scale)'
)
parser.add_argument(
    '--K_train',
    nargs='+', type=int,
    default=[],
    help='Rician Factor train'
)
parser.add_argument(
    '--K_test',
    nargs='+', type=int,
    default=[],
    help='Rician Factor test'
)
parser.add_argument(
    '--no_pkt',
    type=int,
    default=128,
    help='Number of packet'
)
parser.add_argument(
    '--D_step',
    type=int,
    default=4096,
    help='Distortion step in D-R curve'
)
parser.add_argument(
    '--lr',
    nargs='+', type=float,
    default=[],
    help='Learning Rate'
)
parser.add_argument(
    '--Batch_Size',
    type=int,
    default=1,
    help='Batch size'
)
parser.add_argument(
    '--mode_input_scaling',
    type=int,
    default=2,
    help='Mode input scaling'
)

# 입력 받은 인자 parsing
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.pycharm_run == 'disable':          # 배치 파일 사용하여 프롬프트 창에서 실행할 때
    num_epoch = FLAGS.num_epoch
    num_sample_train = FLAGS.num_sample_train
    num_sample_test = FLAGS.num_sample_test
    snr_dB_train = np.array(FLAGS.snr_dB_train)
    snr_dB_test = np.array(FLAGS.snr_dB_test)
    K_train = np.array(FLAGS.K_train)
    K_test = np.array(FLAGS.K_test)
    no_pkt = FLAGS.no_pkt
    D_STEP = FLAGS.D_step
    lr = np.array(FLAGS.lr)
    Batch_Size = FLAGS.Batch_Size
    mode_input_scaling = FLAGS.mode_input_scaling

elif FLAGS.pycharm_run == 'enable':         # 파이참에서 실행할 때
    num_epoch = 150

    num_sample_train = 800
    num_sample_test = 100

    snr_dB_train = np.arange(0, 65, 5)
    snr_dB_test = np.arange(0, 65, 5)

    K_train = np.arange(0, 13, 4)
    K_test = np.arange(0, 13, 4)

    no_pkt = 128

    D_STEP = 2**12      # D-R curve 파일에서 distortion 읽어올 간격

    lr = np.array([3e-4])

    # Batch_Size = 1
    Batch_Size = 20

    # mode_input_scaling = 0 # just log scale
    # mode_input_scaling = 1 # log scale로 변환 후 min과 max 이용. 범위: (0,1)
    mode_input_scaling = 2 # log scale로 변환 후 min과 max 이용. 범위: (-1,1)
    # mode_input_scaling = 3 # log scale로 변환 후 평균=0, 표준편차=1로 변환.
    #주의: mode 3은 굳이 쓸필요 없을것 같음. distortion-rate 값들을 log10(.) 변환후의 표준편차가 0.3정도 이므로,
    # 강제로 분산을 1.0으로 맞추면 분포를 더 퍼지게 함. (의미가 없지는 않으나..)

else:
    print('Pycharm / batch run setting error!!')


# 송신, 수신 안테나 개수(2x2)
Nt = 2
Nr = 2

# R: spectral efficiency, C: spatial multiplexing rate (1: OSTBC, 4/3: DBLAST, 2: VBLAST)
R = np.array([1,2,3,4]).reshape([1,-1])     #1x4
C = np.array([1,4/3,2]).reshape([1,-1])     #1x3

# DNN 구조
# input node 개수: 67(=65+1+1)
# hidden node 개수: 16
# output node 개수: no_pkt*(R.shape[1]+C.shape[1])
# ===> [67, 16, 896]
Layer_dim_list = [16, R.shape[1]*no_pkt+C.shape[1]*no_pkt]

# bits
image_size = 512 * 512 * 1  # 512x512 해상도, 1bpp
maximum_pkt_size = image_size / no_pkt  # max num of bits per packet: 2048

STEP = np.arange(0, int(np.ceil(image_size / 64)) + 1, int(D_STEP / 64))    # D-R curve 간격(D_STEP)에 따른 index




#========================================================================================================
#
# 2. Outage Probability                                                                                 => dnn.py로 옮겨도 될 듯
#
#   rician factor에 따른 OSTBC, DBLAST, VBLAST의 Pout 파일 읽어 온 후, Pout_all에 저장.
#   Pout_all의 shape: K.shape[0]xC.shape[1]x71x5
#
#========================================================================================================
Pout_all = np.zeros([K_train.shape[0], C.shape[1], 60-(-10)+1, R.shape[1]+1])   # K_train.shape[0]x3x71x5
Pout_all[:,:,:,0] = np.arange(-10,61,1)     # 파일에 써져 있는 snr 범위 -10:1:60

file_list = ['OSTBC', 'DBLAST', 'VBLAST']

for k in range(K_train.shape[0]):       # rician factor
    for i in range(len(file_list)):     # spatial multiplexing rate
        for r in range(R.shape[1]):     # data rate
            filename = 'C:\python\sungmi\jscc\jscc\outage_prob\outage_prob_rician\Pout/' + file_list[i] + '_K' + str(K_train[k]).zfill(2) + '_R' + str(R[0,r]) + '_2x2.dat'
            with open(filename, 'r') as f:
                rdr = csv.reader(f, delimiter='\t')
                temp = np.array(list(rdr), dtype=np.float64)
                data = temp.reshape([1, 71, 2])
                # print(data[0,:,1])
                Pout_all[k,i,:,r+1] = data[0,:,1]  # R값에 따른 Pout만을 Pout_all에 넣어줌.




#========================================================================================================
#
# 3. Train, test data
#
#   data: Distortion-Rate curve
#
#========================================================================================================
# input_path에 있는 경로의 파일(image distortion sample) 여러 개를 읽어와서 input에 저장.
# input.shape: (num_sample)x65 (65 = 512^2 * 1.0 / D_STEP + 1, 0을 포함하기 위해 1 더해 줌)

i = 0
num_total_sample = 900
input = np.zeros([num_total_sample, STEP.shape[0]])

# 파일이 있는 경로 가져 오기
input_path = sys.argv[0]            #input_path = C:\python\sungmi\jscc_cross_entropy\main.py
input_path = input_path[:-7]        #input_path = C:\python\sungmi\jscc_cross_entropy\

# 파일에서 읽어 온 데이터를 input에 저장
for input_file in glob.glob(os.path.join(input_path, 'distortion_all\distort_0*')):
# for input_file in glob.glob(os.path.join(input_path, 'distortion_new\distort_a0001*')):
    with open(input_file, 'r') as f:
        rdr = csv.reader(f, delimiter='\t')
        temp = np.array(list(rdr), dtype=np.float64)
        input[i, :] = temp[STEP, 1]
        i = i + 1

    if i == num_total_sample:
        break

# train 데이터와 test 데이터로 나누기
input_train, input_test = np.vsplit(input, [num_sample_train])
# input_train, input_test, input_valid = np.vsplit(input, (num_sample_train, num_sample_train+num_sample_test))




#========================================================================================================
#
# 4. DNN 객체
#
#========================================================================================================
dnnObj = Dnn(batch_size=Batch_Size, mode_input_scale = mode_input_scaling, mode_shuffle = 'enable',  mode_find_optimal = 'disable',
             n_epoch=num_epoch, layer_dim_list=Layer_dim_list, max_pkt_size=maximum_pkt_size, num_pkt=no_pkt, D_step = D_STEP, nr=Nr, nt=Nt, r=R, c=C)




#========================================================================================================
#
# 5. train 관련 파일 초기화
#
#========================================================================================================
# train_dnn()의 프린트 파일들은 전부 'append' 옵션으로 생성.
# simulation할 때 이전 결과 삭제하고 싶으면 아래처럼 main에서 'w'옵션으로 파일 생성
with open('distort_dnn_per_batch.dat', 'w') as f:
    pass
with open('R_dnn_train_epoch.dat', 'w') as f, open('R_prob_dnn_train_epoch.dat', 'w') as f2:
    pass
with open('Cx3_dnn_train_epoch.dat', 'w') as f, open('Cx3_prob_dnn_train_epoch.dat', 'w') as f2:
    pass
with open('D_dnn_train.dat', 'w') as f, open('D_opt_train.dat', 'w') as f2:
    pass
with open('R_dnn_train.dat', 'w') as f, open('R_opt_train.dat', 'w') as f2, open('Cx3_dnn_train.dat', 'w') as f3, open('Cx3_opt_train.dat', 'w') as f4:
    pass
with open('info.dat', 'w') as f, open('info_no_name.dat', 'w') as f2:
    pass




#========================================================================================================
#
# 6. Train
#
#========================================================================================================
for j in range(lr.shape[0]):
    dnnObj.train_dnn(input_train, snr_dB_train, K_train, lr[j])


if num_sample_train % Batch_Size != 0:
    print('===========================================================\n')
    print('Warning: num_sample_train is not a multiple of Batch_Size!!\n')
    print('===========================================================\n')




#========================================================================================================
#
# 7. test 관련 파일 초기화
#
#========================================================================================================
with open('D_dnn_test.dat', 'w') as f, open('D_opt_test.dat', 'w') as f2:
    pass
with open('R_dnn_test.dat', 'w') as f, open('R_opt_test.dat', 'w') as f2, open('Cx3_dnn_test.dat', 'w') as f3, open('Cx3_opt_test.dat', 'w') as f4:
    pass




#========================================================================================================
#
# 8. Test
#
#========================================================================================================
for j in range(lr.shape[0]):
    for k in range(K_test.shape[0]):
        for i in range(snr_dB_test.shape[0]):
            # dnnObj.test_dnn(input_valid, snr_dB_test[i], K_test[k], lr[j])
            dnnObj.test_dnn(input_test, snr_dB_test[i], K_test[k], lr[j])




#========================================================================================================
#
# 9. Simple test
#
#   DNN의 연산 속도 재기
#   : dnnObj.test_dnn 함수 중 불필요한 부분 제거하여 속도 잼
#
#========================================================================================================
if 1:
    with open('dnn_time.dat', 'w') as f:
        pass

    i = 0

    # 연산 속도를 위해 사용할 총 데이터 갯수
    num_total = 1000

    # 한 번에 넣을 DNN input 갯수
    num_sample = 1
    # num_sample = 10
    # num_sample = 100
    # num_sample = 1000

    num_iter = int(num_total/num_sample)

    all_data = np.zeros([num_total, STEP.shape[0]])  # all_data.shape: (num_total)x65 (65 = 512^2 * 1.0 / D_STEP + 1, 0을 포함하기 위해 1 더해 줌)

    # 파일이 있는 경로 가져 오기
    input_path = sys.argv[0]            #input_path = C:\python\sungmi\jscc_cross_entropy\main.py
    input_path = input_path[:-7]        #input_path = C:\python\sungmi\jscc_cross_entropy\

    # 파일에서 읽어 온 데이터를 all_data에 저장
    for input_file in glob.glob('distortion_new\distort_a*'):
        with open(input_file, 'r') as f:
            rdr = csv.reader(f, delimiter='\t')
            temp = np.array(list(rdr), dtype=np.float64)
            # temp = temp.reshape([1,4099,2])
            all_data[i, :] = temp[STEP, 1]
            i = i + 1

        if i == num_total:
            break

    # DNN
    for j in range(lr.shape[0]):
        # 파일에서 weight, bias, input scaling parameter 읽어오기
        dnnObj.read_file_param(all_data.shape[1], lr[j])

        for k in range(K_test.shape[0]):
            for m in range(snr_dB_test.shape[0]):

                for n in range(num_iter):
                    # num_sample만큼의 데이터를 input_test에 저장
                    input_test = all_data[n * num_sample:(n + 1) * num_sample]

                    start_time_sec = time.time()
                    dnnObj.test_dnn_simple(input_test, snr_dB_test[m], K_test[k])
                    total_time = time.time() - start_time_sec

                    with open('dnn_time.dat', 'a') as f:
                        f.write('%d\t%02d\t%02d\t%d\t%20.20g\n' % (n, K_test[k], snr_dB_test[m], num_sample, total_time))




