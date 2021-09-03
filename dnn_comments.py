#========================================================================================================
#
# dnn.py
#
# train과 test 메소드 정의
#
# DNN 입력: D-R 커브, SNR, rician factor
# DNN 구조: [67, 16, 896]
#           - 손실함수: cross entropy (원-핫 인코딩한 값으로 계산)
#           - 레이블: TCOM solution (parametric approach)
# DNN 출력: spectral efficiency, spatial multiplexing rate
#
#========================================================================================================

import csv
import math
import time
import datetime
import numpy as np
from itertools import product
import tensorflow as tf
from sklearn.utils import shuffle

from outage_prob import Pout

#기존 tensorflow 1.13.1 의 tf.placeholder, feed_dict 등을 사용하기 위해 추가해야 하는 import임.
#하지만 이 방식대로 할 경우 현재의 tensorflow 2.0 의 장점들을 사용할 수 없다고 함.
#tensorflow 2.0을 밑의 import 추가 없이 사용하기 위해서는 코드 수정이 필요함.
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

print("===== np version: %s =====" %np.__version__)
print("===== tf version: %s =====" %tf.__version__)
print("===== Is GPU available?: %s =====" %tf.test.is_gpu_available())
#참고: sess = tf.Session(config = tf.ConfigProto(log_device_placement=True)) --> 연산하는 디바이스를 알려줌 (gpu or cpu)


def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


#========================================================================================================
#
# 파일(file_name)에 data 쓰기
#
#========================================================================================================
def file_write(file_name, write_type, data):
    with open(file_name, write_type) as f:
        if data.shape[0] == data.size:  # vector
            for i in range(data.shape[0]):
                f.write('%10.10g\n' % (data[i]))
        else:   # matrix
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write('%10.10g    ' % (data[i, j]))
                f.write('\n')
    f.close()

#========================================================================================================
#
# 파일(file_name)에서 데이터 읽어 오기
#
#========================================================================================================
def file_read(file_name, data_type):
    f = open(file_name, 'r')
    data = f.readlines()
    w = np.zeros_like(data_type)
    for i, line in enumerate(data):
        line = line.rstrip("\n")
        line = line.rstrip()
        if data_type.shape[0] == data_type.size:    # vector
            w[i] = float(line)
        else:   # matrix
            try:
                a = line.split()
                a = [float(j) for j in a]

                w[i,:] = a

            except ValueError as e:
                print(e)
                print ("on line %d" %i)
                print(data_type.shape)
                print("a.shape: %s" %len(a))
                print("w.shape: %s" %w[i,:].shape)
    f.close()
    return w




#========================================================================================================
#
# 클래스 DNN
#
#========================================================================================================
class Dnn:
    def __init__(self, batch_size=100,  mode_input_scale = 0, mode_shuffle = 'disable', mode_find_optimal = 'disable',
                 n_epoch=200, layer_dim_list = [16, 896], max_pkt_size=2**11, num_pkt=128, D_step = 2**12, nr = 2, nt = 2, r=np.array([1,2,3,4]), c=np.array([1,4/3,2])):
        self.n_epoch = n_epoch
        self.layer_dim_list = layer_dim_list
        self.batch_size = batch_size
        self.mode_input_scale = mode_input_scale
        self.mode_shuffle = mode_shuffle
        self.mode_find_optimal = mode_find_optimal

        self.max_pkt_size = max_pkt_size
        self.num_pkt = num_pkt

        self.D_step = D_step

        self.Nr = nr
        self.Nt = nt

        self. R = r
        self.Cx3 = c*3

        self.num_R = r.shape[1]
        self.num_C = c.shape[1]

        self.R_max = np.max(r)
        self.Cx3_max = np.max(c)*3



    # --------------------------------------------------------------------------------------------------------
    # Calc_distort 메소드
    #   : Expected Distortion 계산하는 함수
    #
    #   - 파라미터
    #       input: 한 샘플에 대한 distortion-rate curve
    #             (주의) 스케일링 (i.e., normalize) 하지 않은 입력 샘플
    #       R: spectral efficiency / Cx3: spatial multiplexing rate*3
    #             shape: num_pkt
    #       Pout_all_stbc: outage probability
    #             shape: 3(= s.m.r갯수) x 4(=s.e갯수)
    #
    #   - 반환 값: expected distortion
    #
    #   - 알고리즘
    #       1. pkt 별 outage probabilty
    #       2. expected distortion 계산
    # --------------------------------------------------------------------------------------------------------
    def Calc_distort(self, input, R, Cx3, Pout_all_stbc):

        D = input
        R_int = R.astype('int64')   # R을 Pout_all_stbc의 idx로 쓰기 위해 float->int로 변환
        E_D = 0

        # --------------------------------------------------------------------------------
        # 1. pkt 별 outage probability
        # ex1) R = 1, Cx3 = 4 인 pkt에 대한 Pout: Pout_all_stbc[1,0]
        # ex2) R = 4, Cx3 = 3 인 pkt에 대한 Pout: Pout_all_stbc[0,3]
        # --------------------------------------------------------------------------------
        Pout = np.zeros_like(R, dtype=np.float64)  # num_pkt=128 -> Pout.shape=128

        for i in range(Cx3.shape[0]):  # Cx3.shape[0]= num_pkt
            if (Cx3[i] == 3):       # OSTBC
                Pout[i] = Pout_all_stbc[0, R_int[i] - 1]

            elif (Cx3[i] == 4):     # DBLAST
                Pout[i] = Pout_all_stbc[1, R_int[i] - 1]

            elif (Cx3[i] == 6):     # VBLAST
                Pout[i] = Pout_all_stbc[2, R_int[i] - 1]
            else:
                print("[Calc_distort] Error in choosing space-time coding!!!")

        # --------------------------------------------------------------------------------
        # 2. expected distortion 계산
        # --------------------------------------------------------------------------------
        for success_pkt in range(self.num_pkt, 0, -1):
            # throughput
            total_bits = self.max_pkt_size * (np.sum(R[0:success_pkt] * Cx3[0:success_pkt]) / (self.R_max * self.Cx3_max))

            if len(np.where(R==4)[0]) == self.num_pkt and len(np.where(Cx3==6)[0]) == self.num_pkt:
                # 모든 packet의 R이 4이고 Cx3이 6인 경우, distortion을 interpolation할 필요 없이
                # D 행렬에서 마지막 distortion 값을 가져 오면 됨.
                D_idx = (total_bits / self.D_step).astype('int32')
                Distortion = D[D_idx]
            else:
                # distortion interpolation
                D_idx1 = (total_bits / self.D_step).astype('int32')  # e.g. total_bits=200, D_step=64 -> 3.125 -> 3
                D_idx2 = D_idx1 + 1

                small_v = (D_idx1 * self.D_step).astype('float64')
                big_v = (D_idx2 * self.D_step).astype('float64')

                w1 = (big_v - total_bits) / (big_v - small_v)
                w2 = (total_bits - small_v) / (big_v - small_v)

                Distortion = w1 * D[D_idx1] + w2 * D[D_idx2]

            if success_pkt == self.num_pkt:
                value = Distortion
            else:
                value = Distortion * Pout[success_pkt]

            E_D = E_D + value
            E_D = E_D * (1 - Pout[success_pkt - 1])

        E_D = E_D + D[0] * Pout[0]

        return E_D



    # --------------------------------------------------------------------------------------------------------
    # Find_Optimal_R_Cx3 메소드
    #   : Expected Distortion 계산하는 함수
    #
    #   - 파라미터
    #       input: 한 샘플에 대한 distortion-rate curve
    #             (주의) 스케일링 (i.e., normalize) 하지 않은 입력 샘플
    #       R_case: pkt에 대하여 가능한 모든 spectral efficiency 조합
    #       Cx3_case: pkt에 대하여 가능한 모든 spatial multiplexing rate*3 조합
    #       Pout_all_stbc: outage probability
    #             shape: 3(= s.m.r갯수) x 4(=s.e갯수)
    #
    #   - 반환값: 최적의 expected distortion, R, Cx3
    #
    #   - 알고리즘
    #       1. 초기 최적 expected distortion 값을 inf으로 설정
    #       2. R, Cx3 한 경우에 대해 expected distortion 계산
    #       3. 최적의 R, Cx3, expected distortion 업데이트
    # --------------------------------------------------------------------------------------------------------
    def Find_Optimal_R_Cx3(self,input, R_case, Cx3_case, Pout_all_stbc):  # input[m,:] 즉 한 줄이 들어온 것(single set)

        # --------------------------------------------------------------------------------
        # 1. 초기 최적 expected distortion 값을 inf으로 설정
        # --------------------------------------------------------------------------------
        min_distort = math.inf

        for i in range(R_case.shape[0]):
            for j in range(Cx3_case.shape[0]):
                # --------------------------------------------------------------------------------
                # 2. R, Cx3 한 경우에 대해 expected distortion 계산
                # --------------------------------------------------------------------------------
                distort = self.Calc_distort(input, R_case[i, :], Cx3_case[j, :], Pout_all_stbc)

                # --------------------------------------------------------------------------------
                # 3. 최적의 R, Cx3, expected distortion으로 업데이트
                # --------------------------------------------------------------------------------
                if distort < min_distort:
                    min_distort = distort
                    optimal_R = R_case[i, :]
                    optimal_Cx3 = Cx3_case[j, :]

        return min_distort, optimal_R, optimal_Cx3



    # --------------------------------------------------------------------------------------------------------
    # read_file_param 메소드
    #   : train 완료 후, 저장된 weight, bias 값과
    #     train할 때 사용했던 input normalizing parameter를 읽어 온다.
    #
    #
    #   - 알고리즘
    #       1. weight와 bias 읽어 오기
    #       2. input parameter scaling 읽어 오기
    # --------------------------------------------------------------------------------------------------------
    def read_file_param(self, input_shape, lr):
        # tf.reset_default_graph()

        # 1. weight와 bias 읽어 오기
        self.weights_final = []
        self.biases_final = []
        for i in range(len(self.layer_dim_list)):
            if i == 0:
                weight = np.zeros([input_shape + 2, self.layer_dim_list[i]], dtype=np.float64)
            else:
                weight = np.zeros([self.layer_dim_list[i - 1], self.layer_dim_list[i]], dtype=np.float64)
            bias = np.zeros(self.layer_dim_list[i], dtype=np.float64)

            weight = file_read("C:\python\sungmi\jscc_cross_entropy\simulation\\rician factor snr input\D_step=4096\\batch_size=20\dnn_size=[16]\k=0,4,8,12_snr0,5,...,60_150epoch_lr3e-4,1e-4_[16]\W" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", weight)
            bias = file_read("C:\python\sungmi\jscc_cross_entropy\simulation\\rician factor snr input\D_step=4096\\batch_size=20\dnn_size=[16]\k=0,4,8,12_snr0,5,...,60_150epoch_lr3e-4,1e-4_[16]\B" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", bias)

            # weight = tf.convert_to_tensor(weight)
            # bias = tf.convert_to_tensor(bias)

            self.weights_final.append(weight)
            self.biases_final.append(bias)

        # 2. input parameter scaling 읽어 오기
        with open('input_scaling_param.dat', 'r') as f:
            self.lines = f.readlines()



##############################################################################################################
#
# < train_dnn()의 프린트 파일들 >
#
# input_scaling_param.dat: normalize한 input의 max, min, avg, std 프린트
#
# info.dat: epoch 진행하면서 변하는 여러 개 값들 프린트
# info_no_name.dat: 상동 (단, 항목 이름 없음)
#
# distort_dnn_per_batch.dat: (매 lr별로) 매 k, SNR, batch 마다 업데이트 된 W, b로 계산한 distort_dnn_per_batch[b] 프린트.
#                            *** 파일형식: num_epoch=2, num_batch=2, num_snr=2, num_k=2, lr= [lr1, lr2] 일 때
#                                         lr1 | epoch1 | k1 | snr1 | distort (batch 1)
#                                         lr1 | epoch1 | k1 | snr1 | distort (batch 2)
#                                         lr1 | epoch1 | k1 | snr2 | distort (batch 1)
#                                         lr1 | epoch1 | k1 | snr2 | distort (batch 2)
#                                         lr1 | epoch1 | k2 | snr1 | distort (batch 1)
#                                         lr1 | epoch1 | k2 | snr1 | distort (batch 2)
#                                         lr1 | epoch1 | k2 | snr2 | distort (batch 1)
#                                         lr1 | epoch1 | k2 | snr2 | distort (batch 2)
#                                         lr1 | epoch2 | k1 | snr1 | distort (batch 1)
#                                         lr1 | epoch2 | k1 | snr1 | distort (batch 2)
#                                         lr1 | epoch2 | k1 | snr2 | distort (batch 1)  ...
#
# D_dnn_train.dat: (매 lr별로) train 끝난 후 최종 업데이트된 W, b로 계산한 distort_dnn_per_sample[s] 프린트
#                  *** 파일형식: num_sample=2, num_snr=2, num_k=2, lr=[lr1, lr2] 일 때
#                                         lr1 | k1 | snr1 | distort (sample 1)
#                                         lr1 | k1 | snr1 | distort (sample 2)
#                                         lr1 | k1 | snr2 | distort (sample 1)
#                                         lr1 | k1 | snr2 | distort (sample 2)
#                                         lr1 | k2 | snr1 | distort (sample 1)
#                                         lr1 | k2 | snr1 | distort (sample 2)
#                                         lr1 | k2 | snr2 | distort (sample 1)
#                                         lr1 | k2 | snr2 | distort (sample 2)
#                                         lr2 | k1 | snr1 | distort (sample 1)
#                                         lr2 | k1 | snr1 | distort (sample 2)    ...
#
# D_opt_train.dat: (매 lr별로) distort_opt_per_sample[] 프린트 (파일형식: lr | k | SNR | distort)
#                  ***파일형식: 상동
#
# R_dnn_train.dat: (매 lr별로) train 끝난 후 최종 업데이트된 W, b로 계산한 dnn_R_per_sample[s][r] (즉, DNN이 출력한 샘플별 R) 프린트
#                  *** 파일형식: num_sample=2, num_snr=2, num_k=2, lr=[lr1, lr2] 일 때
#                                         lr1 | k1 | snr1 | r1 r2 ... (sample 1)
#                                         lr1 | k1 | snr1 | r1 r2 ... (sample 2)
#                                         lr1 | k1 | snr2 | r1 r2 ... (sample 1)
#                                         lr1 | k1 | snr2 | r1 r2 ... (sample 2)
#                                         lr1 | k2 | snr1 | r1 r2 ... (sample 1)
#                                         lr1 | k2 | snr1 | r1 r2 ... (sample 2)
#                                         lr1 | k2 | snr2 | r1 r2 ... (sample 1)
#                                         lr1 | k2 | snr2 | r1 r2 ... (sample 2)
#                                         lr2 | k1 | snr1 | r1 r2 ... (sample 1)
#                                         lr2 | k1 | snr1 | r1 r2 ... (sample 2)    ...
#
# R_opt_train.dat: (매 lr별로) opt_R_per_sample[s][r] (즉, 샘플별 optimal R) 프린트
#                  *** 파일형식: 상동
#
# Cx3_dnn_train.dat: (매 lr별로) train 끝난 후 최종 업데이트된 W, b로 계산한 dnn_Cx3_per_sample[s][r] (즉, DNN이 출력한 샘플별 Cx3) 프린트
#                  *** 파일형식: num_sample=2, num_snr=2, num_k=2, lr=[lr1, lr2] 일 때
#                                         lr1 | k1 | snr1 | c1 c2 ... (sample 1)
#                                         lr1 | k1 | snr1 | c1 c2 ... (sample 2)
#                                         lr1 | k1 | snr2 | c1 c2 ... (sample 1)
#                                         lr1 | k1 | snr2 | c1 c2 ... (sample 2)
#                                         lr1 | k2 | snr1 | c1 c2 ... (sample 1)
#                                         lr1 | k2 | snr1 | c1 c2 ... (sample 2)
#                                         lr1 | k2 | snr2 | c1 c2 ... (sample 1)
#                                         lr1 | k2 | snr2 | c1 c2 ... (sample 2)
#                                         lr2 | k1 | snr1 | c1 c2 ... (sample 1)
#                                         lr2 | k1 | snr1 | c1 c2 ... (sample 2)    ...
#
# Cx3_opt_train.dat: (매 lr별로) opt_Cx3_per_sample[s][r] (즉, 샘플별 optimal Cx3) 프린트
#                  *** 파일형식: 상동
#
# R_dnn_train_epoch.dat: epoch이 진행되면서 R 값의 변화 기록
#
# R_prob_dnn_train_epoch.dat: epoch이 진행되면서 softmax 출력(R에 대한 확률) 변화 기록
#
# Cx3_dnn_train_epoch.dat: epoch이 진행되면서 Cx3 값의 변화 기록
#
# Cx3_prob_dnn_train_epoch.dat: epoch이 진행되면서 softmax 출력(Cx3에 대한 확률) 변화 기록
#
# weight_bias\(폴더) W1~.dat, W2~.dat, ...: train 끝난 후 최종 업데이트된 weight 프린트
#                  *** 파일이름 형식: lr=[1e-4,3e-4] 일때
#                                       W1_lr1e-4.dat ... (weight1, lr1)
#                                       W1_lr3e-4.dat ... (weight1, lr2)
#                                       W2_lr1e-4.dat ... (weight2, lr1)
#                                       W2_lr1e-4.dat ... (weight2, lr2)
#
# weight_bias\(폴더) B1~.dat, B2~.dat, ...: train 끝난 후 최종 업데이트된 bias 프린트
#                  *** 파일이름 형식: 상동
#
# weight_bias\(폴더) W1_ep400.dat, W1_ep800.dat, ..., W2_ep400.dat, ...: epoch 진행하면서 업데이트 되는 weight 프린트
# weight_bias\(폴더) B1_ep400.dat, B1_ep800.dat, ..., B2_ep400.dat, ...: epoch 진행하면서 업데이트 되는 bias 프린트
#
##############################################################################################################

    # --------------------------------------------------------------------------------------------------------
    # train_dnn 메소드
    #   : DNN 학습 관련 메소드
    #
    #   - 파라미터
    #       input: distortion-rate curve
    #             (주의) 스케일링 (i.e., normalize) 하지 않은 입력 샘플
    #       snr_dB: dB scale의 SNR
    #       K: rician factor
    #       lr: learning rate
    #       Pout_all: 모든 outage probability
    #
    #   - 알고리즘
    #       1. Settings
    #       2. Normalize the training data
    #       3. Build Model(graph of tensors)
    #       4. label 생성
    #       5. full search로 최적의 R, C, E(D) 찾기
    #       6. 학습 진행
    #       7. 결과 출력
    # --------------------------------------------------------------------------------------------------------
    def train_dnn(self, input, snr_dB, K, lr):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        # 층마다 존재하는 weight와 bias를 weights, biases에 저장
        self.weights = []
        self.biases = []

        # num_batch 계산
        num_batch = int(input.shape[0] / self.batch_size)  # input.shape[0]: input의 행의 갯수 (=800)

        # 다음 실행 때도 같은 난수가 발생하도록 seed 설정
        seed_weight = 1000
        seed_shuffle = 2000
        np.random.seed(seed_shuffle)

        # outage probability
        Poutobj = Pout(self.R, self.Cx3, K)
        Poutobj.generate_Pout()

        # --------------------------------------------------------------------------------
        # 2. Normalize the training data
        # --------------------------------------------------------------------------------
        input_not_scaled = input
        input_log = np.log10(input)

        max_input_log = np.max(input_log)
        min_input_log = np.min(input_log)

        avg_input_log = np.mean(input_log)
        std_input_log = np.std(input_log)

        with open('input_scaling_param.dat','w') as f:
            f.write('   %g\n   %g\n   %g\n   %g\n' % (max_input_log, min_input_log, avg_input_log, std_input_log))

        if (self.mode_input_scale == 1):        # log scale로 변환 후 min과 max 이용. 범위: (0,1)
            input = (input_log - min_input_log) / (max_input_log - min_input_log)

        elif (self.mode_input_scale == 2):      # log scale로 변환 후 min과 max 이용. 범위: (-1,1)
            temp = (input_log - min_input_log) / (max_input_log - min_input_log)
            input = 2 * temp - 1

        elif (self.mode_input_scale == 3):      # log scale로 변환 후 평균 = 0, 표준편차 = 1로 변환.
            input = (input_log - avg_input_log) / std_input_log

        else:   # just log scale
            input = input_log


        # --------------------------------------------------------------------------------
        # 3. Build Model(graph of tensors)
        #
        # 1) placeholder
        # 2) 뉴럴 네트워크 통과하는 연산
        # 3) 손실 함수
        # 4) learning rate scheduling O,X
        # 5) 그래프 실행 전, 변수 초기화
        # --------------------------------------------------------------------------------
        with tf.device('/CPU:0'):
            tf.reset_default_graph()

            # 1) placeholder
            x_ph = tf.placeholder(tf.float64, shape=[None,input.shape[1]+2])  # '+2'를 하는 이유: distortion, k, snr을 입력으로 넣음
            y_ph = tf.placeholder(tf.float64, shape=[None, self.layer_dim_list[-1]])

            # ======================================================
            # 2) 뉴럴 네트워크 통과하는 연산
            #
            #   - 네트워크 layer 설명
            #       e.g: layer_dim_list = [16 896]
            #            layer_dim_list[0]: hidden layer 1의 노드 수
            #            layer_dim_list[1]: output layer의 노드 수
            #       (주의) layer_dim_list: 입력 layer는 포함 X
            #
            #   - weight: He 초기값 사용, bias: 0으로 초기화
            #
            #   - activation function
            #       hidden layer: ReLU, output layer: softmax
            # ======================================================
            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph     # in_layer: DNN의 input layer 아님. i번째 layer를 가리키는 index (input layer, hidden1, hidden2,..)
                    in_dim = input.shape[1]+2   # in_dim: in_layer의 노드수 / '+2'를 하는 이유: distortion, k, snr을 입력으로 넣음
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer  # out_layer: DNN의 output layer 아님.
                    in_dim = self.layer_dim_list[i-1]
                    out_dim = self.layer_dim_list[i]

                # He initialization: 표준 편차 root(2/n), n은 앞 계층의 노드 수
                weight = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=tf.sqrt(2.0 / tf.cast(in_dim, tf.float64)), seed=seed_weight * (i * i + 1), dtype=tf.float64), dtype=tf.float64)
                # 0으로 초기화
                bias = tf.Variable(tf.zeros(out_dim, dtype=tf.float64), dtype=tf.float64)

                mult = tf.matmul(in_layer, weight) + bias

                # activation function
                if i < len(self.layer_dim_list)-1:  # hidden layer
                    out_layer = tf.nn.relu(mult)

                else:   # output layer
                    # ============================================================================================================
                    # (128pkt*num_R + 128pkt*numC) 개의 output -> 각각은 확률 값임 (softmax 통과하기 때문에)
                    #
                    # 아래와 같은 순서의 output
                    # pkt1_R1 pkt1_R2 pkt1_R3 pkt1_R4 | pkt2_R1 pkt2_R2 pkt2_R3 pkt2_R4 | ... | pkt128_R1 pkt128_R2 pkt128_R3 pkt128_R4 |
                    # pkt1_C1 pkt1_C2 pkt1_C3 | pkt2_C1 pkt2_C2 pkt2_C3 | ... | pkt128_C1 pkt128_C2 pkt128_C3
                    #
                    # R은 4개씩, C는 3개씩 끊어서 최대 값 찾음
                    # pkt1_R1 pkt1_R2 pkt1_R3 pkt1_R4 에서 가장 큰 값을 갖는 인덱스 = pkt1의 R
                    # pkt1_C1 pkt1_C2 pkt1_C3에서 가장 큰 값을 갖는 인덱스 = pkt1의 C
                    # ============================================================================================================

                    ################ nan 발생할 경우 해결 방법1: 각각의 softmax에 대해 max값을 빼준다. ################
                    output_prob_R = tf.concat([tf.nn.softmax(mult[:,self.num_R*j:self.num_R*(j+1)]
                                                             - tf.reduce_max(mult[:,self.num_R*j:self.num_R*(j+1)],axis=1,keepdims=True)) for j in range(self.num_pkt)],1)
                    output_prob_C = tf.concat([tf.nn.softmax(mult[:, self.num_R*self.num_pkt + self.num_C * j: self.num_R*self.num_pkt +self.num_C * (j + 1)]
                                                             - tf.reduce_max(mult[:, self.num_R*self.num_pkt + self.num_C * j: self.num_R*self.num_pkt +self.num_C * (j + 1)],axis=1,keepdims=True)) for j in range(self.num_pkt)], 1)
                    output = tf.concat([output_prob_R, output_prob_C], 1)

                    ###### 각각의 softmax에서 최대 값의 index 찾는 방법 ######
                    # 방법 1
                    # 원래 사용하던 방법, R과 C의 개수에 상관없이 사용 가능
                    # output_R = tf.concat([tf.reshape(tf.argmax(output_prob_R[:, num_R * j:num_R * (j + 1)], axis=1),[-1,1]) + 1 for j in range(self.num_pkt)],1)
                    # output_C = tf.concat([tf.reshape(tf.argmax(output_prob_C[:, num_C * j:num_C * (j + 1)], axis=1),[-1,1]) + 1 for j in range(self.num_pkt)],1)

                    # 방법 2
                    # optimize graph (test 연산 속도 줄이기 위해 고안해 낸 알고리즘)
                    # R의 개수가 4개, C의 개수가 3개인 상황에서만 사용 가능
                    output_R = tf.argmax([output_prob_R[:, 0::self.num_R], output_prob_R[:, 1::self.num_R], output_prob_R[:, 2::self.num_R], output_prob_R[:, 3::self.num_R]], 0)
                    output_C = tf.argmax([output_prob_C[:, 0::self.num_C], output_prob_C[:, 1::self.num_C], output_prob_C[:, 2::self.num_C]], 0)


                # ======================================================
                # Weight 행렬 설명
                # e.g. layer = [9, 200, 300, 3]
                # self.weights[0] = input layer -> hidden layer 1 (9x200)
                # self.weights[1] = hidden layer 1 -> hidden layer 2 (200x300)
                # self.weights[2] = hidden layer 2 -> output layer (300x3)
                # ======================================================
                self.weights.append(weight)
                self.biases.append(bias)

            # 3) 손실 함수
            ################ nan 발생할 경우 해결 방법2: cross entropy 계산 시, log에 아주 작은 값을 더해줌 ################
            cross_entropy = y_ph * -1 * tf.log(output + 1e-100)
            loss_temp = tf.reduce_sum(cross_entropy, axis=1)
            loss = tf.reduce_mean(loss_temp)

            # 4) learning rate scheduling O,X
            ###### lr schedule ######
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = lr
            lr_shift_period = self.n_epoch * K.shape[0] * snr_dB.shape[0] * num_batch / 2
            lr_shift_rate = 0.3

            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, lr_shift_period, lr_shift_rate,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)  # lr: learning_rate
            train = optimizer.minimize(loss, global_step=global_step)

            ###### no lr schedule ######
            # learning_rate = lr
            # optimizer = tf.train.AdamOptimizer(learning_rate)  # lr: learning_rate
            # train = optimizer.minimize(loss)  # cost function or object function, train -> 클래스객체

            # 5) 그래프 실행 전, 변수 초기화
            # Initialization: loss, train등 모든 class 객체를 선언한 후에 해야 함.
            init = tf.global_variables_initializer()  # 위에서 선언한 variable들을 init에 연결함
            sess = tf.Session()
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))   # 어떤 하드웨어 장치를 사용해서 연산하는지 확인 가능
            sess.run(init)  # 세션 실행을 해야 weight와 bias에 초기값이 할당됨.


            start_time_sec = time.time()
            start_time = datetime.datetime.now()
            # print('======== Start Time: %s ========\n' % start_time)



            # --------------------------------------------------------------------------------
            # 4. label 생성
            #    label shape: (K.shape[0]) x (snr_dB.shape[0]) x (num_sample) x (num_pkt*4+num_pkt*3)
            #                                               [R1 R2 R3 ... C1 C2 C3 ...]
            #
            # 5. full search로 최적의 R, C, E(D) 찾기
            # --------------------------------------------------------------------------------
            label = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_R * self.num_pkt + self.num_C * self.num_pkt], dtype=int)

            # Find_Optimal_R_Cx3 실행할때 필요!! (R_case, C_case 먼저 만들어 줌)
            if self.mode_find_optimal == 'enable':
                R_case = product(np.array([1, 2, 3, 4]), repeat=self.num_pkt)
                R_case = np.array(list(R_case))  # R_case: 가능한 R의 모든 조합

                Cx3_case = product(3 * np.array([1, 4 / 3, 2]), repeat=self.num_pkt)
                Cx3_case = np.array(list(Cx3_case))

            distort_opt_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0]])
            opt_R_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt])
            opt_Cx3_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt])

            ################## 각각의 snr, k마다 4, 5 계산 ##################
            for k in range(K.shape[0]):
                for i in range(snr_dB.shape[0]):
                    # label 생성
                    filename = 'label\MIMO_TCOM_reference\D_step=' + str(self.D_step) + '\\rician_factor\R_C_tcom_' + str(self.num_pkt) + 'pkt_reference_snr' + str(snr_dB[i]).zfill(2) + '_K' + str(K[k]).zfill(2) + '_R=[1 2 3 4].dat'
                    with open(filename, 'r') as f:
                        rdr = csv.reader(f, delimiter='\t')
                        tcom_temp = np.array(list(rdr), dtype=np.int)
                        tcom_opt_R_C = tcom_temp[:input.shape[0], 2:].reshape([input.shape[0], self.num_pkt * 2])


                    # np.eye: numpy를 이용한 one hot encoding 방법
                    # [R label 만들기]
                    # ex) R1 = 1이면 1 0 0 0
                    tcom_opt_R = tcom_opt_R_C[:, :self.num_pkt]-1       #input.shape[0]xself.num_pkt    ex) [1, 2]
                    one_hot_tcom_opt_R = np.eye(self.num_R)[tcom_opt_R]      #input.shape[0]xself.num_pktxnum_R  ex) [[0, 1, 0, 0], [0, 0, 1, 0]]
                    label[k, i, :, :self.num_pkt * self.num_R] = np.reshape(one_hot_tcom_opt_R,[input.shape[0], -1])

                    # [C label 만들기]
                    # ex) C1 = 4이면 0 1 0
                    tcom_opt_C = tcom_opt_R_C[:, self.num_pkt:]         #input.shape[0]xself.num_pkt
                    tcom_opt_C[tcom_opt_C == 3] = 0
                    tcom_opt_C[tcom_opt_C == 4] = 1
                    tcom_opt_C[tcom_opt_C == 6] = 2
                    one_hot_tcom_opt_C = np.eye(self.num_C)[tcom_opt_C]      #input.shape[0]xself.num_pktxnum_C
                    label[k, i, :, self.num_pkt * self.num_R:] = np.reshape(one_hot_tcom_opt_C, [input.shape[0], -1])


                    # find optimal R & Cx3: 샘플별로 Optimal 성능 (distortion) 계산
                    Pout_k_snr = Poutobj.get_Pout_k_snr(k, snr_dB)
                    # snr_idx = np.where(Pout_all[0, 0, :, 0] == snr_dB[i])

                    for m in range(input.shape[0]):
                        if self.mode_find_optimal == 'enable':
                            # distort_opt_per_sample[k, i, m], opt_R_per_sample[k, i, m], opt_Cx3_per_sample[k, i, m] = self.Find_Optimal_R_Cx3(input_not_scaled[m, :], R_case, Cx3_case, np.squeeze(Pout_all[k,:, snr_idx[0], 1:]))
                            distort_opt_per_sample[k, i, m], opt_R_per_sample[k, i, m], opt_Cx3_per_sample[k, i, m] = self.Find_Optimal_R_Cx3(input_not_scaled[m, :], R_case, Cx3_case, Pout_k_snr)
                        else:
                            distort_opt_per_sample[k, i, m] = 1000000  # pkt 갯수가 크면 Find_Optimal_R_Cx3이 시간이 오래걸리므로 아무 값이나 넣은 것


            with open('info.dat', 'a') as f:
                f.write('\n')
            with open('info_no_name.dat', 'a') as f2:
                f2.write('\n')

            f3 = open("distort_dnn_per_batch.dat", 'a')

            # f_lr = open("lr_schedule.dat", 'a')

            dnn_R_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt])
            dnn_Cx3_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt])
            dnn_R_propability = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt*self.num_R])
            dnn_Cx3_propability = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt*self.num_C])
            distort_dnn_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0]])



            # --------------------------------------------------------------------------------
            # 6. 학습 진행
            # --------------------------------------------------------------------------------
            ##########################       epoch       ##########################
            for e in range(self.n_epoch):

                # shuffle
                if (self.mode_shuffle == 'enable'):
                    label_shuffle = np.zeros_like(label)
                    input_shuffle = shuffle(input, random_state=seed_shuffle*e)
                    for k in range(K.shape[0]):
                        for i in range(snr_dB.shape[0]):
                            label_shuffle[k,i,:] = shuffle(label[k,i,:], random_state=seed_shuffle*e)
                else:
                    input_shuffle = input
                    label_shuffle = label

                ##########################       batch       ##########################
                for j in range(num_batch):
                    input_batch = input_shuffle[j * self.batch_size: (j + 1) * self.batch_size]

                    ##########################       K       ##########################
                    for a in range(K.shape[0]):
                        K_vec = np.tile(K[a], (self.batch_size, 1))
                        input_batch_k = np.concatenate((input_batch, K_vec), axis=1)

                        ##########################       snr       ##########################
                        for i in range(snr_dB.shape[0]):
                            snr_dB_vec = np.tile(snr_dB[i], (self.batch_size, 1))  # np.tile(a, (n,m)): a를 (n,m)번 반복
                            input_batch_k_snr = np.concatenate((input_batch_k, snr_dB_vec), axis=1)

                            label_batch = label_shuffle[a, i, j * self.batch_size: (j + 1) * self.batch_size]

                            # 주의: 이 경우는 loss를 먼저 계산하고, train 실행 (W, b 업데이트)
                            distort_dnn_per_batch = sess.run(loss, feed_dict={x_ph: input_batch_k_snr, y_ph: label_batch})
                            distort_dnn_per_batch = self.batch_size * distort_dnn_per_batch

                            # Back propagation에 의해 W, b 업데이트 됨
                            sess.run(train, feed_dict={x_ph: input_batch_k_snr, y_ph: label_batch})

                            lr_shift = sess.run(optimizer._lr)

                            # distort_dnn_per_batch.dat에 loss값 write
                            f3.write('%g    %d    %2d    %d    %g\n' % (lr_shift, e + 1, K[a], snr_dB[i], distort_dnn_per_batch))  # e = 0은 이미 한번 train인 된 상태.

                        #####################################################################################################
                        #  주의:
                        #  가령 num_epoch = 10, num_sample_train = 6, batch_size = 3 이면, one epoch마다 W, b가 두 번 업데이트 됨.
                        #  distort_dnn_per_batch[0]은 1st 업데이트 된 W, b로 계산한 1st batch의 (3개 샘플의) loss 합.
                        #  distort_dnn_per_batch[1]은 2nd 업데이트 된 W, b로 계산한 2nd batch의 (3개 샘플의) loss 합
                        #  distort_dnn_per_batch[19]은 20th 업데이트 (최종 업데이트)된 W, b로 계산한 20th batch의 (3개 샘플의) loss 합
                        #####################################################################################################



                # --------------------------------------------------------------------------------
                # 7. 결과 출력
                #
                # 1) 텐서 변수들 session run
                # 2) DNN 출력 값을 이용하여 expected distortion 계산
                # 3) 파일과 파이참에 결과 출력
                #
                # 주의: one epoch, one k, one snr 동안의 update가 다 끝난 후의 결과들
                # --------------------------------------------------------------------------------
                ##########################       K       ##########################
                for a in range(K.shape[0]):
                    K_vec = np.tile(K[a], (input.shape[0], 1))
                    input_concat_K = np.concatenate((input, K_vec), axis=1)

                    ##########################       SNR       ##########################
                    for i in range(snr_dB.shape[0]):
                        snr_dB_vec = np.tile(snr_dB[i], (input.shape[0], 1))
                        input_concat = np.concatenate((input_concat_K, snr_dB_vec), axis=1)

                        # 1) 텐서 변수들 session run

                        # DNN output
                        #    output_R과 output_C은 인덱스 값을 저장하고 있으므로
                        #    R의 경우, 인덱스 값을 1,2,3,4로 / C의 경우, 인덱스 값을 3,4,6으로 매핑해줘야 함
                        dnn_R_per_sample[a,i] = sess.run(output_R, feed_dict={x_ph: input_concat}) + 1
                        dnn_Cx3_per_sample[a,i] = sess.run(output_C, feed_dict={x_ph: input_concat})
                        dnn_Cx3_per_sample[a,i,dnn_Cx3_per_sample[a,i] == 2] = 6
                        dnn_Cx3_per_sample[a,i,dnn_Cx3_per_sample[a,i] == 1] = 4
                        dnn_Cx3_per_sample[a,i,dnn_Cx3_per_sample[a,i] == 0] = 3

                        # DNN softmax output
                        dnn_R_propability[a,i] = sess.run(output_prob_R, feed_dict={x_ph: input_concat})
                        dnn_Cx3_propability[a,i] = sess.run(output_prob_C, feed_dict={x_ph: input_concat})

                        # DNN weights, biases
                        ww, bb = sess.run([self.weights, self.biases])


                        # 2) DNN 출력인 R과 Cx3을 이용하여 expected distortion 계산
                        #    샘플별로 DNN의 성능 (distortion) 계산
                        #    주의: one epoch, one k, one snr 동안의 update가 다 끝난후의 W, b를 가지고 DNN 성능 계산
                        Pout_k_snr = Poutobj.get_Pout_k_snr(a, snr_dB[i])
                        for m in range(input.shape[0]): # e.g. input.shape[0]=800
                            distort_dnn_per_sample[a, i, m] = self.Calc_distort(input_not_scaled[m, :], dnn_R_per_sample[a, i, m, :], dnn_Cx3_per_sample[a, i, m, :], Pout_k_snr)

                            # snr_idx = np.where(Pout_all[0, 0, :, 0] == snr_dB[i])
                            # distort_dnn_per_sample[a, i, m] = self.Calc_distort(input_not_scaled[m, :], dnn_R_per_sample[a,i, m, :], dnn_Cx3_per_sample[a,i, m, :], np.squeeze(Pout_all[a, :, snr_idx[0], 1:]))


                        # 3) 파일과 파이참에 결과 출력
                        if (e+1) % 20 == 0:
                            now_time = datetime.datetime.now()
                            remain_time = (now_time - start_time) * (self.n_epoch*K.shape[0]*snr_dB.shape[0]) / (e*K.shape[0]*snr_dB.shape[0] + a*snr_dB.shape[0] + (i + 1)) - (now_time - start_time)  # ?? remain time 계산 이거 맞음?

                            # 주의: 아래에서 프린트 하는 epoch = e+1임. (e=0은 이미 한번 train인 된 상태)
                            # epoch = e+1 = 0 은 W, b를 업데이트 하지 않은 것 (W_init, b_init 상태) -> 이때는 display 하지 않음.
                            # epoch = e+1 = 1 은 W, b를 한번 업데이트 한것
                            # epoch = e+1 = 2 는 W, b를 두번 업데이트 한것

                            print(
                                'epoch= %6d | K= %6d | snr_dB= %6d | w1= %8.5g | b1= %8.5g | distort_dnn_per_batch= %8.5g | R_1= %8.5g | R_2= %8.5g | R_3= %8.5g | Cx3_1= %8.5g | Cx3_2= %8.5g | Cx3_3= %8.5g'
                                ' | mean(distort_opt_per_sample[])= %8.5g | mean(distort_dnn_per_sample[])= %8.5g | distort ratio b/w dnn & opt = %8.5g (%%) | remain = %s(h:m:s)'
                                % (e + 1, K[a], snr_dB[i], ww[0][0, 0], bb[0][0], distort_dnn_per_batch, dnn_R_per_sample[a,i][0][0], dnn_R_per_sample[a,i][0][1], dnn_R_per_sample[a,i][0][2],
                                   dnn_Cx3_per_sample[a,i][0][0], dnn_Cx3_per_sample[a,i][0][1], dnn_Cx3_per_sample[a,i][0][2],
                                   np.mean(distort_opt_per_sample[a,i]), np.mean(distort_dnn_per_sample[a,i]), np.mean(distort_dnn_per_sample[a,i]) / np.mean(distort_opt_per_sample[a,i]) * 100, remain_time))

                            with open('info.dat', 'a') as f:
                                f.write('epoch= %6d | snr_dB= %6d | K= %6d | w1= %8.5g | b1= %8.5g | distort_dnn_per_batch= %8.5g | R_1= %8.5g | R_2= %8.5g | R_3= %8.5g | Cx3_1= %8.5g | Cx3_2= %8.5g | Cx3_3= %8.5g'
                                ' | mean(distort_opt_per_sample[])= %8.5g | mean(distort_dnn_per_sample[])= %8.5g | distort ratio b/w dnn & opt = %8.5g (%%) | remain = %s(h:m:s)'
                                % (e + 1, snr_dB[i], K[a], ww[0][0, 0], bb[0][0], distort_dnn_per_batch, dnn_R_per_sample[a,i][0][0], dnn_R_per_sample[a,i][0][1], dnn_R_per_sample[a,i][0][2],
                                   dnn_Cx3_per_sample[a,i][0][0], dnn_Cx3_per_sample[a,i][0][1], dnn_Cx3_per_sample[a,i][0][2],
                                   np.mean(distort_opt_per_sample[a,i]), np.mean(distort_dnn_per_sample[a,i]), np.mean(distort_dnn_per_sample[a,i]) / np.mean(distort_opt_per_sample[a,i]) * 100, remain_time))

                            with open('info_no_name.dat', 'a') as f2:
                                f2.write(
                                    '%6d   %6d   %6d   %8.5g   %8.5g   %8.5g   %8.5g   %8.5g   %8.5g   %8.5g   %8.5g   %8.5g   %8.5g   %6d   %8.5g   %8.5g   %8.5g\n' % (
                                        e+1, snr_dB[i], K[a], ww[0][0, 0], bb[0][0], distort_dnn_per_batch, dnn_R_per_sample[a,i][0][0], dnn_R_per_sample[a,i][0][1],dnn_R_per_sample[a,i][0][2],
                                        dnn_Cx3_per_sample[a,i][0][0], dnn_Cx3_per_sample[a,i][0][1], dnn_Cx3_per_sample[a,i][0][2],
                                        lr, self.batch_size, np.mean(distort_opt_per_sample[a,i]), np.mean(distort_dnn_per_sample[a,i]), np.mean(distort_dnn_per_sample[a,i])/np.mean(distort_opt_per_sample[a,i])*100))

                            if (i == snr_dB.shape[0] - 1):
                                print('\n')

                            # if (e+1) %20 == 0:
                        if (e + 1) % 1 == 0:
                            with open('R_dnn_train_epoch.dat', 'a') as f:
                                # for m in range(input.shape[0]):
                                for m in range(5):
                                    f.write('%d\t%d\t%d\t' % (snr_dB[i], K[a], e + 1))
                                    for k in range(self.num_pkt):
                                        f.write('%10.10g  ' % dnn_R_per_sample[a, i, m, k])
                                    f.write('\n')

                            with open('Cx3_dnn_train_epoch.dat', 'a') as f:
                                # for m in range(input.shape[0]):
                                for m in range(5):
                                    f.write('%d\t%d\t%d\t' % (snr_dB[i], K[a], e + 1))
                                    for k in range(self.num_pkt):
                                        f.write('%10.10g  ' % dnn_Cx3_per_sample[a, i, m, k])
                                    f.write('\n')

                            with open('R_prob_dnn_train_epoch.dat', 'a') as f:
                                # for m in range(input.shape[0]):
                                for m in range(5):
                                    f.write('%d\t%d\t%d\t' % (snr_dB[i], K[a], e + 1))
                                    for k in range(self.num_pkt * 4):
                                        f.write('%10.10g  ' % dnn_R_propability[a, i, m, k])
                                    f.write('\n')

                            with open('Cx3_prob_dnn_train_epoch.dat', 'a') as f:
                                # for m in range(input.shape[0]):
                                for m in range(5):
                                    f.write('%d\t%d\t%d\t' % (snr_dB[i], K[a], e + 1))
                                    for k in range(self.num_pkt * 3):
                                        f.write('%10.10g  ' % dnn_Cx3_propability[a, i, m, k])
                                    f.write('\n')


                if e == 0:
                    print(
                        '#pkts = %6d | max_pkt_size = %6d | snr = %s | k = %s |  #trn_samp = %6d | n_epoch= %6d | lrn_rate = %8.5g | bat_size = %5d | input_scaling_mode = %d | shuffle= %s | seed_weight = %6d | DNN size = %s'
                        % (self.num_pkt, self.max_pkt_size, snr_dB, K, input.shape[0], self.n_epoch, lr, self.batch_size, self.mode_input_scale, self.mode_shuffle, seed_weight, self.layer_dim_list))

                    with open('info.dat', 'a') as f:
                        f.write(
                            '#pkts = %6d | max_pkt_size = %6d | snr = %s | k = %s | #trn_samp = %6d | n_epoch= %6d | lrn_rate = %8.5g | bat_size = %5d | input_scaling_mode = %d | shuffle= %s | seed_weight = %6d | DNN size = %s'
                            % (self.num_pkt, self.max_pkt_size, snr_dB, K, input.shape[0], self.n_epoch, lr, self.batch_size, self.mode_input_scale, self.mode_shuffle, seed_weight, self.layer_dim_list))


                # 최종 update 후의 weight, bias 파일에 저장
                if e == self.n_epoch - 1:
                    for i in range(len(self.layer_dim_list)):
                        file_write('W' + str(i + 1) + '_lr' + str(format(lr,"1.0e")) + '.dat', 'w', ww[i][:, :])
                        file_write('B' + str(i + 1) + '_lr' + str(format(lr,"1.0e")) + '.dat', 'w', bb[i][:])

                #epoch 20간격으로 W,b 파일에 저장
                # if (e+1) % 400 == 0:
                if e+1>=470 and e+1<=490:
                    for i in range(len(self.layer_dim_list)):
                        file_write('weight_bias\W' + str(i+1)+'_ep'+ str(e+1) + '.dat', 'w', ww[i][:, :])
                        file_write('weight_bias\B' + str(i+1)+'_ep'+ str(e+1) + '.dat', 'w', bb[i][:])


            ##########################################################
            # End of "epoch loop"
            ##########################################################
            sess.close()
        f3.close()


        # 3) 파일과 파이참에 결과 출력
        # dnn_R_per_sample[], dnn_Cx3_per_sample[], distort_dnn_per_sample[]는 매 epoch 마다 계산되지만,
        # 덮어써지는 값들이므로 최종 epoch 끝난 후의 값이다.
        for a in range(K.shape[0]):
            for i in range(snr_dB.shape[0]):
                with open('D_dnn_train.dat', 'a') as f, open('D_opt_train.dat', 'a') as f2:
                    for m in range(input.shape[0]):
                        f.write('%d\t %d\t %g\t %g\n' % (snr_dB[i], K[a], lr, distort_dnn_per_sample[a, i, m]))
                        f2.write('%d\t %d\t %g\t %g\n' % (snr_dB[i], K[a], lr, distort_opt_per_sample[a, i, m]))

                with open('R_dnn_train.dat', 'a') as f, open('R_opt_train.dat', 'a') as f2, open('Cx3_dnn_train.dat', 'a') as f3, open('Cx3_opt_train.dat', 'a') as f4:
                    for m in range(input.shape[0]):
                        f.write('%d\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                        f2.write('%d\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                        f3.write('%d\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                        f4.write('%d\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                        for k in range(self.num_pkt):
                            f.write('%10.10g  ' %dnn_R_per_sample[a, i, m, k])
                            f2.write('%10.10g  ' %opt_R_per_sample[a, i, m, k])
                            f3.write('%10.10g  ' % dnn_Cx3_per_sample[a, i, m, k])
                            f4.write('%10.10g  ' % opt_Cx3_per_sample[a, i, m, k])
                        f.write('\n')
                        f2.write('\n')
                        f3.write('\n')
                        f4.write('\n')


        # train하는 데 걸린 시간
        # print('======== End Time: %s ========\n' %datetime.datetime.now())
        print('======== Elapsed Time: %s (h:m:s) ========\n' % (datetime.datetime.now()-start_time))
        print('======== Elapsed Time: %5.5g (sec) ========\n' % (time.time()-start_time_sec))

        with open('info.dat', 'a') as f:
            f.write(
                '======== Elapsed Time: %s (h:m:s) ========\n'% (datetime.datetime.now()-start_time))



##############################################################################################################
#
# < test_dnn() 프린트 파일들 >
#
# info.dat: Test 마친 후 여러개 값들 프린트
# info_no_name.dat: 상동 (단, 항목 이름 없음)
#
# W1_test.dat, W2_test.dat, ...: Train된 weight를 제대로 읽어왔는지 검증하기 위해 프린트
# B1_test.dat, B2_test.dat, ...: 상동
#
# D_dnn_test.dat: (매 lr별로)(매 K별로)(매 SNR별로) train 끝난 후 최종 업데이트된 W, b로 계산한 distort_dnn_per_sample[s] 프린트
#                  *** 파일형식: num_sample=2, num_snr=2, num_k=2, lr=[lr1, lr2] 일때
#                                         lr1 | k1 | snr1 | distort (sample 1)
#                                         lr1 | k1 | snr1 | distort (sample 2)
#                                         lr1 | k1 | snr2 | distort (sample 1)
#                                         lr1 | k1 | snr2 | distort (sample 2)
#                                         lr1 | k2 | snr1 | distort (sample 1)
#                                         lr1 | k2 | snr1 | distort (sample 2)
#                                         lr1 | k2 | snr2 | distort (sample 1)
#                                         lr1 | k2 | snr2 | distort (sample 2)
#                                         lr2 | k1 | snr1 | distort (sample 1)
#                                         lr2 | k1 | snr1 | distort (sample 2)    ...
#
# D_opt_test.dat: (매 lr별로)(매 K별로)(매 SNR별로) distort_opt_per_sample[] 프린트 (파일형식: lr | k | SNR | distort)
#                  ***파일형식: 상동
#
# R_dnn_test.dat: (매 lr별로)(매 K별로)(매 SNR별로) train 끝난 후 최종 업데이트된 W, b로 계산한 dnn_output_per_sample[s][r] (즉, DNN이 출력한 샘플별 R) 프린트
#                  *** 파일형식: num_sample=2, num_snr=2, num_k=2, lr=[lr1, lr2] 일때
#                                         lr1 | k1 | snr1 | r1 r2 ... (sample 1)
#                                         lr1 | k1 | snr1 | r1 r2 ... (sample 2)
#                                         lr1 | k1 | snr2 | r1 r2 ... (sample 1)
#                                         lr1 | k1 | snr2 | r1 r2 ... (sample 2)
#                                         lr1 | k2 | snr1 | r1 r2 ... (sample 1)
#                                         lr1 | k2 | snr1 | r1 r2 ... (sample 2)
#                                         lr1 | k2 | snr2 | r1 r2 ... (sample 1)
#                                         lr1 | k2 | snr2 | r1 r2 ... (sample 2)
#                                         lr2 | k1 | snr1 | r1 r2 ... (sample 1)
#                                         lr2 | k1 | snr1 | r1 r2 ... (sample 2)    ...
#
# R_opt_test.dat: (매 lr별로)(매 K별로)(매 SNR별로) opt_R_per_sample[s][r] (즉, 샘플별 optimal R) 프린트
#                  *** 파일형식: 상동
#
##############################################################################################################

    # --------------------------------------------------------------------------------------------------------
    # test_dnn 메소드
    #   : DNN 테스트 관련 메소드
    #
    #   - 파라미터
    #       input: distortion-rate curve
    #             (주의) 스케일링 (i.e., normalize) 하지 않은 입력 샘플
    #       snr_dB: dB scale의 SNR
    #       K: rician factor
    #       lr: learning rate
    #       Pout_all_stbc: outage probability
    #
    #   - 알고리즘
    #       1. Settings
    #       2. Normalize the training data
    #       3. graph of tensors
    #       4. label 생성
    #       5. full search를 이용하여 샘플 별 최적의 R, C, E(D) 찾기
    #       6. 텐서 변수인 DNN output session run
    #       7. DNN 출력을 이용하여 expected distortion 계산
    #       8. 결과 출력 (파일, 파이참)
    # --------------------------------------------------------------------------------------------------------
    def test_dnn(self, input, snr_dB, K, lr):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        bound_R_C = self.num_R * self.num_pkt

        # outage probability
        Poutobj = Pout(self.R, self.Cx3, np.array([K]))
        Poutobj.generate_Pout()


        # --------------------------------------------------------------------------------
        # 2. Normalize the training data
        # --------------------------------------------------------------------------------
        input_not_scaled = input
        input_log = np.log10(input)

        with open('input_scaling_param.dat','r') as f:
            lines = f.readlines()

        max_input_log = float(lines[0].strip())
        min_input_log = float(lines[1].strip())
        avg_input_log = float(lines[2].strip())
        std_input_log = float(lines[3].strip())

        if (self.mode_input_scale == 1):
            input = (input_log - min_input_log) / (max_input_log - min_input_log)

        elif (self.mode_input_scale == 2):
            temp = (input_log - min_input_log) / (max_input_log - min_input_log)
            input = 2 * temp - 1

        elif (self.mode_input_scale == 3):
            input = (input_log - avg_input_log) / std_input_log

        else:
            input = input_log


        # --------------------------------------------------------------------------------
        # 3. graph of tensors
        #
        # 1) placeholder
        # 2) 뉴럴 네트워크 통과하는 연산
        # 3) 그래프 실행 전, 변수 초기화
        # --------------------------------------------------------------------------------
        with tf.device('/CPU:0'):
            tf.reset_default_graph()


            # 1) placeholder
            x_ph = tf.placeholder(tf.float64, shape=[None, input.shape[1]+2]) # (num_sample)x4098
            # y_ph = tf.placeholder(tf.float64, shape=[None, self.layer_dim_list[-1]])


            # 2) 뉴럴 네트워크 통과하는 연산
            #   test의 경우, output layer에서 softmax 통과할 필요 없음
            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph  # in_layer: DNN의 input layer 아님. i번째 layer를 가리키는 index (input layer, hidden1, hidden2,..)
                    in_dim = input.shape[1]+2  # in_dim: in_layer의 노드수
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer  # out_layer: DNN의 output layer 아님.
                    in_dim = self.layer_dim_list[i - 1]
                    out_dim = self.layer_dim_list[i]

                weight = np.zeros([in_dim, out_dim], dtype=np.float64)
                bias = np.zeros(out_dim, dtype=np.float64)

                # weight, bias 파일에서 읽어 옴
                weight = file_read("W" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", weight)
                bias = file_read("B" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", bias)

                #Train된 weight를 제대로 읽어왔는지 검증하기 위해 프린트
                file_write('W' + str(i + 1) + '_test.dat', 'w', weight)
                file_write('B' + str(i + 1) + '_test.dat', 'w', bias)

                weight = tf.convert_to_tensor(weight)
                bias = tf.convert_to_tensor(bias)

                mult = tf.matmul(in_layer, weight) + bias

                if i < len(self.layer_dim_list) - 1:    # hidden layer
                    out_layer = tf.nn.relu(mult)
                else:   # output layer
                    ######  각각의 softmax에서 최대 값의 index 찾는 방법 ######
                    # 방법 1
                    # output_R = tf.concat([tf.reshape(tf.argmax(mult[:, num_R * j:num_R * (j + 1)], axis=1), [-1, 1]) + 1 for j in range(self.num_pkt)], 1)
                    # output_C = tf.concat([tf.reshape(tf.argmax(mult[:, num_R * self.num_pkt + num_C * j:num_R * self.num_pkt + num_C * (j + 1)], axis=1), [-1, 1]) + 1 for j in range(self.num_pkt)], 1)

                    # 방법 2: sess.run할 때 numpy로 형태 변환해야 함
                    # output_R = []
                    # output_C = []
                    #
                    # for j in range(self.num_pkt):
                    #     R_pkt = tf.reshape(tf.argmax(mult[:, num_R * j:num_R * (j + 1)], axis=1)+1, [-1, 1])
                    #     C_pkt = tf.reshape(tf.argmax(mult[:, num_R * self.num_pkt + num_C * j:num_R * self.num_pkt + num_C * (j + 1)], axis=1), [-1, 1])
                    #
                    #     # if j == 0:
                    #     #     output_R = R_pkt
                    #     #     output_C = C_pkt
                    #     # else:
                    #     #     output_R = tf.concat([output_R, R_pkt], 1)
                    #     #     output_C = tf.concat([output_C, C_pkt], 1)
                    #
                    #     output_R.append(R_pkt)
                    #     output_C.append(C_pkt)

                    # 방법 1, 2 => R과 C의 개수에 상관없이 사용 가능

                    # 방법 3
                    # optimize graph(test 연산 속도 줄이기 위해 고안해 낸 알고리즘)
                    # R의 개수가 4개, C의 개수가 3개인 상황에서만 사용 가능
                    output_R = tf.argmax([mult[:, 0:bound_R_C:self.num_R], mult[:, 1:bound_R_C:self.num_R], mult[:, 2:bound_R_C:self.num_R], mult[:, 3:bound_R_C:self.num_R]], 0)
                    output_C = tf.argmax([mult[:, bound_R_C + 0::self.num_C], mult[:, bound_R_C + 1::self.num_C], mult[:, bound_R_C + 2::self.num_C]], 0)


            # test의 경우, 손실 함수 계산할 필요 없음
            # cross_entropy = y_ph * -1 * tf.log(output)
            # loss_temp = tf.reduce_sum(cross_entropy, axis=1)
            # loss = tf.reduce_mean(loss_temp)


            # 3) 그래프 실행 전, 변수 초기화
            # Initialization: loss, train등 모든 class 객체를 선언한 후에 해야 함.
            init = tf.global_variables_initializer()  # 위에서 선언한 variable들을 init에 연결함
            sess = tf.Session()
            sess.run(init)  # 세션 실행을 해야 weight와 bias에 초기값이 할당됨.


            # --------------------------------------------------------------------------------
            # 4. label 생성
            #    label shape: (num_sample) x (num_pkt*4+num_pkt*3)
            #                               [R1 R2 R3 ... C1 C2 C3 ...]
            # --------------------------------------------------------------------------------
            filename = 'label\MIMO_TCOM_reference\D_step=' + str(self.D_step) + '\\rician_factor\R_C_tcom_' + str(self.num_pkt) + 'pkt_reference_snr' + str(snr_dB).zfill(2) + '_K' + str(K).zfill(2) + '_R=[1 2 3 4].dat'

            with open(filename, 'r') as f:
                rdr = csv.reader(f, delimiter='\t')
                tcom_temp = np.array(list(rdr), dtype=np.int)
                tcom_opt_R_C = tcom_temp[tcom_temp.shape[0]-input.shape[0]:, 2:].reshape([input.shape[0], self.num_pkt * 2])

            label = np.zeros([input.shape[0], self.num_R * self.num_pkt + self.num_C * self.num_pkt])

            # test label도 train에서 label 만들 때 사용한 방식과 똑같이
            # np.eye: numpy를 이용한 one hot encoding 방법
            # [R label 만들기]
            # ex) R1 = 1이면 1 0 0 0
            tcom_opt_R = tcom_opt_R_C[:, :self.num_pkt] - 1  # input.shape[0]xself.num_pkt    ex) [1, 2]
            one_hot_tcom_opt_R = np.eye(self.num_R)[tcom_opt_R]  # input.shape[0]xself.num_pktxnum_R  ex) [[0, 1, 0, 0], [0, 0, 1, 0]]
            label[:, :self.num_pkt * self.num_R] = np.reshape(one_hot_tcom_opt_R, [input.shape[0], -1])

            # [C label 만들기]
            # ex) C1 = 4이면 0 1 0
            tcom_opt_C = tcom_opt_R_C[:, self.num_pkt:]  # input.shape[0]xself.num_pkt
            tcom_opt_C[tcom_opt_C == 3] = 0
            tcom_opt_C[tcom_opt_C == 4] = 1
            tcom_opt_C[tcom_opt_C == 6] = 2
            one_hot_tcom_opt_C = np.eye(self.num_C)[tcom_opt_C]  # input.shape[0]xself.num_pktxnum_C
            label[:, self.num_pkt * self.num_R:] = np.reshape(one_hot_tcom_opt_C, [input.shape[0], -1])


            # --------------------------------------------------------------------------------
            # 5. full search를 이용하여 샘플 별 최적의 R, C, E(D) 찾기
            # --------------------------------------------------------------------------------
            # Find_Optimal_R_Cx3 실행할때 필요!! (R_case, C_case 먼저 만들어 줌)
            if self.mode_find_optimal == 'enable':
                R_case = product(np.array([1, 2, 3, 4]), repeat=self.num_pkt)
                R_case = np.array(list(R_case))  # R_case: 가능한 R의 모든 조합

                Cx3_case = product(3 * np.array([1, 4 / 3, 2]), repeat=self.num_pkt)
                Cx3_case = np.array(list(Cx3_case))

            distort_opt_per_sample = np.zeros(input.shape[0])
            opt_R_per_sample = np.zeros([input.shape[0],self.num_pkt])
            opt_Cx3_per_sample = np.zeros([input.shape[0], self.num_pkt])


            # find optimal R & Cx3: 샘플별로 Optimal 성능 (distortion) 계산
            Pout_k_snr = Poutobj.get_Pout_k_snr(0, snr_dB)

            for m in range(input.shape[0]):
                if self.mode_find_optimal == 'enable':
                    # distort_opt_per_sample[m], opt_R_per_sample[m], opt_Cx3_per_sample[m] = self.Find_Optimal_R_Cx3(input_not_scaled[m, :], R_case, Cx3_case, snr_dB, Pout_all_stbc)
                    distort_opt_per_sample[m], opt_R_per_sample[m], opt_Cx3_per_sample[m] = self.Find_Optimal_R_Cx3(input_not_scaled[m, :], R_case, Cx3_case, snr_dB, Pout_k_snr)
                else:
                    distort_opt_per_sample[m] = 1000000 # pkt 갯수가 크면 Find_Optimal_R_S이 시간이 오래걸리므로 아무값이나 넣은것


            # --------------------------------------------------------------------------------
            # 6. 텐서 변수인 DNN output session run
            # --------------------------------------------------------------------------------
            # DNN input: D-R curve, rician factor k, snr_dB
            K_vec = np.tile(K, (input.shape[0], 1))
            input_concat = np.concatenate((input, K_vec), axis=1)

            snr_dB_vec = np.tile(snr_dB, (input.shape[0], 1))
            input_concat = np.concatenate((input_concat, snr_dB_vec), axis=1)

            # DNN output
            dnn_R_per_sample = sess.run(output_R, feed_dict={x_ph: input_concat})
            dnn_Cx3_per_sample = sess.run(output_C, feed_dict={x_ph: input_concat})

            sess.close()

        dnn_R_per_sample += 1
        dnn_Cx3_per_sample[dnn_Cx3_per_sample == 2] = 6
        dnn_Cx3_per_sample[dnn_Cx3_per_sample == 1] = 4
        dnn_Cx3_per_sample[dnn_Cx3_per_sample == 0] = 3


        # --------------------------------------------------------------------------------
        # 7. DNN 출력을 이용하여 expected distortion 계산
        #    샘플별로 DNN의 성능 (distortion) 계산 (디버깅 용)
        # --------------------------------------------------------------------------------
        distort_dnn_per_sample = np.zeros(input.shape[0])
        Pout_k_snr = Poutobj.get_Pout_k_snr(0, snr_dB)
        for m in range(input.shape[0]):  # e.g. input.shape[0]=800
            distort_dnn_per_sample[m] = self.Calc_distort(input_not_scaled[m, :], dnn_R_per_sample[m, :], dnn_Cx3_per_sample[m,:], Pout_k_snr)


        # --------------------------------------------------------------------------------
        # 8. 결과 출력 (파일, 파이참)
        # --------------------------------------------------------------------------------
        with open('info.dat', 'a') as f:
            f.write('\n')
        with open('info_no_name.dat', 'a') as f2:
            f2.write('\n')

        # print system setting
        print(
            '#pkts = %6d | max_pkt_size = %6d | snr = %6d | K = %6d | #test_samp = %6d | input_scaling_mode = %d | DNN size = %s'
            % (self.num_pkt, self.max_pkt_size, snr_dB, K, input.shape[0], self.mode_input_scale, self.layer_dim_list))

        with open('info.dat', 'a') as f:
            f.write(
                '#pkts = %6d | max_pkt_size = %6d | snr = %6d | K = %6d | #test_samp = %6d | input_scaling_mode = %d | DNN size = %s\n'
                % (self.num_pkt, self.max_pkt_size, snr_dB, K, input.shape[0], self.mode_input_scale, self.layer_dim_list))

        # print simulation result
        print(
            'mean(distort_opt_per_sample[])= %8.5g | mean(distort_dnn_per_sample[])= %8.5g | distort_ratio = %8.5g (%%)'
            % (np.mean(distort_opt_per_sample), np.mean(distort_dnn_per_sample),
               np.mean(distort_dnn_per_sample) / np.mean(distort_opt_per_sample) * 100))

        with open('info.dat', 'a') as f:
            f.write(
                'mean(distort_opt_per_sample[])= %8.5g | mean(distort_dnn_per_sample[])= %8.5g | distort_ratio = %8.5g (%%)\n'
                % (np.mean(distort_opt_per_sample), np.mean(distort_dnn_per_sample),
                   np.mean(distort_dnn_per_sample) / np.mean(distort_opt_per_sample) * 100))

        with open('info_no_name.dat', 'a') as f2:
            f2.write(
                '%8.5g   %8.5g   %8.5g\n' % (np.mean(distort_opt_per_sample), np.mean(distort_dnn_per_sample),
                    np.mean(distort_dnn_per_sample) / np.mean(distort_opt_per_sample) * 100))



        with open('D_dnn_test.dat', 'a') as f, open('D_opt_test.dat', 'a') as f2:
            for m in range(input.shape[0]):
                # f.write('%g\n'%distort_dnn_per_sample[m])
                # f.write('%d\t %g\n' % (snr_dB, distort_dnn_per_sample[m]))
                f.write('%d\t %d\t %g\t %g\n' % (snr_dB, K, lr, distort_dnn_per_sample[m]))
                # f2.write('%g\n' % distort_opt_per_sample[m])
                f2.write('%d\t %d\t %g\t %g\n' % (snr_dB, K, lr, distort_opt_per_sample[m]))

        with open('R_dnn_test.dat', 'a') as f, open('R_opt_test.dat', 'a') as f2, open('Cx3_dnn_test.dat', 'a') as f3, open('Cx3_opt_test.dat', 'a') as f4:
            for m in range(input.shape[0]):
                # f.write('%d\t' % snr_dB)
                # f2.write('%d\t' % snr_dB)
                f.write('%d\t%d\t%g\t' %(snr_dB, K, lr))
                f2.write('%d\t%d\t%g\t' %(snr_dB, K, lr))
                f3.write('%d\t%d\t%g\t' % (snr_dB, K, lr))
                f4.write('%d\t%d\t%g\t' % (snr_dB, K, lr))
                for i in range(self.num_pkt):
                    f.write('%10.10g  ' %dnn_R_per_sample[m, i])
                    f2.write('%10.10g  ' %opt_R_per_sample[m, i])
                    f3.write('%10.10g  ' % dnn_Cx3_per_sample[m, i])
                    f4.write('%10.10g  ' % opt_Cx3_per_sample[m, i])
                f.write('\n')
                f2.write('\n')
                f3.write('\n')
                f4.write('\n')




    # --------------------------------------------------------------------------------------------------------
    # test_dnn_simple 메소드
    #   : DNN test 시간 측정을 위해 불필요한 부분들 모두 제거한 메소드
    #
    #   - 파라미터
    #       input: distortion-rate curve
    #             (주의) 스케일링 (i.e., normalize) 하지 않은 입력 샘플
    #       snr_dB: dB scale의 SNR
    #       K: rician factor
    #
    #   - 알고리즘
    #       1. Settings
    #       2. Normalize the training data
    #       3. graph of tensors
    #       4. DNN input: D-R curve, rician factor k, snr_dB
    #       5. 텐서 변수인 DNN output session run
    # --------------------------------------------------------------------------------------------------------
    def test_dnn_simple(self, input, snr_dB, K):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        bound_R_C = self.num_R * self.num_pkt

        # --------------------------------------------------------------------------------
        # 2. Normalize the training data
        # --------------------------------------------------------------------------------
        input_log = np.log10(input)

        max_input_log = float(self.lines[0].strip())
        min_input_log = float(self.lines[1].strip())
        avg_input_log = float(self.lines[2].strip())
        std_input_log = float(self.lines[3].strip())

        if (self.mode_input_scale == 1):
            input = (input_log - min_input_log) / (max_input_log - min_input_log)

        elif (self.mode_input_scale == 2):
            temp = (input_log - min_input_log) / (max_input_log - min_input_log)
            input = 2 * temp - 1

        elif (self.mode_input_scale == 3):
            input = (input_log - avg_input_log) / std_input_log

        else:
            input = input_log


        # --------------------------------------------------------------------------------
        # 3. graph of tensors
        #
        # 1) placeholder
        # 2) 뉴럴 네트워크 통과하는 연산
        # 3) 그래프 실행 전, 변수 초기화
        # --------------------------------------------------------------------------------
        with tf.device('/CPU:0'):
            tf.reset_default_graph()


            # 1) placeholder
            x_ph = tf.placeholder(tf.float64, shape=[None, input.shape[1] + 2])  # (num_sample)x1027


            # 2) 뉴럴 네트워크 통과하는 연산
            #   test의 경우, output layer에서 softmax 통과할 필요 없음
            for i in range(len(self.layer_dim_list)):

                if i == 0:
                    in_layer = x_ph  # in_layer: DNN의 input layer 아님. i번째 layer를 가리키는 index (input layer, hidden1, hidden2,..)
                else:
                    in_layer = out_layer  # out_layer: DNN의 output layer 아님.

                weight_layer = tf.convert_to_tensor(self.weights_final[i][:, :])
                bias_layer = tf.convert_to_tensor(self.biases_final[i][:])

                mult = tf.matmul(in_layer, weight_layer) + bias_layer

                if i < len(self.layer_dim_list) - 1:
                    out_layer = tf.nn.relu(mult)
                else:
                    output_R = tf.argmax([mult[:, 0:bound_R_C:self.num_R], mult[:, 1:bound_R_C:self.num_R], mult[:, 2:bound_R_C:self.num_R], mult[:, 3:bound_R_C:self.num_R]], 0)
                    output_C = tf.argmax([mult[:, bound_R_C + 0::self.num_C], mult[:, bound_R_C + 1::self.num_C], mult[:, bound_R_C + 2::self.num_C]], 0)
                    # output_R = tf.argmax([mult[:, 0:bound_R_C:self.num_R], mult[:, 1:bound_R_C:self.num_R], mult[:, 2:bound_R_C:self.num_R], mult[:, 3:bound_R_C:self.num_R]])
                    # output_C = tf.argmax([mult[:, bound_R_C + 0::self.num_C], mult[:, bound_R_C + 1::self.num_C], mult[:, bound_R_C + 2::self.num_C]])


            # 3) 그래프 실행 전, 변수 초기화
            # Initialization: loss, train등 모든 class 객체를 선언한 후에 해야 함.
            init = tf.global_variables_initializer()  # 위에서 선언한 variable들을 init에 연결함
            sess = tf.Session()
            sess.run(init)  # 세션 실행을 해야 weight와 bias에 초기값이 할당됨.


            # --------------------------------------------------------------------------------
            # 4. DNN input: D-R curve, rician factor k, snr_dB
            # --------------------------------------------------------------------------------
            K_vec = np.tile(K, (input.shape[0], 1))
            input_concat = np.concatenate((input, K_vec), axis=1)

            snr_dB_vec = np.tile(snr_dB, (input.shape[0], 1))
            input_concat = np.concatenate((input_concat, snr_dB_vec), axis=1)


            # --------------------------------------------------------------------------------
            # 5. 텐서 변수인 DNN output session run
            # --------------------------------------------------------------------------------
            dnn_R_per_sample = sess.run(output_R, feed_dict={x_ph: input_concat})
            dnn_Cx3_per_sample = sess.run(output_C, feed_dict={x_ph: input_concat})

            sess.close()

        dnn_R_per_sample += 1
        dnn_Cx3_per_sample[dnn_Cx3_per_sample == 2] = 6
        dnn_Cx3_per_sample[dnn_Cx3_per_sample == 1] = 4
        dnn_Cx3_per_sample[dnn_Cx3_per_sample == 0] = 3


