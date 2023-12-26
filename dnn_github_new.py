#========================================================================================================
#
# dnn.py
#
#========================================================================================================

import csv
import time
import datetime
import numpy as np
import tensorflow as tf
import os

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

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
# DNN class
#
#========================================================================================================
class Dnn:
    def __init__(self, batch_size=100,  mode_input_scale = 0, mode_shuffle = 'disable',
                 n_epoch=200, layer_dim_list = [16, 896], max_pkt_size=2**11, num_pkt=128, D_step = 2**12, nr = 2, nt = 2, r=np.array([1,2,3,4]), c=np.array([1,4/3,2]), mode_lr_shift = 'enable'):

        self.n_epoch = n_epoch
        self.layer_dim_list = layer_dim_list
        self.batch_size = batch_size
        self.mode_input_scale = mode_input_scale
        self.mode_shuffle = mode_shuffle

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

        self.mode_lr_shift = mode_lr_shift

    # --------------------------------------------------------------------------------------------------------
    # Calc_distort
    # --------------------------------------------------------------------------------------------------------
    def Calc_distort(self, input, R, Cx3, Pout_all_stbc):

        D = input
        R_int = R.astype('int64')
        E_D = 0

        # --------------------------------------------------------------------------------
        # 1. Outage probability per pkt
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
        # 2. Calculate expected distortion
        # --------------------------------------------------------------------------------
        for success_pkt in range(self.num_pkt, 0, -1):
            # throughput
            total_bits = self.max_pkt_size * (np.sum(R[0:success_pkt] * Cx3[0:success_pkt]) / (self.R_max * self.Cx3_max))

            if len(np.where(R==4)[0]) == self.num_pkt and len(np.where(Cx3==6)[0]) == self.num_pkt:
                D_idx = (total_bits / self.D_step).astype('int32')
                Distortion = D[D_idx]
            else:
                # distortion interpolation
                D_idx1 = (total_bits / self.D_step).astype('int32')
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
    # train_dnn
    # --------------------------------------------------------------------------------------------------------
    def train_dnn(self, input, snr_dB, K, lr, Pout_all, file_path, epoch_step):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        self.weights = []
        self.biases = []

        total_num_sample = input.shape[0] * snr_dB.shape[0] * K.shape[0]
        new_num_batch = int(total_num_sample / self.batch_size)

        seed_weight = 1000
        seed_shuffle = 2000
        np.random.seed(seed_shuffle)


        # --------------------------------------------------------------------------------
        # 2. Normalize the training data
        # --------------------------------------------------------------------------------
        input_not_scaled = input
        input_log = np.log10(input)

        max_input_log = np.max(input_log)
        min_input_log = np.min(input_log)

        avg_input_log = np.mean(input_log)
        std_input_log = np.std(input_log)

        with open(file_path+'input_scaling_param.dat','w') as f:
            f.write('   %g\n   %g\n   %g\n   %g\n' % (max_input_log, min_input_log, avg_input_log, std_input_log))

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
        # 3. Build Model(graph of tensors)
        # --------------------------------------------------------------------------------
        with tf.device('/CPU:0'):
            tf.reset_default_graph()

            # 1) placeholder
            x_ph = tf.placeholder(tf.float64, shape=[None,input.shape[1]+2])
            y_ph = tf.placeholder(tf.float64, shape=[None, self.layer_dim_list[-1]])

            # 2) neural network structure
            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph
                    in_dim = input.shape[1]+2
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer
                    in_dim = self.layer_dim_list[i-1]
                    out_dim = self.layer_dim_list[i]

                weight = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=tf.sqrt(2.0 / tf.cast(in_dim, tf.float64)), seed=seed_weight * (i * i + 1), dtype=tf.float64), dtype=tf.float64)
                bias = tf.Variable(tf.zeros(out_dim, dtype=tf.float64), dtype=tf.float64)

                mult = tf.matmul(in_layer, weight) + bias

                # activation function
                if i < len(self.layer_dim_list)-1:  # hidden layer
                    out_layer = tf.nn.relu(mult)

                else:   # output layer
                    output_prob_R = tf.concat([tf.nn.softmax(mult[:,self.num_R*j:self.num_R*(j+1)]
                                                             - tf.reduce_max(mult[:,self.num_R*j:self.num_R*(j+1)],axis=1,keepdims=True)) for j in range(self.num_pkt)],1)
                    output_prob_C = tf.concat([tf.nn.softmax(mult[:, self.num_R*self.num_pkt + self.num_C * j: self.num_R*self.num_pkt +self.num_C * (j + 1)]
                                                             - tf.reduce_max(mult[:, self.num_R*self.num_pkt + self.num_C * j: self.num_R*self.num_pkt +self.num_C * (j + 1)],axis=1,keepdims=True)) for j in range(self.num_pkt)], 1)
                    output = tf.concat([output_prob_R, output_prob_C], 1)

                    output_R = tf.argmax([output_prob_R[:, 0::self.num_R], output_prob_R[:, 1::self.num_R], output_prob_R[:, 2::self.num_R], output_prob_R[:, 3::self.num_R]], 0)
                    output_C = tf.argmax([output_prob_C[:, 0::self.num_C], output_prob_C[:, 1::self.num_C], output_prob_C[:, 2::self.num_C]], 0)

                self.weights.append(weight)
                self.biases.append(bias)

            # 3) loss function
            cross_entropy = y_ph * -1 * tf.log(output + 1e-100)
            loss_temp = tf.reduce_sum(cross_entropy, axis=1)
            loss = tf.reduce_mean(loss_temp)

            # 4) learning rate scheduling O,X
            if self.mode_lr_shift == 'enable':
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = lr
                lr_shift_period = self.n_epoch * new_num_batch / 2
                lr_shift_rate = 0.3
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, lr_shift_period, lr_shift_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate) # lr: learning_rate
                train = optimizer.minimize(loss, global_step=global_step)

            else:
                optimizer = tf.train.AdamOptimizer(lr)
                train = optimizer.minimize(loss)

            # 5) initialization
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            start_time_sec = time.time()
            start_time = datetime.datetime.now()
            print('======== Start Time: %s ========\n' % start_time)

            # --------------------------------------------------------------------------------
            # 4. Label
            # --------------------------------------------------------------------------------
            label = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_R * self.num_pkt + self.num_C * self.num_pkt], dtype=int)

            for k in range(K.shape[0]):
                for i in range(snr_dB.shape[0]):

                    one_snr_dB = snr_dB[i]
                    if one_snr_dB % 1 == 0:
                        one_snr_dB = int(one_snr_dB)

                    filename = 'label\R_C_tcom_' + str(self.num_pkt) + 'pkt_reference_snr' + str(one_snr_dB).zfill(2) + '_K' + str(K[k]).zfill(2) + '_R=[1 2 3 4].dat'
                    with open(filename, 'r') as f:
                        rdr = csv.reader(f, delimiter='\t')
                        tcom_temp = np.array(list(rdr), dtype=np.float)
                        tcom_opt_R_C = tcom_temp[:input.shape[0], 2:].reshape([input.shape[0], self.num_pkt * 2])

                    tcom_opt_R_C = np.array(tcom_opt_R_C, dtype=np.int)

                    # one-hot encoding: spectral efficiency
                    tcom_opt_R = tcom_opt_R_C[:, :self.num_pkt]-1
                    one_hot_tcom_opt_R = np.eye(self.num_R)[tcom_opt_R]      #input.shape[0]xself.num_pktxnum_R  ex) [[0, 1, 0, 0], [0, 0, 1, 0]]
                    label[k, i, :, :self.num_pkt * self.num_R] = np.reshape(one_hot_tcom_opt_R,[input.shape[0], -1])

                    # one-hot encoding: space-time coding
                    tcom_opt_C = tcom_opt_R_C[:, self.num_pkt:]         #input.shape[0]xself.num_pkt
                    tcom_opt_C[tcom_opt_C == 3] = 0
                    tcom_opt_C[tcom_opt_C == 4] = 1
                    tcom_opt_C[tcom_opt_C == 6] = 2
                    one_hot_tcom_opt_C = np.eye(self.num_C)[tcom_opt_C]      #input.shape[0]xself.num_pktxnum_C
                    label[k, i, :, self.num_pkt * self.num_R:] = np.reshape(one_hot_tcom_opt_C, [input.shape[0], -1])


            # ----------------------------------------------------------------------------------------
            # 5. New sampling
            # ----------------------------------------------------------------------------------------
            # 1) total_sample
            K_vec = np.repeat(K, snr_dB.shape[0]).reshape([-1, 1])  # (13x4, 1)
            snr_vec = np.tile(snr_dB, K.shape[0]).reshape([-1, 1])  # (13x4, 1)

            for i in range(input.shape[0]):
                img_vec = np.tile(input[i], (K.shape[0] * snr_dB.shape[0], 1))
                sample_img_k = np.concatenate((img_vec, K_vec), axis=1)  # (13x4, 65+1)
                sample_img_k_snr = np.concatenate((sample_img_k, snr_vec), axis=1)  # (13x4, 65+1 +1)

                if i == 0:
                    total_sample = sample_img_k_snr
                else:
                    total_sample = np.concatenate((total_sample, sample_img_k_snr), axis=0)  # 밑으로 붙임

            # 2) total_label
            total_label = np.zeros([total_num_sample, self.num_R * self.num_pkt + self.num_C * self.num_pkt], dtype=int)
            total_idx = 0

            for i in range(input.shape[0]):
                for k in range(K.shape[0]):
                    for s in range(snr_dB.shape[0]):
                        total_label[total_idx, :] = label[k, s, i, :]
                        total_idx = total_idx + 1


            # --------------------------------------------------------------------------------
            # 6. Train
            # --------------------------------------------------------------------------------
            dnn_R_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt])
            dnn_Cx3_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt])
            dnn_R_propability = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt*self.num_R])
            dnn_Cx3_propability = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_pkt*self.num_C])
            distort_dnn_per_sample = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0]])

            loss_per_batch = np.zeros([new_num_batch])
            loss_per_epoch = np.zeros([self.n_epoch])


            for e in range(self.n_epoch):
                for j in range(new_num_batch):  # new_num_batch = 800*13*4 / batch_size
                    total_input_batch = total_sample[j * self.batch_size: (j + 1) * self.batch_size]  # (batch_size, 65+1+1)
                    total_label_batch = total_label[j * self.batch_size: (j + 1) * self.batch_size]  # (batch_size, 128*(4+3))

                    distort_dnn_per_batch_tmp = sess.run(loss,feed_dict={x_ph: total_input_batch, y_ph: total_label_batch})
                    loss_per_batch[j] = distort_dnn_per_batch_tmp

                    # Back propagation
                    sess.run(train, feed_dict={x_ph: total_input_batch, y_ph: total_label_batch})

                    if self.mode_lr_shift == 'enable':
                        lr_shift = sess.run(optimizer._lr)
                    else:
                        lr_shift = lr

                loss_per_epoch[e] = np.mean(loss_per_batch)

                # --------------------------------------------------------------------------------
                # 7. Train result per epoch
                # --------------------------------------------------------------------------------
                ##########################       K       ##########################
                for a in range(K.shape[0]):
                    K_vec = np.tile(K[a], (input.shape[0], 1))
                    input_concat_K = np.concatenate((input, K_vec), axis=1)
                    ##########################       SNR       ##########################
                    for i in range(snr_dB.shape[0]):
                        snr_dB_vec = np.tile(snr_dB[i], (input.shape[0], 1))
                        input_concat = np.concatenate((input_concat_K, snr_dB_vec), axis=1)

                        # 1) session run
                        dnn_R_per_sample[a, i] = sess.run(output_R, feed_dict={x_ph: input_concat}) + 1
                        dnn_Cx3_per_sample[a, i] = sess.run(output_C, feed_dict={x_ph: input_concat})
                        dnn_Cx3_per_sample[a, i, dnn_Cx3_per_sample[a, i] == 2] = 6
                        dnn_Cx3_per_sample[a, i, dnn_Cx3_per_sample[a, i] == 1] = 4
                        dnn_Cx3_per_sample[a, i, dnn_Cx3_per_sample[a, i] == 0] = 3

                        # DNN softmax output
                        dnn_R_propability[a, i] = sess.run(output_prob_R, feed_dict={x_ph: input_concat})
                        dnn_Cx3_propability[a, i] = sess.run(output_prob_C, feed_dict={x_ph: input_concat})

                        # DNN weights, biases
                        ww, bb = sess.run([self.weights, self.biases])

                        # 2) Calculate expected distortion
                        for m in range(input.shape[0]):
                            snr_idx = np.where(Pout_all[0, 0, :, 0] == snr_dB[i])
                            distort_dnn_per_sample[a, i, m] = self.Calc_distort(input_not_scaled[m, :],dnn_R_per_sample[a, i, m, :],dnn_Cx3_per_sample[a, i, m, :],np.squeeze(Pout_all[a, :, snr_idx[0], 1:]))

                # 3) write
                if (e + 1) % 5 == 0:
                    now_time = datetime.datetime.now()
                    remain_time = (now_time - start_time) * self.n_epoch / (e + 1) - (now_time - start_time)
                    print('epoch= %6d | lr= %6g | loss_per_epoch= %8.5g | remain = %s(h:m:s)' % (e + 1, lr_shift, loss_per_epoch[e], remain_time))

                if e == self.n_epoch - 1:
                    for i in range(len(self.layer_dim_list)):
                        file_write(file_path + 'W' + str(i + 1) + '_lr' + str(format(lr, "1.0e")) + '.dat', 'w',ww[i][:, :])
                        file_write(file_path + 'B' + str(i + 1) + '_lr' + str(format(lr, "1.0e")) + '.dat', 'w',bb[i][:])

                if (e + 1) % epoch_step == 0 and e != self.n_epoch - 1:
                    ep_file_path = file_path + '/ep=' + str(e + 1) + '/'
                    if not os.path.exists(ep_file_path):
                        os.makedirs(ep_file_path)
                    for i in range(len(self.layer_dim_list)):
                        file_write(ep_file_path + 'W' + str(i + 1) + '_lr' + str(format(lr, "1.0e")) + '_ep' + str(e + 1) + '.dat', 'w', ww[i][:, :])
                        file_write(ep_file_path + 'B' + str(i + 1) + '_lr' + str(format(lr, "1.0e")) + '_ep' + str(e + 1) + '.dat', 'w', bb[i][:])
                    for a in range(K.shape[0]):
                        for i in range(snr_dB.shape[0]):
                            with open(ep_file_path + 'D_dnn_train_ep' + str(e + 1) + '.dat', 'a') as f:
                                for m in range(input.shape[0]):
                                    f.write('%g\t %d\t %g\t %g\n' % (snr_dB[i], K[a], lr, distort_dnn_per_sample[a, i, m]))

                            with open(ep_file_path + 'R_dnn_train_ep' + str(e + 1) + '.dat', 'a') as f, open(ep_file_path + 'Cx3_dnn_train_ep' + str(e + 1) + '.dat', 'a') as f3:
                                for m in range(input.shape[0]):
                                    f.write('%g\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                                    f3.write('%g\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                                    for k in range(self.num_pkt):
                                        f.write('%10.10g  ' % dnn_R_per_sample[a, i, m, k])
                                        f3.write('%10.10g  ' % dnn_Cx3_per_sample[a, i, m, k])
                                    f.write('\n')
                                    f3.write('\n')
                    epoch = np.arange(0, e + 1, 1)
                    with open(ep_file_path + 'loss_per_ep' + str(e + 1) + '.dat', 'w') as f7:
                        for j in range(e + 1):
                            f7.write('%d\t %8.10g\n' % (epoch[j] + 1, loss_per_epoch[j]))
                        f7.close()

            sess.close()


        # --------------------------------------------------------------------------------
        # 8. Train result for last epoch
        # --------------------------------------------------------------------------------
        for a in range(K.shape[0]):
            for i in range(snr_dB.shape[0]):
                with open(file_path + 'D_dnn_train.dat', 'a') as f:
                    for m in range(input.shape[0]):
                        f.write('%g\t %d\t %g\t %g\n' % (snr_dB[i], K[a], lr, distort_dnn_per_sample[a, i, m]))

                with open(file_path + 'R_dnn_train.dat', 'a') as f, open(file_path + 'Cx3_dnn_train.dat', 'a') as f3:
                    for m in range(input.shape[0]):
                        f.write('%g\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                        f3.write('%g\t%d\t%g\t' % (snr_dB[i], K[a], lr))
                        for k in range(self.num_pkt):
                            f.write('%10.10g  ' % dnn_R_per_sample[a, i, m, k])
                            f3.write('%10.10g  ' % dnn_Cx3_per_sample[a, i, m, k])
                        f.write('\n')
                        f3.write('\n')

        epoch = np.arange(0, self.n_epoch, 1)
        with open(file_path + 'loss_per_epoch.dat', 'w') as f7:
            for j in range(self.n_epoch):
                f7.write('%d\t %8.10g\n' % (epoch[j] + 1, loss_per_epoch[j]))
            f7.close()

        print('======== Elapsed Time: %s (h:m:s) ========\n' % (datetime.datetime.now() - start_time))
        print('======== Elapsed Time: %5.5g (sec) ========\n' % (time.time() - start_time_sec))



    # --------------------------------------------------------------------------------------------------------
    # test_dnn
    # --------------------------------------------------------------------------------------------------------
    def test_dnn(self, input, snr_dB, K, lr, Pout_all_stbc, file_path):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        bound_R_C = self.num_R * self.num_pkt

        # --------------------------------------------------------------------------------
        # 2. Normalize the training data
        # --------------------------------------------------------------------------------
        input_not_scaled = input
        input_log = np.log10(input)

        with open(file_path + 'input_scaling_param.dat', 'r') as f:
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
        # 3. Build Model (graph of tensors)
        # --------------------------------------------------------------------------------
        with tf.device('/CPU:0'):
            tf.reset_default_graph()

            # 1) placeholder
            x_ph = tf.placeholder(tf.float64, shape=[None, input.shape[1] + 2])  # (num_sample)x4098

            # 2) neural network structure
            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph
                    in_dim = input.shape[1] + 2
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer
                    in_dim = self.layer_dim_list[i - 1]
                    out_dim = self.layer_dim_list[i]

                weight = np.zeros([in_dim, out_dim], dtype=np.float64)
                bias = np.zeros(out_dim, dtype=np.float64)

                weight = file_read(file_path + "W" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", weight)
                bias = file_read(file_path + "B" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", bias)

                weight = tf.convert_to_tensor(weight)
                bias = tf.convert_to_tensor(bias)

                mult = tf.matmul(in_layer, weight) + bias

                if i < len(self.layer_dim_list) - 1:  # hidden layer
                    out_layer = tf.nn.relu(mult)
                else:  # output layer
                    output_R = tf.argmax([mult[:, 0:bound_R_C:self.num_R], mult[:, 1:bound_R_C:self.num_R],mult[:, 2:bound_R_C:self.num_R], mult[:, 3:bound_R_C:self.num_R]], 0)
                    output_C = tf.argmax([mult[:, bound_R_C + 0::self.num_C], mult[:, bound_R_C + 1::self.num_C],mult[:, bound_R_C + 2::self.num_C]], 0)

            # 3) initialization
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)


            # --------------------------------------------------------------------------------
            # 4. DNN output session run
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
        # 5. Calculate expected distortion
        # --------------------------------------------------------------------------------
        distort_dnn_per_sample = np.zeros(input.shape[0])
        for m in range(input.shape[0]):
            distort_dnn_per_sample[m] = self.Calc_distort(input_not_scaled[m, :], dnn_R_per_sample[m, :],dnn_Cx3_per_sample[m, :], Pout_all_stbc)

        # --------------------------------------------------------------------------------
        # 6. Write test result
        # --------------------------------------------------------------------------------
        with open(file_path + "/" + 'D_dnn_test.dat', 'a') as f:
            for m in range(input.shape[0]):
                f.write('%g\t %d\t %g\t %g\n' % (snr_dB, K, lr, distort_dnn_per_sample[m]))

        with open(file_path + "/" + 'R_dnn_test.dat', 'a') as f, open(file_path + "/" + 'Cx3_dnn_test.dat', 'a') as f3:
            for m in range(input.shape[0]):
                f.write('%g\t%d\t%g\t' % (snr_dB, K, lr))
                f3.write('%g\t%d\t%g\t' % (snr_dB, K, lr))
                for i in range(self.num_pkt):
                    f.write('%10.10g  ' % dnn_R_per_sample[m, i])
                    f3.write('%10.10g  ' % dnn_Cx3_per_sample[m, i])
                f.write('\n')
                f3.write('\n')
