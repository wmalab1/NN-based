#========================================================================================================
#
# dnn.py
#
#========================================================================================================

import csv
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


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

#========================================================================================================
#
# DNN class
#
#========================================================================================================
class Dnn:
    def __init__(self, batch_size=100, n_epoch=200, layer_dim_list = [16, 896], num_pkt=128, D_step = 2**12, r=np.array([1,2,3,4]), c=np.array([1,4/3,2])):
        self.n_epoch = n_epoch
        self.layer_dim_list = layer_dim_list
        self.batch_size = batch_size

        self.num_pkt = num_pkt

        self.D_step = D_step

        self.num_R = r.shape[1]
        self.num_C = c.shape[1]

        self.weight = []
        self.bias = []

        self.max_input_log = 0
        self.min_input_log = 0



    # --------------------------------------------------------------------------------------------------------
    # train_dnn
    # --------------------------------------------------------------------------------------------------------
    def train_dnn(self, input, snr_dB, K, lr):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        weights = []
        biases = []

        num_batch = int(input.shape[0] / self.batch_size)

        seed_weight = 1000
        seed_shuffle = 2000
        np.random.seed(seed_shuffle)


        # --------------------------------------------------------------------------------
        # 2. normalize the training data
        # --------------------------------------------------------------------------------
        input_log = np.log10(input)

        self.max_input_log = np.max(input_log)
        self.min_input_log = np.min(input_log)

        temp = (input_log - self.min_input_log) / (self.max_input_log - self.min_input_log)
        input = 2 * temp - 1


        # --------------------------------------------------------------------------------
        # 3. Build Model(graph of tensors)
        # --------------------------------------------------------------------------------
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

            weights.append(weight)
            biases.append(bias)

        # 3) loss function
        cross_entropy = y_ph * -1 * tf.log(output + 1e-100)
        loss_temp = tf.reduce_sum(cross_entropy, axis=1)
        loss = tf.reduce_mean(loss_temp)

        # 4) learning rate scheduling
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = lr
        lr_shift_period = self.n_epoch * K.shape[0] * snr_dB.shape[0] * num_batch / 2
        lr_shift_rate = 0.3

        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, lr_shift_period, lr_shift_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)  # lr: learning_rate
        train = optimizer.minimize(loss, global_step=global_step)

        # 5) initialization
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)


        # --------------------------------------------------------------------------------
        # 4. label
        # --------------------------------------------------------------------------------
        label = np.zeros([K.shape[0], snr_dB.shape[0], input.shape[0], self.num_R * self.num_pkt + self.num_C * self.num_pkt], dtype=int)

        for k in range(K.shape[0]):
            for i in range(snr_dB.shape[0]):
                # get labels from files
                filename = 'label\MIMO_TCOM_reference\D_step=' + str(self.D_step) + '\\rician_factor\R_C_tcom_' + str(self.num_pkt) + 'pkt_reference_snr' + str(snr_dB[i]).zfill(2) + '_K' + str(K[k]).zfill(2) + '_R=[1 2 3 4].dat'
                with open(filename, 'r') as f:
                    rdr = csv.reader(f, delimiter='\t')
                    tcom_temp = np.array(list(rdr), dtype=np.int)
                    tcom_opt_R_C = tcom_temp[:input.shape[0], 2:].reshape([input.shape[0], self.num_pkt * 2])

                # one-hot encoding: spectral efficiency
                tcom_opt_R = tcom_opt_R_C[:, :self.num_pkt]-1       #input.shape[0]xself.num_pkt    ex) [1, 2]
                one_hot_tcom_opt_R = np.eye(self.num_R)[tcom_opt_R]      #input.shape[0]xself.num_pktxnum_R    ex) [[0, 1, 0, 0], [0, 0, 1, 0]]
                label[k, i, :, :self.num_pkt * self.num_R] = np.reshape(one_hot_tcom_opt_R,[input.shape[0], -1])

                # one-hot encoding: space-time coding
                tcom_opt_C = tcom_opt_R_C[:, self.num_pkt:]         #input.shape[0]xself.num_pkt
                tcom_opt_C[tcom_opt_C == 3] = 0
                tcom_opt_C[tcom_opt_C == 4] = 1
                tcom_opt_C[tcom_opt_C == 6] = 2
                one_hot_tcom_opt_C = np.eye(self.num_C)[tcom_opt_C]      #input.shape[0]xself.num_pktxnum_C
                label[k, i, :, self.num_pkt * self.num_R:] = np.reshape(one_hot_tcom_opt_C, [input.shape[0], -1])


        # --------------------------------------------------------------------------------
        # 5. train
        # --------------------------------------------------------------------------------
        ##########################       epoch       ##########################
        for e in range(self.n_epoch):

            # shuffle
            label_shuffle = np.zeros_like(label)
            input_shuffle = shuffle(input, random_state=seed_shuffle*e)
            for k in range(K.shape[0]):
                for i in range(snr_dB.shape[0]):
                    label_shuffle[k,i,:] = shuffle(label[k,i,:], random_state=seed_shuffle*e)

            ##########################       batch       ##########################
            for j in range(num_batch):
                input_batch = input_shuffle[j * self.batch_size: (j + 1) * self.batch_size]

                ##########################       K       ##########################
                for a in range(K.shape[0]):
                    K_vec = np.tile(K[a], (self.batch_size, 1))
                    input_batch_k = np.concatenate((input_batch, K_vec), axis=1)

                    ##########################       snr       ##########################
                    for i in range(snr_dB.shape[0]):
                        snr_dB_vec = np.tile(snr_dB[i], (self.batch_size, 1))
                        input_batch_k_snr = np.concatenate((input_batch_k, snr_dB_vec), axis=1)

                        label_batch = label_shuffle[a, i, j * self.batch_size: (j + 1) * self.batch_size]

                        # loss
                        distort_dnn_per_batch = sess.run(loss, feed_dict={x_ph: input_batch_k_snr, y_ph: label_batch})
                        distort_dnn_per_batch = self.batch_size * distort_dnn_per_batch

                        # Back propagation
                        sess.run(train, feed_dict={x_ph: input_batch_k_snr, y_ph: label_batch})

        ##########################################################
        # End of "epoch loop"
        ##########################################################

        # --------------------------------------------------------------------------------
        # 6. DNN weights, biases
        # --------------------------------------------------------------------------------
        self.weight, self.bias = sess.run([weights, biases])

        sess.close()




    # --------------------------------------------------------------------------------------------------------
    # test_dnn
    # --------------------------------------------------------------------------------------------------------
    def test_dnn(self, input, snr_dB, K):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        bound_R_C = self.num_R * self.num_pkt


        # --------------------------------------------------------------------------------
        # 2. Normalize the training data
        # --------------------------------------------------------------------------------
        input_log = np.log10(input)

        temp = (input_log - self.min_input_log) / (self.max_input_log - self.min_input_log)
        input = 2 * temp - 1


        # --------------------------------------------------------------------------------
        # 3. Build Model(graph of tensors)
        # --------------------------------------------------------------------------------
        tf.reset_default_graph()

        # 1) placeholder
        x_ph = tf.placeholder(tf.float64, shape=[None, input.shape[1]+2]) # (num_sample)x4098

        # 2) neural network structure
        for i in range(len(self.layer_dim_list)):
            if i == 0:
                in_layer = x_ph
            else:
                in_layer = out_layer

            weight_layer = tf.convert_to_tensor(self.weight[i][:, :])
            bias_layer = tf.convert_to_tensor(self.bias[i][:])

            mult = tf.matmul(in_layer, weight_layer) + bias_layer

            if i < len(self.layer_dim_list) - 1:    # hidden layer
                out_layer = tf.nn.relu(mult)
            else:   # output layer
                output_R = tf.argmax([mult[:, 0:bound_R_C:self.num_R], mult[:, 1:bound_R_C:self.num_R], mult[:, 2:bound_R_C:self.num_R], mult[:, 3:bound_R_C:self.num_R]], 0)
                output_C = tf.argmax([mult[:, bound_R_C + 0::self.num_C], mult[:, bound_R_C + 1::self.num_C], mult[:, bound_R_C + 2::self.num_C]], 0)

        # 5) initialization
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