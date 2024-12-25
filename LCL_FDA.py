from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
import math
from utils.sampling import tf_sampling


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def sampling(batch_size, npoint, pts, feature=None):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * D, input point cloud
    output:
    sub_pts: B * npoint * D, sub-sampled point cloud
    '''
    fps_idx = tf_sampling.farthest_point_sample(npoint, pts)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, npoint,1))
    idx = tf.concat([batch_indices, tf.expand_dims(fps_idx, axis=2)], axis=2)
    idx.set_shape([batch_size, npoint, 2])
    if feature is None:
        return tf.gather_nd(pts, idx)
    else:
        return tf.gather_nd(pts, idx), tf.gather_nd(feature, idx)
def sampling1(batch_size, npoint, pts, feature=None):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * D, input point cloud
    output:
    sub_pts: B * npoint * D, sub-sampled point cloud
    fps_idx: B * npoint
    '''
    fps_idx = tf_sampling.farthest_point_sample(npoint, pts)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, npoint,1))
    idx = tf.concat([batch_indices, tf.expand_dims(fps_idx, axis=2)], axis=2)
    idx.set_shape([batch_size, npoint, 2])
    if feature is None:
        return tf.gather_nd(pts, idx)
    else:
        return tf.gather_nd(pts, idx), tf.gather_nd(feature, idx),fps_idx

class Network:
    def __init__(self, dataset, config, restore_snap):
        flat_inputs = dataset.flat_inputs
        self.config = config
        self.restore_snap = restore_snap
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['features'] = flat_inputs[0]
            self.inputs['labels'] = flat_inputs[1]
            self.inputs['input_inds'] = flat_inputs[2]
            self.inputs['cloud_inds'] = flat_inputs[3]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            self.time_stamp = time.strftime('_%Y-%m-%d_%H-%M-%S', time.gmtime())
            self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + self.time_stamp + '.txt', 'a')

        with tf.variable_scope('layers'):
            # self.logits, self.new_xyz, self.xyz = self.inference(self.inputs, self.is_training)
            # self.logits = self.inference(self.inputs, self.is_training)
            self.logits, self.feature_out, self.fp_idx_list = self.inference(self.inputs, self.is_training)

        with tf.variable_scope('loss'):
            self.labels = tf.reshape(self.labels, [-1])
            valid_logits, valid_labels = self.valid_logits_label(self.logits, self.labels, self.config.num_classes,
                                                                 self.config.ignored_label_inds)
            loss_a = self.get_loss(valid_logits, valid_labels, self.class_weights)
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            loss1 = []
            for i in range(len(self.feature_out)):
                valid_logits_j, valid_labels_j = self.valid_logits_label(self.feature_out[i], self.labels,
                                                                         self.config.num_classes,
                                                                         self.config.ignored_label_inds)
                loss_b = self.get_loss(valid_logits_j, valid_labels_j, self.class_weights)
                loss1.append(loss_b)
            self.loss = loss_a + sum(loss1) / (len(loss1) + 1e-6)



        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if self.restore_snap is not None:
            self.saver.restore(self.sess, self.restore_snap)

    def inference(self, inputs, is_training):

        d_out = self.config.d_out  ##[16, 64, 128, 256, 512]
        ratio = self.config.sub_sampling_ratio ##[4, 4, 4, 4, 2]
        k_n = self.config.k_n  ##16
        feature = inputs['features']
        og_xyz = feature[:, :, :3]
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')  ##[B,N,C]
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)  ##[B,N,1,C]

        # ###########################Encoder############################
        f_encoder_list = []
        input_xyz = og_xyz
        input_up_samples = []
        new_xyz_list = []
        xyz_list = []
        n_pts = self.config.num_points
        input_sub_idx_list = []
        fp_idx_list = []
        for i in range(self.config.num_layers):
            input_neigh_idx = tf.py_func(DP.knn_search, [input_xyz, input_xyz, k_n], tf.int32)
            n_pts = n_pts // ratio[i]
            sub_xyz, inputs_sub_idx, fp_idx = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: sampling1(self.config.batch_size, n_pts, input_xyz, input_neigh_idx), lambda: sampling1(self.config.val_batch_size, n_pts, input_xyz, input_neigh_idx))
            input_sub_idx_list.append(inputs_sub_idx)
            fp_idx_list.append(fp_idx)
            inputs_interp_idx = tf.py_func(DP.knn_search, [sub_xyz, input_xyz, 1], tf.int32)
            input_up_samples.append(inputs_interp_idx)
            f_encoder_i = self.bilateral_context_block(feature, input_xyz, input_neigh_idx, d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs_sub_idx)
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
            xyz_list.append(input_xyz)
            # new_xyz_list.append(new_xyz)
            input_xyz = sub_xyz
        # ###########################Encoder############################


        # Adaptive Fusion Module
        f_decoder_0 = []  # full-sized feature maps
        f_decoder_1 = []
        f_decoder_2 = []
        f_decoder_3 = []
        f_decoder_4 = []
        f_multi_decode = []


        ### 0
        f_multi_decoder_0 = []
        for j in range(0, self.config.num_layers):
            feature_i = self.feature2(f_encoder_list[j], f_encoder_list[j + 1],  input_up_samples[j],
                                      'f_decoder0' + '_' + str(j), is_training)
            f_multi_decoder_0.append(feature_i)
        idx_down_list_0 = [input_sub_idx_list[0],input_sub_idx_list[1],input_sub_idx_list[2]]
        f_0 = self.down_sample_fusion0(f_encoder_list[-2], f_multi_decoder_0, idx_down_list_0, 'f_0' + '_' + str(0), is_training)
        del f_multi_decoder_0[-1]
        f_multi_decoder_0.append(f_0)
        f_multi_decode.append(f_multi_decoder_0)


        ### 1
        f_multi_decoder_1 = []
        for j in range(0, self.config.num_layers - 1):
            feature_i = self.feature2( f_multi_decoder_0[j], f_multi_decoder_0[j+1], input_up_samples[j],
                                      'f_decoder1' + '_' + str(j), is_training)
            f_multi_decoder_1.append(feature_i)

        idx_down_list_1 = [input_sub_idx_list[0], input_sub_idx_list[1]]
        idx_up_list_1 = [input_up_samples[-1], input_up_samples[-2]]
        up_list_1 = [f_encoder_list[-1]]
        f_1 = self.down_sample_fusion(f_multi_decoder_0[-2], f_multi_decoder_1, idx_down_list_1, up_list_1, idx_up_list_1, 'f_1' + '_' + str(1), is_training)
        del f_multi_decoder_1[-1]
        f_multi_decoder_1.append(f_1)
        f_multi_decode.append(f_multi_decoder_1)


        ### 2
        f_multi_decoder_2 = []
        for j in range(0, self.config.num_layers - 2):
            feature_i = self.feature2(f_multi_decoder_1[j], f_multi_decoder_1[j + 1], input_up_samples[j],
                                      'f_decoder2' + '_' + str(j), is_training)
            f_multi_decoder_2.append(feature_i)

        idx_down_list_2 = [input_sub_idx_list[0]]
        idx_up_list_2 = [input_up_samples[-1], input_up_samples[-2],input_up_samples[-3]]
        up_list_2 = [f_encoder_list[-1], f_multi_decoder_0[-1]]
        f_2 = self.down_sample_fusion(f_multi_decoder_1[-2],f_multi_decoder_2, idx_down_list_2, up_list_2, idx_up_list_2,
                             'f_2' + '_' + str(2), is_training)
        del f_multi_decoder_2[-1]
        f_multi_decoder_2.append(f_2)
        f_multi_decode.append(f_multi_decoder_2)


        ### 3
        f_multi_decoder_3 = []
        for j in range(0, self.config.num_layers - 3):
            feature_i = self.feature2(f_multi_decoder_2[j], f_multi_decoder_2[j + 1], input_up_samples[j],
                                      'f_decoder3' + '_' + str(j), is_training)
            f_multi_decoder_3.append(feature_i)

        # idx_down_list_3 = [input_sub_idx_list[0]]
        idx_up_list_3 = [input_up_samples[-1], input_up_samples[-2],input_up_samples[-3],input_up_samples[-4]]
        up_list_3 = [f_encoder_list[-1], f_multi_decoder_0[-1],f_multi_decoder_1[-1]]
        f_3 = self.down_sample_fusion1(f_multi_decoder_2[-2],f_multi_decoder_3,  up_list_3, idx_up_list_3,
                             'f_3' + '_' + str(3), is_training)
        del f_multi_decoder_3[-1]
        f_multi_decoder_3.append(f_3)
        f_multi_decode.append(f_multi_decoder_3)

        # ###########################Decoder############################
        f_layer_fc1 = helper_tf_util.conv2d(f_3, 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        # print('f_layer_fc1:',f_layer_fc1.get_shape().as_list())
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)

        feature_decoder = []
        for i in range(len(f_multi_decode)):
            feature = f_multi_decode[i][-1]
            if i == 0:
                feature = self.nearest_interpolation1(feature, [input_up_samples[-1],input_up_samples[-2],input_up_samples[-3],input_up_samples[-4]], 4)
            elif i== 1:
                feature = self.nearest_interpolation1(feature, [input_up_samples[-2],input_up_samples[-3],input_up_samples[-4]], 3)
            elif i == 2:
                feature = self.nearest_interpolation1(feature, [input_up_samples[-3],input_up_samples[-4]], 2)
            elif i == 3:
                feature = self.nearest_interpolation1(feature, [input_up_samples[-4]], 1)
            feature = helper_tf_util.conv2d(feature, self.config.num_classes, [1, 1], 'fc'+'_'+str(i)+'_', [1, 1], 'VALID', True,
                                        is_training, activation_fn=tf.nn.sigmoid)
            feature_decoder.append(feature)
        f_out = tf.squeeze(f_layer_fc3, [2])
        # return f_out, new_xyz_list, xyz_list
        return f_out, feature_decoder, fp_idx_list
    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss



    def bilateral_context_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        """
        Inputs:
        feature: [B, N, 1, c] input features
        xyz: [B, N, 3] input coordinates
        neigh_idx: [B, N, k] indices of k neighbors

        Output:
        output_feat: [B, N, 1, 2*d_out] encoded (output) features
        shifted_neigh_xyz: [B, N, k, 3] shifted neighbor coordinates, for augmentation loss
        """
        batch_size = tf.shape(xyz)[0]  ##B
        num_points = tf.shape(xyz)[1]  ##N

        # Input Encoding
        feature = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)  # [B,N,1,C]

        # Bilateral Augmentation
        neigh_feat = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        neigh_xyz = self.gather_neighbour(xyz, neigh_idx) # B, N, k, 3
        tile_feat = tf.tile(feature, [1, 1, self.config.k_n, 1]) # B, N, k, d_out/2
        tile_xyz = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, self.config.k_n, 1]) # B, N, k, 3


        # [relative_alpha,relative_beta,relative_dis]
        relative_xyz = tile_xyz - neigh_xyz
        # print('a1', relative_xyz[:, :, :, 1].get_shape().as_list())   relative_xyz[:, :, :, 1]  B,N,16
        relative_alpha = tf.expand_dims(tf.atan2(relative_xyz[:, :, :, 1], relative_xyz[:, :, :, 0]),
                                        axis=-1)
        relative_xydis = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(relative_xyz[:, :, :, :2]), axis=-1)),
                                        axis=-1)

        relative_beta = tf.expand_dims(tf.atan2(relative_xyz[:, :, :, 2], tf.squeeze(relative_xydis, axis=-1)),
                                       axis=-1)

        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))

        xyz_1 = tf.concat([relative_xyz, relative_dis], axis = -1)   ## B,N,K,4
        xyz_2 = tf.concat([relative_alpha, relative_beta, relative_dis, tf.expand_dims(relative_xyz[:, :, :, 2], axis=-1)], axis = -1)   ## B,N,K,4

        ##：
        xyz1_11 = helper_tf_util.conv2d(xyz_1, xyz_1.get_shape()[-1].value , [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)   ##B,N,K,4
        xyz1_21 = helper_tf_util.conv2d(xyz_2, xyz_2.get_shape()[-1].value , [1, 1], name + 'mlp3', [1, 1], 'VALID', True, is_training)   ##B,N,K,4
        xyz1 = tf.concat([xyz1_11, xyz1_21], axis = -1)  #### B,N,K,8
        xyz11 = tf.reduce_mean(xyz1, axis=2, keepdims=True) ##B,N,1,8
        xyz11_1 = tf.tile( xyz11, [1, 1, self.config.k_n, 1])  ##B,N,k,8
        xyz12 = tf.reduce_max(xyz1, axis=2, keepdims=True)  ##B,N,1,8
        xyz12_2 = tf.tile( xyz12, [1, 1, self.config.k_n, 1])  ##B,N,k,8
        xyz1_111 = helper_tf_util.conv2d(xyz_1, xyz12.get_shape()[-1].value , [1, 1], name + 'mlp4', [1, 1], 'VALID', True, is_training)   ##B,N,K,8
        # xyz111 = tf.concat([xyz11_1, xyz12_2, xyz1_111], axis = -1) ##B,N,K,24


        ##：
        xyz2 = tf.concat([xyz_1, xyz_2], axis =-1)  ##B,N,K,8
        xyz2 = helper_tf_util.conv2d(xyz2, xyz2.get_shape()[-1].value , [1, 1], name + 'mlp5', [1, 1], 'VALID', True, is_training)   ##B,N,K,8
        xyz21 = tf.reduce_mean(xyz2, axis=2, keepdims=True)  ##B,N,1,8
        xyz21_1 = tf.tile(xyz21, [1, 1, self.config.k_n, 1])  ##B,N,k,8
        xyz22 = tf.reduce_max(xyz2, axis=2, keepdims=True)  ##B,N,1,8
        xyz22_2 = tf.tile(xyz22, [1, 1, self.config.k_n, 1])  ##B,N,k,8
        xyz1_222 = helper_tf_util.conv2d(xyz_2, xyz2.get_shape()[-1].value, [1, 1], name + 'mlp6', [1, 1], 'VALID',
                                         True, is_training)  ##B,N,K,8
        # xyz222 = tf.concat([xyz21_1, xyz22_2, xyz1_222], axis=-1)  ##B,N,K,24

        ##
        xyz333 = helper_tf_util.conv2d(tile_xyz, xyz2.get_shape()[-1].value, [1, 1], name + 'mlp7', [1, 1], 'VALID',
                                         True, is_training)  ##： B,N,K,8
        # xyz333 = helper_tf_util.conv2d(tile_xyz, 2 * xyz222.get_shape()[-1].value, [1, 1], name + 'mlp7', [1, 1], 'VALID',
        #                                True, is_training)  ##：B,N,K,48
        # xyz_o = tf.concat([xyz111,xyz222,xyz333], axis =-1)  ## B,N,K,56   ## B,N,K,96
        xyz_o = tf.concat([xyz11_1, xyz12_2, xyz1_111,xyz21_1, xyz22_2, xyz1_222,xyz333],axis=-1)

        xyz_g = helper_tf_util.conv2d(xyz_o, xyz_o.get_shape()[-1].value, [1, 1], name + 'mlp8', [1, 1], 'VALID',
                                         True, is_training)  ##B,N,K,56

        feat_info = tf.concat([neigh_feat - tile_feat, tile_feat], axis=-1)  ## B,N,K,d_out

        xyz_f = helper_tf_util.conv2d(feat_info, feat_info.get_shape()[-1].value, [1, 1], name + 'mlp9', [1, 1], 'VALID',
                                         True, is_training) ## B,N,K,d_out
        mxi1 = tf.concat([xyz_g,xyz_f],axis=-1)
        mxi2 = helper_tf_util.conv2d(mxi1, 3*d_out//2, [1, 1], name + 'mlp10', [1, 1], 'VALID',
                                         True, is_training)
        T1,T2,T3 = tf.split(mxi2,num_or_size_splits=3,axis=-1)


        ###
        T11 =  helper_tf_util.conv2d(T1, d_out//2,  [1, 1], name + 'mlp11', [1, 1], 'VALID',
                                         False, is_training,activation_fn=None)
        S_T1 = tf.sigmoid(T11)
        T11 = tf.multiply(T1,S_T1) + T1
        T1 = tf.reduce_max(T11,axis=2,keepdims=True)

        ##Local features
        ##1
        T2_1_1 = tf.transpose(T2,perm= [0, 1, 3, 2])  ##
        # print(" T2_1_1", T2_1_1.get_shape().as_list())
        T3_1_1 = tf.transpose(T3, perm=[0, 1, 3, 2])

        T2_1 = helper_tf_util.conv2d(T2_1_1, self.config.k_n, [1, 1], name + 'mlp12', [1, 1], 'VALID',
                                         False, is_training, activation_fn = None)
        T3_1 = helper_tf_util.conv2d(T3_1_1, self.config.k_n, [1, 1], name + 'mlp13', [1, 1], 'VALID',
                                   False, is_training, activation_fn = None)
        T2_11 = tf.transpose(T2_1, perm=[0, 1, 3, 2])
        # print(" T2_11", T2_11.get_shape().as_list())
        T3_11 = tf.transpose(T3_1, perm=[0, 1, 3, 2])

        T2_s1 = tf.nn.softmax(T2_11, axis=2)
        T3_s1 = tf.nn.softmax(T3_11, axis=2)
        T21 = tf.multiply(T2,T3_s1) + T2
        T31 = tf.multiply(T3,T2_s1) + T3
        T21 = helper_tf_util.conv2d(T21, T21.get_shape()[-1].value, [1, 1], name + 'mlp14', [1, 1], 'VALID',
                                   False, is_training, activation_fn = None)
        T31 = helper_tf_util.conv2d(T31, T31.get_shape()[-1].value, [1, 1], name + 'mlp15', [1, 1], 'VALID',
                                   False, is_training, activation_fn = None)

        T211 = tf.reduce_sum(T21, axis=2, keepdims = True)
        T311 = tf.reduce_sum(T31, axis=2, keepdims = True)

        T2 = helper_tf_util.conv2d(T211, T211.get_shape()[-1].value, [1, 1], name + 'mlp16', [1, 1], 'VALID',
                                   True, is_training)
        T3 = helper_tf_util.conv2d(T311, T311.get_shape()[-1].value, [1, 1], name + 'mlp17', [1, 1], 'VALID',
                                   True, is_training)

        G  = tf.concat([T1,T2,T3], axis=-1)
        overall_encoding = helper_tf_util.conv2d(G ,d_out, [1, 1], name + 'mlp18', [1, 1], 'VALID', True,
                                                 is_training)  # B, N, 1, d_out  ## 将总特征mlp
        output_feat = helper_tf_util.conv2d(overall_encoding, d_out * 2, [1, 1], name + 'mlp19', [1, 1], 'VALID',  True, is_training, activation_fn=tf.nn.leaky_relu) # B, N, 1, 2*d_out ##总特征拉伸到d_out*2
        return output_feat
    def feature1(self,feature1, feature2,idx,name,is_training):
        # feature1: B,N1,1,C1 Peer-level
        # feature2: B,N2,1,C2  Points requiring upsampling
        f_interp_i = self.nearest_interpolation(feature2, idx)
        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([feature1, f_interp_i], axis=3),
                                                      feature1.get_shape()[-1].value, [1, 1],
                                                      name + 'f_decoder_i', [1, 1], 'VALID',
                                                      bn=True,
                                                      is_training=is_training)
        return f_decoder_i

    def feature2(self,feature1, feature2, idx1, name, is_training):
        # feature1: B,N1,1,C1 Peer-level
        # feature2: B,N2,1,C2 Points requiring upsampling
        # feature3: B,N2,1,C2 Points requiring downsampling
        # idx1: Upsampling index
        # idx1: Downsampling index
        f_interp_i = self.nearest_interpolation(feature2, idx1)
        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([feature1, f_interp_i], axis=3),
                                                      feature1.get_shape()[-1].value, [1, 1],
                                                      name + 'f_decoder_i', [1, 1], 'VALID',
                                                      bn=True,
                                                      is_training=is_training)

        return f_decoder_i

    def down_sample_fusion(self, feature,feature_down_list, idx_down_list, feature_up_list,idx_up_list, name, is_training):
        ##idx_down_list :Sorted in descending order, from low-level features to high-level features.
        ##idx_up_list: Sorted in ascending order, from high-level features to low-level features
        sampled_feature = []
        for i in range(len(feature_down_list) - 1):
            feature1 = feature_down_list[i]
            sampled_feature1 = []
            for j in range(len(feature_down_list) - 1 - i):
                feature_i = self.random_sample(feature1, idx_down_list[j + i])
                sampled_feature1.append(feature_i)
                feature1 = feature_i
            sampled_feature.append(sampled_feature1[-1])
        sampled_feature.append(feature_down_list[-1])

        uped_feature = []
        for i in range(len(feature_up_list)):
            feature2 = feature_up_list[i]
            uped_feature1 = []
            for j in range(len(feature_up_list) - i + 1):
                feature_i2 = self.nearest_interpolation(feature2, idx_up_list[j+i])
                uped_feature1.append(feature_i2)
                feature2 = feature_i2
            uped_feature.append(uped_feature1[-1])
        feature1_list = uped_feature + sampled_feature

        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat(feature1_list, axis=3),
                                                      feature.get_shape()[-1].value, [1, 1],
                                                      name + 'feature3i', [1, 1], 'VALID',
                                                      bn=True,
                                                      is_training=is_training)
        return f_decoder_i

    def down_sample_fusion0(self, feature,feature_down_list, idx_down_list, name, is_training):
        ##idx_list :Sorted in descending order, from low-level features to high-level features
        sampled_feature = []
        for i in range(len(feature_down_list)-1):
            feature1 = feature_down_list[i]
            sampled_feature1 = []
            for j in range(len(feature_down_list)-1 - i):
                feature_i = self.random_sample(feature1, idx_down_list[j+i])
                sampled_feature1.append(feature_i)
                feature1 = feature_i
            sampled_feature.append(sampled_feature1[-1])
        sampled_feature.append(feature_down_list[-1])
        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat(sampled_feature, axis=3),
                                                      feature.get_shape()[-1].value, [1, 1],
                                                      name + 'feature3', [1, 1], 'VALID',
                                                      bn=True,
                                                      is_training=is_training)
        return f_decoder_i

    def down_sample_fusion1(self, feature,feature_down_list, feature_up_list,idx_up_list, name, is_training):

        ##idx_up_list :Sorted in ascending order, from high-level features to low-level features.
        uped_feature = []
        for i in range(len(feature_up_list)):
            feature2 = feature_up_list[i]
            uped_feature1 = []
            for j in range(len(feature_up_list) - i + 1):
                feature_i2 = self.nearest_interpolation(feature2, idx_up_list[j+i])
                uped_feature1.append(feature_i2)
                feature2 = feature_i2
            uped_feature.append(uped_feature1[-1])
        uped_feature.append(feature_down_list[-1])

        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat(uped_feature, axis=3),
                                                      feature.get_shape()[-1].value, [1, 1],
                                                      name + 'feature3i', [1, 1], 'VALID',
                                                      bn=True,
                                                      is_training=is_training)
        return f_decoder_i
    def feature3(self,feature1, feature2, feature3, idx1, idx2, name, is_training):
        f_interp_i2 = self.nearest_interpolation(feature2, idx1)
        f_interp_i3 = self.nearest_interpolation(feature3, idx2)
        f_interp_i3 = self.nearest_interpolation(f_interp_i3, idx1)
        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([feature1, f_interp_i2, f_interp_i3], axis=3),
                                                      feature1.get_shape()[-1].value, [1, 1],
                                                      name + 'feature3', [1, 1], 'VALID',
                                                      bn=True,
                                                      is_training=is_training)
        return f_decoder_i

    def feature4(self, feature1, feature2, feature3, feature4, idx1, idx2, idx3,  name, is_training):
        f_interp_i2 = self.nearest_interpolation(feature2, idx1)
        f_interp_i3 = self.nearest_interpolation(feature3, idx2)
        f_interp_i3 = self.nearest_interpolation(f_interp_i3, idx1)
        f_down_i = self.random_sample(feature4, idx3)
        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([feature1, f_interp_i2, f_interp_i3, f_down_i], axis=3),
                                                      feature1.get_shape()[-1].value, [1, 1],
                                                      name + 'feature3', [1, 1], 'VALID',
                                                      bn=True,
                                                      is_training=is_training)
        return f_decoder_i
    def downsample(self,feature,idx_list,i):
        feature1 = feature
        for j in range(i):
            feature_i = self.random_sample(feature1, idx_list[j])
            feature1 = feature_i
        return feature1
    def nearest_interpolation1(self,feature,idx_list,i):
        feature1 = feature
        for j in range(i):
            feature_i = self.nearest_interpolation(feature1, idx_list[j])
            feature1 = feature_i
        return feature1

    def valid_logits_label(self,logits,labels,num_class,ignored_label_inds):
        logits = tf.reshape(logits, [-1, num_class])
        # Boolean mask of points that should be ignored
        ignored_bool = tf.zeros_like(labels, dtype=tf.bool)
        for ign_label in ignored_label_inds:
            ignored_bool = tf.logical_or(ignored_bool, tf.equal(labels, ign_label))

        # Collect logits and labels that are not ignored
        valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        valid_logits = tf.gather(logits, valid_idx, axis=0)
        valid_labels_init = tf.gather(labels, valid_idx, axis=0)

        # Reduce label values in the range of logit shape
        reducing_list = tf.range(num_class, dtype=tf.int32)
        inserted_value = tf.zeros((1,), dtype=tf.int32)
        for ign_label in ignored_label_inds:
            reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = tf.gather(reducing_list, valid_labels_init)
        return valid_logits,valid_labels

    def valid_logits(self,logits,label,num_class,ignored_label_inds):
        logits = tf.reshape(logits, [-1, num_class])
        labels = tf.reshape(label, [-1])
        # Boolean mask of points that should be ignored
        ignored_bool = tf.zeros_like(labels, dtype=tf.bool)
        for ign_label in ignored_label_inds:
            ignored_bool = tf.logical_or(ignored_bool, tf.equal(labels, ign_label))
        # Collect logits and labels that are not ignored
        valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        valid_logits = tf.gather(logits, valid_idx, axis=0)
        return valid_logits


    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        ##pc : [B,N,C]
        ## neighbor_idx: [B,N,K]
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])  ##[B,N*K]
        features = tf.batch_gather(pc, index_input)    ##
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])   ## [B,N,K,d]
        return features

    def label_consistency_loss(self, y_pred, alpha=0.5):
        #Calculate label consistency loss
        n_tasks = len(y_pred)
        l_consistency = 0.0
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                l_consistency += tf.reduce_mean(tf.abs(tf.nn.softmax(y_pred[i]) - tf.nn.softmax(y_pred[j])))
        l_consistency /= n_tasks * (n_tasks - 1) / 2
        loss = alpha * l_consistency
        return loss