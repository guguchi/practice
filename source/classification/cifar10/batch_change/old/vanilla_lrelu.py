#coding:utf-8
import argparse
import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
from cifar10_data import *
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.0005, "学習率")
tf.app.flags.DEFINE_integer('iteration', 3, "学習反復回数")
tf.app.flags.DEFINE_integer('step', 1000000, "学習数")
tf.app.flags.DEFINE_integer('batch_size', 25, "バッチサイズ")
tf.app.flags.DEFINE_integer('layer_size', 5, "レイヤー数")
tf.app.flags.DEFINE_integer('test_batch_size', 25, "テストバッチサイズ")
tf.app.flags.DEFINE_float('gpu_memory', 0.1, "gpuメモリ使用割合")
tf.app.flags.DEFINE_integer('test_example', 10000, "テストデータ数")
tf.app.flags.DEFINE_string('cifar_data_dir', '/home/ishii/Desktop/research/practice/data/', "cifar10保存先")
tf.app.flags.DEFINE_string('save_data_path', '/home/ishii/Desktop/research/practice/data/classification/cifar10/layer_change/', "データ保存先")



def deepnn(x, phase_train):
    # 1 layer
    W_1 = weight_variable([32*32, 32*32])
    b_1 = bias_variable([32*32])
    h_1 = lrelu(tf.matmul(x, W_1) + b_1)

    if phase_train == True:
        h_j_1 = tf.Variable(tf.ones([FLAGS.batch_size*32*32]))
        h_1_index = tf.where(tf.reshape(h_1, [-1]) < 0.0)
        h_j_1 = tf.scatter_update(h_j_1, h_1_index, -0.2 * tf.ones_like(h_1_index, tf.float32))
        jacobian = tf.reshape(h_j_1, [FLAGS.batch_size, 1, 32*32]) * W_1
    else:
        h_j_1 = tf.Variable(tf.ones([FLAGS.test_batch_size*32*32]))
        h_1_index = tf.where(tf.reshape(h_1, [-1]) < 0.0)
        h_j_1 = tf.scatter_update(h_j_1, h_1_index, -0.2 * tf.ones_like(h_1_index, tf.float32))
        jacobian = tf.reshape(h_j_1, [FLAGS.test_batch_size, 1, 32*32]) * W_1

    if FLAGS.layer_size == 1:
        # output
        W_out = weight_variable([32*32, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_1, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_1

    # 2 layer
    W_2 = weight_variable([32*32, 32*32])
    b_2 = bias_variable([32*32])
    h_2 = lrelu(tf.matmul(h_1, W_2) + b_2)

    if phase_train == True:
        h_j_2 = tf.Variable(tf.ones([FLAGS.batch_size*32*32]))
        h_2_index = tf.where(tf.reshape(h_2, [-1]) < 0.0)
        h_j_2 = tf.scatter_update(h_j_2, h_2_index, -0.2 * tf.ones_like(h_2_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_2, [FLAGS.batch_size, 1, 32*32]) * W_2, jacobian)
    else:
        h_j_2 = tf.Variable(tf.ones([FLAGS.test_batch_size*32*32]))
        h_2_index = tf.where(tf.reshape(h_2, [-1]) < 0.0)
        h_j_2 = tf.scatter_update(h_j_2, h_2_index, -0.2 * tf.ones_like(h_2_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_2, [FLAGS.test_batch_size, 1, 32*32]) * W_2, jacobian)

    if FLAGS.layer_size == 2:
        # output
        W_out = weight_variable([32*32, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_2, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_2

    # 3 layer
    W_3 = weight_variable([32*32, 32*32])
    b_3 = bias_variable([32*32])
    h_3 = lrelu(tf.matmul(h_2, W_3) + b_3)

    if phase_train == True:
        h_j_3 = tf.Variable(tf.ones([FLAGS.batch_size*32*32]))
        h_3_index = tf.where(tf.reshape(h_3, [-1]) < 0.0)
        h_j_3 = tf.scatter_update(h_j_3, h_3_index, -0.2 * tf.ones_like(h_3_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_3, [FLAGS.batch_size, 1, 32*32]) * W_3, jacobian)
    else:
        h_j_3 = tf.Variable(tf.ones([FLAGS.test_batch_size*32*32]))
        h_3_index = tf.where(tf.reshape(h_3, [-1]) < 0.0)
        h_j_3 = tf.scatter_update(h_j_3, h_3_index, -0.2 * tf.ones_like(h_3_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_3, [FLAGS.test_batch_size, 1, 32*32]) * W_3, jacobian)

    if FLAGS.layer_size == 3:
        # output
        W_out = weight_variable([32*32, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_3, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_3

    # 4 layer
    W_4 = weight_variable([32*32, 32*32])
    b_4 = bias_variable([32*32])
    h_4 = lrelu(tf.matmul(h_3, W_4) + b_4)

    if phase_train == True:
        h_j_4 = tf.Variable(tf.ones([FLAGS.batch_size*32*32]))
        h_4_index = tf.where(tf.reshape(h_4, [-1]) < 0.0)
        h_j_4 = tf.scatter_update(h_j_4, h_4_index, -0.2 * tf.ones_like(h_4_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_4, [FLAGS.batch_size, 1, 32*32]) * W_4, jacobian)
    else:
        h_j_4 = tf.Variable(tf.ones([FLAGS.test_batch_size*32*32]))
        h_4_index = tf.where(tf.reshape(h_4, [-1]) < 0.0)
        h_j_4 = tf.scatter_update(h_j_4, h_4_index, -0.2 * tf.ones_like(h_4_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_4, [FLAGS.test_batch_size, 1, 32*32]) * W_4, jacobian)

    if FLAGS.layer_size == 4:
        # output
        W_out = weight_variable([32*32, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_4, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_4

    # 5 layer
    W_5 = weight_variable([32*32, 32*32])
    b_5 = bias_variable([32*32])
    h_5 = lrelu(tf.matmul(h_4, W_5) + b_5)

    if phase_train == True:
        h_j_5 = tf.Variable(tf.ones([FLAGS.batch_size*32*32]))
        h_5_index = tf.where(tf.reshape(h_5, [-1]) < 0.0)
        h_j_5 = tf.scatter_update(h_j_5, h_5_index, -0.2 * tf.ones_like(h_5_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_5, [FLAGS.batch_size, 1, 32*32]) * W_5, jacobian)
    else:
        h_j_5 = tf.Variable(tf.ones([FLAGS.test_batch_size*32*32]))
        h_5_index = tf.where(tf.reshape(h_5, [-1]) < 0.0)
        h_j_5 = tf.scatter_update(h_j_5, h_5_index, -0.2 * tf.ones_like(h_5_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_5, [FLAGS.test_batch_size, 1, 32*32]) * W_5, jacobian)

    if FLAGS.layer_size == 5:
        # output
        W_out = weight_variable([32*32, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_5, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_5

    # 6 layer
    W_6 = weight_variable([32*32, 32*32])
    b_6 = bias_variable([32*32])
    h_6 = lrelu(tf.matmul(h_5, W_6) + b_6)

    if phase_train == True:
        h_j_6 = tf.Variable(tf.ones([FLAGS.batch_size*32*32]))
        h_6_index = tf.where(tf.reshape(h_6, [-1]) < 0.0)
        h_j_6 = tf.scatter_update(h_j_6, h_6_index, -0.2 * tf.ones_like(h_6_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_6, [FLAGS.batch_size, 1, 32*32]) * W_6, jacobian)
    else:
        h_j_6 = tf.Variable(tf.ones([FLAGS.test_batch_size*32*32]))
        h_6_index = tf.where(tf.reshape(h_6, [-1]) < 0.0)
        h_j_6 = tf.scatter_update(h_j_6, h_6_index, -0.2 * tf.ones_like(h_6_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_6, [FLAGS.test_batch_size, 1, 32*32]) * W_6, jacobian)

    if FLAGS.layer_size == 6:
        # output
        W_out = weight_variable([32*32, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_6, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_6

    # 7 layer
    W_7 = weight_variable([32*32, 32*32])
    b_7 = bias_variable([32*32])
    h_7 = lrelu(tf.matmul(h_6, W_7) + b_7)

    if phase_train == True:
        h_j_7 = tf.Variable(tf.ones([FLAGS.batch_size*32*32]))
        h_7_index = tf.where(tf.reshape(h_7, [-1]) < 0.0)
        h_j_7 = tf.scatter_update(h_j_7, h_7_index, -0.2 * tf.ones_like(h_7_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_7, [FLAGS.batch_size, 1, 32*32]) * W_7, jacobian)
    else:
        h_j_7 = tf.Variable(tf.ones([FLAGS.test_batch_size*32*32]))
        h_7_index = tf.where(tf.reshape(h_7, [-1]) < 0.0)
        h_j_7 = tf.scatter_update(h_j_7, h_7_index, -0.2 * tf.ones_like(h_7_index, tf.float32))
        jacobian = tf.matmul(tf.reshape(h_j_7, [FLAGS.test_batch_size, 1, 32*32]) * W_7, jacobian)

    # output
    W_out = weight_variable([32*32, 10])
    b_out = bias_variable([10])

    y_out = tf.matmul(h_7, W_out) + b_out
    entropy_all = compute_entropy_with_svd(jacobian)
    return y_out, entropy_all, h_7


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def svd(A, full_matrices=False, compute_uv=True, name=None):
    _, M, N = A.get_shape().as_list()
    P = min(M, N)
    S0, U0, V0 = map(tf.stop_gradient, tf.svd(A, full_matrices=True, name=name))
    Ui = tf.transpose(U0, (0, 2, 1))
    Vti = V0
    S = tf.matmul(Ui, tf.matmul(A, Vti))
    S = tf.matrix_diag_part(S)
    return S


def compute_entropy_with_svd(jacobian):
    with tf.device('/cpu:0'):
        s = svd(jacobian, compute_uv=False)
    #if self.layer_method == "each":
    s_index = len(np.where(s == 0.0)[0])
    s = tf.maximum(tf.abs(s), 0.1 ** 8)
    log_determine = tf.log(s) + s_index * 8.0 * tf.log(10.0)
    entropy = tf.reduce_mean(log_determine)
    return entropy


def plot(samples, layer_samples):
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(6, 6)
    gs.update(wspace=0.05, hspace=0.05)

    for (i, sample) in enumerate(samples):
        ax = plt.subplot(gs[2*i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(32, 32), cmap='Greys_r')

        ax = plt.subplot(gs[2*i + 1])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(layer_samples[i].reshape(32, 32), cmap='Greys_r')

    return fig


def main(argv):
    # data preparation
    maybe_download_and_extract(FLAGS.cifar_data_dir)
    num_iter = 10.0#int(math.ceil(FLAGS.test_example / FLAGS.test_batch_size))

    # data load
    train_images_batch, trains_labels_batch = train_input(FLAGS.cifar_data_dir + 'cifar-10-batches-bin/',
                                              FLAGS.batch_size)
    #train_images, train_labels = train_input(FLAGS.cifar_data_dir + 'cifar-10-batches-bin/',
    #                                      FLAGS.test_batch_size)
    test_images, test_labels = test_input(FLAGS.cifar_data_dir + 'cifar-10-batches-bin/',
                                          FLAGS.test_batch_size)

    # placeholder
    x = tf.placeholder(tf.float32, [None, 32*32])
    y_ = tf.placeholder(tf.float32, [None, 10])
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # model
    y_out, entropy_all, h_last = deepnn(x, phase_train)

    # objective
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    #GradientDescentOptimizer / AdamOptimizer / MomentumOptimizer

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # save data
    train_accuracy_list_batch = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    #train_accuracy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    test_accuracy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    cross_entropy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    entropy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    entropy_train_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)

    save_path = FLAGS.save_data_path + 'lrelu_layer_{}_batch_{}_alpha_{}/'.format(
        FLAGS.layer_size, FLAGS.batch_size, FLAGS.learning_rate)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = tf.ConfigProto(
             gpu_options=tf.GPUOptions(
             per_process_gpu_memory_fraction=FLAGS.gpu_memory # 最大値の50%まで
             )
    )
    config_cpu = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    sess_train_example_batch = tf.Session(config=config_cpu)
    tf.train.start_queue_runners(sess=sess_train_example_batch)
    #sess_train_example = tf.Session(config=config_cpu)
    #tf.train.start_queue_runners(sess=sess_train_example)
    sess_test_example = tf.Session(config=config_cpu)
    tf.train.start_queue_runners(sess=sess_test_example)

    for _iter in range(FLAGS.iteration):
        print '=== iteration {} ==='.format(_iter)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(FLAGS.step):

                _train_images_batch, _trains_labels_batch = sess_train_example_batch.run([
                    train_images_batch, trains_labels_batch])

                _, cross_entropy_curr, train_accuracy_batch = sess.run(
                    [train_step, cross_entropy, accuracy],
                    feed_dict={x: _train_images_batch, y_: _trains_labels_batch, phase_train: True})

                train_accuracy_list_batch[_iter][i] = train_accuracy_batch
                cross_entropy_list[_iter][i] = cross_entropy_curr

                if i % 5000 == 0 or i == FLAGS.step - 1:
                    train_entropy = entropy_all.eval(feed_dict={
                        x: _train_images_batch, phase_train: False})
                    entropy_train_list[_iter][i] = train_entropy

                    true_test_accuracy = 0.0
                    _step = 0
                    while _step < num_iter:
                        _test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                        test_accuracy = accuracy.eval(feed_dict={
                            x: _test_images, y_: _test_labels, phase_train: False})
                        true_test_accuracy += test_accuracy
                        _step += 1
                    true_test_accuracy = true_test_accuracy / num_iter
                    test_accuracy_list[_iter][i] = true_test_accuracy

                    #_test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                    entropy_curr = entropy_all.eval(feed_dict={
                        x: _test_images, phase_train: False})

                    entropy_list[_iter][i] = entropy_curr
                    #entropy_curr = 1.0
                    print('step %d, test accuracy %g, entropy %g, train entropy %g' % (i, test_accuracy, entropy_curr, train_entropy))
                    print '---------------------'

                if _iter == 0 and (i % 5000 == 0 or i == FLAGS.step - 1):
                    _test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                    layer_samples = sess.run([h_last],
                                  feed_dict={x: _test_images[:18], phase_train: False})
                    fig = plot(_test_images[:18], layer_samples[0])
                    plt.savefig(save_path+'layer_samples_{}.png'.format(i))
                    plt.close()

    #np.save(save_path+'train_accuracy.npy', train_accuracy_list)
    np.save(save_path+'train_accuracy_batch.npy', train_accuracy_list_batch)
    np.save(save_path+'test_accuracy.npy', test_accuracy_list)
    np.save(save_path+'cross_entropy.npy', cross_entropy_list)
    np.save(save_path+'entropy.npy', entropy_list)
    np.save(save_path+'entropy_train.npy', entropy_train_list)


if __name__ == '__main__':

    tf.app.run(main=main)
