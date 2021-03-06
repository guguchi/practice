#coding:utf-8
import argparse
import sys
import os
import math

import numpy as np
import tensorflow as tf
from cifar10_data import *
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lam', 0.01, "正則化")
tf.app.flags.DEFINE_float('l2_lam', 0.01, "L2正則化")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "学習率")
tf.app.flags.DEFINE_integer('iteration', 10, "学習反復回数")
tf.app.flags.DEFINE_integer('step', 50000, "学習数")
tf.app.flags.DEFINE_integer('batch_size', 50, "バッチサイズ")
tf.app.flags.DEFINE_integer('test_batch_size', 50, "テストバッチサイズ")
tf.app.flags.DEFINE_float('gpu_memory', 0.1, "gpuメモリ使用割合")
tf.app.flags.DEFINE_integer('test_example', 10000, "テストデータ数")
tf.app.flags.DEFINE_string('cifar_data_dir', './../../../data/', "cifar10保存先")
tf.app.flags.DEFINE_string('save_data_path', './../../../data/classification/cifar10/entropy_all_sigmoid_plus/0702/', "データ保存先")


def deepnn(x):
    # 1 layer
    W_1 = weight_variable([32*32, 32*32])
    b_1 = bias_variable([32*32])
    h_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
    jacobian = tf.reshape((1.0 - h_1) * h_1, [FLAGS.batch_size, 1, 32*32) * W_1

    # 2 layer
    W_2 = weight_variable([32*32, 32*32])
    b_2 = bias_variable([32*32])
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)
    jacobian = tf.matmul(tf.reshape((1.0 - h_2) * h_2, [FLAGS.batch_size, 1, 32*32) * W_2, jacobian)

    # 3 layer
    W_3 = weight_variable([32*32, 32*32])
    b_3 = bias_variable([32*32])
    h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)
    jacobian = tf.matmul(tf.reshape((1.0 - h_3) * h_3, [FLAGS.batch_size, 1, 32*32) * W_3, jacobian)

    # 4 layer
    W_4 = weight_variable([32*32, 32*32])
    b_4 = bias_variable([32*32])
    h_4 = tf.nn.sigmoid(tf.matmul(h_3, W_4) + b_4)
    jacobian = tf.matmul(tf.reshape((1.0 - h_4) * h_4, [FLAGS.batch_size, 1, 32*32) * W_4, jacobian)

    # 5 layer
    W_5 = weight_variable([32*32, 32*32])
    b_5 = bias_variable([32*32])
    h_5 = tf.nn.sigmoid(tf.matmul(h_4, W_5) + b_5)
    jacobian = tf.matmul(tf.reshape((1.0 - h_5) * h_5, [FLAGS.batch_size, 1, 32*32) * W_5, jacobian)

    # 6 layer
    W_6 = weight_variable([32*32, 32*32])
    b_6 = bias_variable([32*32])
    h_6 = tf.nn.sigmoid(tf.matmul(h_5, W_6) + b_6)
    jacobian = tf.matmul(tf.reshape((1.0 - h_6) * h_6, [FLAGS.batch_size, 1, 32*32) * W_6, jacobian)

    # 7 layer
    W_7 = weight_variable([32*32, 32*32])
    b_7 = bias_variable([32*32])
    h_7 = tf.nn.sigmoid(tf.matmul(h_6, W_7) + b_7)
    jacobian = tf.matmul(tf.reshape((1.0 - h_7) * h_7, [FLAGS.batch_size, 1, 32*32) * W_7, jacobian)

    # output
    W_out = weight_variable([32*32, 10])
    b_out = bias_variable([10])
    fro_norm_8 = fro_norm(W_out)

    y_out = tf.matmul(h_7, W_out) + b_out
    entropy_all = compute_entropy_with_svd(jacobian)
    return y_out, entropy_all, fro_norm_8


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def svd(A, full_matrices=False, compute_uv=True, name=None):
    M, N = A.get_shape().as_list()
    P = min(M, N)
    S0, U0, V0 = map(tf.stop_gradient, tf.svd(A, full_matrices=True, name=name))
    Ui = tf.transpose(U0)
    Vti = V0
    S = tf.matmul(Ui, tf.matmul(A, Vti))
    S = tf.matrix_diag_part(S)
    return S


def compute_entropy_with_svd(jacobian):
    with tf.device('/cpu:0'):
        s = svd(jacobian, compute_uv=False)
    s = tf.maximum(s, 0.1 ** 8)
    log_determine = tf.log(tf.abs(s))
    entropy = -tf.reduce_mean(log_determine)
    return entropy


def fro_norm(W):
    norm = tf.trace(tf.matmul(W, tf.transpose(W)))
    return norm


def main(argv):
    # data preparation
    maybe_download_and_extract(FLAGS.cifar_data_dir)
    num_iter = int(math.ceil(FLAGS.test_example / FLAGS.batch_size))

    # data load
    train_images, trains_labels = train_input(FLAGS.cifar_data_dir + 'cifar-10-batches-bin/',
                                              FLAGS.batch_size)
    test_images, test_labels = test_input(FLAGS.cifar_data_dir + 'cifar-10-batches-bin/',
                                          FLAGS.test_batch_size)

    # placeholder
    x = tf.placeholder(tf.float32, [None, 32*32])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # model
    y_out, entropy_all, fro_norm_8 = deepnn(x)

    # objective
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy - FLAGS.lam * entropy_all + FLAGS.l2_lam * fro_norm_8)
    #GradientDescentOptimizer / AdamOptimizer / MomentumOptimizer

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # save data
    train_accuracy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    test_accuracy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    cross_entropy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)

    save_path = FLAGS.save_data_path + 'lam_{}_l2_{}_batch_{}_alpha_{}/'.format(
        FLAGS.lam, FLAGS.l2_lam, FLAGS.batch_size, FLAGS.learning_rate)
    if not os.path.exists(FLAGS.save_data_path):
        os.makedirs(FLAGS.save_data_path)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory

    sess_train_example = tf.Session()
    tf.train.start_queue_runners(sess=sess_train_example)
    sess_test_example = tf.Session()
    tf.train.start_queue_runners(sess=sess_test_example)

    for _iter in range(FLAGS.iteration):
        print '=== iteration {} ==='.format(_iter)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(FLAGS.step):

                _train_images, _trains_labels = sess_train_example.run([train_images, trains_labels])

                _, cross_entropy_curr, train_accuracy = sess.run(
                    [train_step, cross_entropy, accuracy],
                    feed_dict={x: _train_images, y_: _trains_labels})

                train_accuracy_list[_iter][i] = train_accuracy
                cross_entropy_list[_iter][i] = cross_entropy_curr

                print 'step {}, train accuracy {}'.format(i, train_accuracy)

                if i % 500 == 0 or i == FLAGS.step - 1:
                    true_test_accuracy = 0.0
                    _step = 0
                    while _step < num_iter:
                        _test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                        test_accuracy = accuracy.eval(feed_dict={
                            x: _test_images, y_: _test_labels})
                        true_test_accuracy += test_accuracy
                        _step += 1
                    true_test_accuracy = true_test_accuracy / num_iter
                    print('step %d, test accuracy %g' % (i, true_test_accuracy))
                    test_accuracy_list[_iter][i] = true_test_accuracy


    np.save(FLAGS.save_data_path+'train_accuracy.npy', train_accuracy_list)
    np.save(FLAGS.save_data_path+'test_accuracy.npy', test_accuracy_list)
    np.save(FLAGS.save_data_path+'cross_entropy.npy', cross_entropy_list)


if __name__ == '__main__':

    tf.app.run(main=main)
