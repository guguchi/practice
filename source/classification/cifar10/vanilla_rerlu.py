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
tf.app.flags.DEFINE_float('learning_rate', 0.0005, "学習率")
tf.app.flags.DEFINE_integer('iteration', 3, "学習反復回数")
tf.app.flags.DEFINE_integer('step', 500000, "学習数")
tf.app.flags.DEFINE_integer('batch_size', 50, "バッチサイズ")
tf.app.flags.DEFINE_integer('test_batch_size', 1000, "テストバッチサイズ")
tf.app.flags.DEFINE_float('gpu_memory', 0.1, "gpuメモリ使用割合")
tf.app.flags.DEFINE_integer('test_example', 10000, "テストデータ数")
tf.app.flags.DEFINE_string('cifar_data_dir', './../../../data/', "cifar10保存先")
tf.app.flags.DEFINE_string('save_data_path', './../../../data/classification/cifar10/vanilla_batch/', "データ保存先")


def deepnn(x):
    # 1 layer
    W_1 = weight_variable([32*32, 32*32])
    b_1 = bias_variable([32*32])
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
    jacobian = W_1

    # 2 layer
    W_2 = weight_variable([32*32, 32*32])
    b_2 = bias_variable([32*32])
    h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)
    jacobian = tf.matmul(W_2, jacobian)

    # 3 layer
    W_3 = weight_variable([32*32, 32*32])
    b_3 = bias_variable([32*32])
    h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)
    jacobian = tf.matmul(W_3, jacobian)

    # 4 layer
    W_4 = weight_variable([32*32, 32*32])
    b_4 = bias_variable([32*32])
    h_4 = tf.nn.relu(tf.matmul(h_3, W_4) + b_4)
    jacobian = tf.matmul(W_4, jacobian)

    # 5 layer
    W_5 = weight_variable([32*32, 32*32])
    b_5 = bias_variable([32*32])
    h_5 = tf.nn.relu(tf.matmul(h_4, W_5) + b_5)
    jacobian = tf.matmul(W_5, jacobian)

    # 6 layer
    W_6 = weight_variable([32*32, 32*32])
    b_6 = bias_variable([32*32])
    h_6 = tf.nn.relu(tf.matmul(h_5, W_6) + b_6)
    jacobian = tf.matmul(W_6, jacobian)

    # 7 layer
    W_7 = weight_variable([32*32, 32*32])
    b_7 = bias_variable([32*32])
    h_7 = tf.nn.relu(tf.matmul(h_6, W_7) + b_7)
    jacobian = tf.matmul(W_7, jacobian)

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


def svd(A, full_matrices=False, compute_uv=True, name=None):
    # since dA = dUSVt + UdSVt + USdVt
    # we can simply recompute each matrix using A = USVt
    # while blocking gradients to the original op.
    M, N = A.get_shape().as_list()
    P = min(M, N)
    S0, U0, V0 = map(tf.stop_gradient, tf.svd(A, full_matrices=True, name=name))
    #Ui, Vti = map(tf.matrix_inverse, [U0, tf.transpose(V0, (0, 2, 1))])
    Ui = tf.transpose(U0)
    Vti = V0
    # A = USVt
    # S = UiAVti
    S = tf.matmul(Ui, tf.matmul(A, Vti))
    S = tf.matrix_diag_part(S)
    return S


def compute_entropy_with_svd(jacobian):
    with tf.device('/cpu:0'):
        s = svd(jacobian, compute_uv=False)
    #if self.layer_method == "each":
    s = tf.maximum(s, 0.1 ** 8)
    log_determine = tf.log(tf.abs(s))
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

    # model
    y_out, entropy_all, h_7 = deepnn(x)

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

    save_path = FLAGS.save_data_path + 'batch_{}_alpha_{}/'.format(FLAGS.batch_size, FLAGS.learning_rate)
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

                _train_images_batch, _trains_labels_batch = sess_train_example_batch.run([train_images_batch, trains_labels_batch])

                _, cross_entropy_curr, train_accuracy_batch = sess.run(
                    [train_step, cross_entropy, accuracy],
                    feed_dict={x: _train_images_batch, y_: _trains_labels_batch})

                train_accuracy_list_batch[_iter][i] = train_accuracy_batch
                cross_entropy_list[_iter][i] = cross_entropy_curr

                if i % 5000 == 0 or i == FLAGS.step - 1:
                    """
                    true_train_accuracy = 0.0
                    _step = 0
                    while _step < num_iter:
                        _train_images, _trains_labels = sess_train_example.run([train_images, train_labels])

                        train_accuracy = accuracy.eval(feed_dict={
                            x: _train_images, y_: _trains_labels})
                        true_train_accuracy += train_accuracy
                        _step += 1
                    true_train_accuracy = true_train_accuracy / num_iter
                    print('step %d, test accuracy %g' % (i, true_train_accuracy))
                    train_accuracy_list[_iter][i] = true_train_accuracy
                    """
                    true_test_accuracy = 0.0
                    _step = 0
                    while _step < num_iter:
                        _test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                        test_accuracy = accuracy.eval(feed_dict={
                            x: _test_images, y_: _test_labels})
                        true_test_accuracy += test_accuracy
                        _step += 1
                    true_test_accuracy = true_test_accuracy / num_iter
                    test_accuracy_list[_iter][i] = true_test_accuracy

                    _test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                    entropy_curr, layer_samples = entropy_all.eval( feed_dict={
                        x: _test_images, y_: _test_labels})

                    entropy_list[_iter][i] = entropy_curr

                    print('step %d, test accuracy %g, entropy %g' % (i, true_test_accuracy, entropy_curr))
                    print '---------------------'

                if _iter == 0 and (i % 5000 == 0 or i == FLAGS.step - 1):
                    _test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                    layer_samples = sess.run([h_5],
                                  feed_dict={x: _test_images[:18]})
                    fig = plot(_test_images[:18], layer_samples[0])
                    plt.savefig(save_path+'layer_samples_{}.png'.format(i))
                    plt.close()


    #np.save(save_path+'train_accuracy.npy', train_accuracy_list)
    np.save(save_path+'train_accuracy_batch.npy', train_accuracy_list_batch)
    np.save(save_path+'test_accuracy.npy', test_accuracy_list)
    np.save(save_path+'cross_entropy.npy', cross_entropy_list)
    np.save(save_path+'entropy.npy', entropy_list)


if __name__ == '__main__':

    tf.app.run(main=main)
