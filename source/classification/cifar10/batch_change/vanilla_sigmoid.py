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
tf.app.flags.DEFINE_integer('iteration', 1, "学習反復回数")
tf.app.flags.DEFINE_integer('step', 1000000, "学習数")
tf.app.flags.DEFINE_integer('batch_size', 25, "バッチサイズ")
tf.app.flags.DEFINE_integer('layer_size', 5, "レイヤー数")
tf.app.flags.DEFINE_integer('test_batch_size', 25, "テストバッチサイズ")
tf.app.flags.DEFINE_float('gpu_memory', 0.1, "gpuメモリ使用割合")
tf.app.flags.DEFINE_integer('test_example', 10000, "テストデータ数")
tf.app.flags.DEFINE_string('cifar_data_dir', '/home/ishii/Desktop/research/practice/data/', "cifar10保存先")
tf.app.flags.DEFINE_string('save_data_path', '/home/ishii/Desktop/research/practice/data/classification/cifar10/batch_change/', "データ保存先")


def deepnn(x, pred):
    # 1 layer
    with tf.variable_scope('classifier') as scope:
        W_1 = weight_variable([32*32, 32*32], 'W_1')
        b_1 = bias_variable([32*32], 'b_1')
    h_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
    jacobian = tf.reshape((1.0 - h_1) * h_1, [tf.shape(h_1)[0], 1, tf.shape(h_1)[1]]) * W_1

    if FLAGS.layer_size == 1:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([32*32, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_1, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_1, singular_value

    # 2 layer
    with tf.variable_scope('classifier') as scope:
        W_2 = weight_variable([32*32, 32*32], 'W_2')
        b_2 = bias_variable([32*32], 'b_2')
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)
    jacobian = tf.matmul(tf.reshape((1.0 - h_2) * h_2, [tf.shape(h_2)[0], 1, tf.shape(h_2)[1]]) * W_2, jacobian)

    if FLAGS.layer_size == 2:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([32*32, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_2, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_2, singular_value

    # 3 layer
    with tf.variable_scope('classifier') as scope:
        W_3 = weight_variable([32*32, 32*32], 'W_3')
        b_3 = bias_variable([32*32], 'b_3')
    h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)
    jacobian = tf.matmul(tf.reshape((1.0 - h_3) * h_3, [tf.shape(h_3)[0], 1, tf.shape(h_3)[1]]) * W_3, jacobian)

    if FLAGS.layer_size == 3:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([32*32, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_3, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_3, singular_value

    # 4 layer
    with tf.variable_scope('classifier') as scope:
        W_4 = weight_variable([32*32, 32*32], 'W_4')
        b_4 = bias_variable([32*32], 'b_4')
    h_4 = tf.nn.sigmoid(tf.matmul(h_3, W_4) + b_4)
    jacobian = tf.matmul(tf.reshape((1.0 - h_4) * h_4, [tf.shape(h_4)[0], 1, tf.shape(h_4)[1]]) * W_4, jacobian)

    if FLAGS.layer_size == 4:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([32*32, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_4, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_4, singular_value

    # 5 layer
    with tf.variable_scope('classifier') as scope:
        W_5 = weight_variable([32*32, 32*32], 'W_5')
        b_5 = bias_variable([32*32], 'b_5')
    h_5 = tf.nn.sigmoid(tf.matmul(h_4, W_5) + b_5)
    jacobian = tf.matmul(tf.reshape((1.0 - h_5) * h_5, [tf.shape(h_5)[0], 1, tf.shape(h_5)[1]]) * W_5, jacobian)

    if FLAGS.layer_size == 5:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([32*32, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_5, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_5, singular_value

    # 6 layer
    with tf.variable_scope('classifier') as scope:
        W_6 = weight_variable([32*32, 32*32], 'W_6')
        b_6 = bias_variable([32*32], 'b_6')
    h_6 = tf.nn.sigmoid(tf.matmul(h_5, W_6) + b_6)
    jacobian = tf.matmul(tf.reshape((1.0 - h_6) * h_6, [tf.shape(h_6)[0], 1, tf.shape(h_6)[1]]) * W_6, jacobian)

    if FLAGS.layer_size == 6:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([32*32, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_6, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_6, singular_value

    # 7 layer
    with tf.variable_scope('classifier') as scope:
        W_7 = weight_variable([32*32, 32*32], 'W_7')
        b_7 = bias_variable([32*32], 'b_7')
    h_7 = tf.nn.sigmoid(tf.matmul(h_6, W_7) + b_7)
    jacobian = tf.matmul(tf.reshape((1.0 - h_7) * h_7, [tf.shape(h_7)[0], 1, tf.shape(h_7)[1]]) * W_7, jacobian)

    # output
    with tf.variable_scope('classifier') as scope:
        W_out = weight_variable([32*32, 10], 'W_out')
        b_out = bias_variable([10], 'b_out')

    y_out = tf.matmul(h_7, W_out) + b_out
    singular_value = compute_svd(jacobian)
    return y_out, h_7, singular_value


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def svd(A, full_matrices=False, compute_uv=True, name=None):
    _, M, N = A.get_shape().as_list()
    P = min(M, N)
    S0, U0, V0 = map(tf.stop_gradient, tf.svd(A, full_matrices=True, name=name))
    Ui = tf.transpose(U0, (0, 2, 1))
    Vti = V0
    S = tf.matmul(Ui, tf.matmul(A, Vti))
    S = tf.matrix_diag_part(S)
    return S


def compute_svd(jacobian):
    with tf.device('/cpu:0'):
        s = svd(jacobian, compute_uv=False)
    return s


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
    test_images, test_labels = test_input(FLAGS.cifar_data_dir + 'cifar-10-batches-bin/',
                                          FLAGS.test_batch_size)

    # placeholder
    x = tf.placeholder(tf.float32, [None, 32*32])
    y_ = tf.placeholder(tf.float32, [None, 10])
    pred = tf.placeholder(tf.bool, name='pred')

    # model
    y_out, h_last, singular_value = deepnn(x, pred)

    # variable
    train_variables = tf.trainable_variables()
    theta = [v for v in train_variables if v.name.startswith("classifier")]

    # objective
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy, var_list=theta)
    #GradientDescentOptimizer / AdamOptimizer / MomentumOptimizer

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # save data
    train_accuracy_list_batch = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    test_accuracy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    cross_entropy_list = np.zeros((FLAGS.iteration, FLAGS.step), dtype=np.float32)
    with tf.device('/cpu:0'):
        train_singular_value_list = np.zeros((FLAGS.iteration, 250, FLAGS.batch_size, 32*32), dtype=np.float32)
        test_singular_value_list = np.zeros((FLAGS.iteration, 250, FLAGS.test_batch_size, 32*32), dtype=np.float32)

    save_path = FLAGS.save_data_path + 'sigmoid_layer_{}_batch_{}_alpha_{}/'.format(
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
    sess_test_example = tf.Session(config=config_cpu)
    tf.train.start_queue_runners(sess=sess_test_example)

    I = 0
    for _iter in range(FLAGS.iteration):
        print '=== iteration {} ==='.format(_iter)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={pred: False})

            for i in range(FLAGS.step):

                _train_images_batch, _trains_labels_batch = sess_train_example_batch.run([
                    train_images_batch, trains_labels_batch])

                _, cross_entropy_curr, train_accuracy_batch = sess.run(
                    [train_step, cross_entropy, accuracy],
                    feed_dict={x: _train_images_batch, y_: _trains_labels_batch, pred: True})

                train_accuracy_list_batch[_iter][i] = train_accuracy_batch
                cross_entropy_list[_iter][i] = cross_entropy_curr

                if i % 5000 == 0 or i == FLAGS.step - 1:
                    singular_value_curr = sess.run([singular_value], feed_dict={
                        x: _train_images_batch, pred: True})
                    #print np.shape(singular_value_curr)
                    train_singular_value_list[_iter][I] = singular_value_curr[0]

                    true_test_accuracy = 0.0
                    _step = 0
                    while _step < num_iter:
                        _test_images, _test_labels = sess_test_example.run([test_images, test_labels])

                        test_accuracy = accuracy.eval(feed_dict={
                            x: _test_images, y_: _test_labels, pred: False})
                        true_test_accuracy += test_accuracy
                        _step += 1
                    true_test_accuracy = true_test_accuracy / num_iter
                    test_accuracy_list[_iter][i] = test_accuracy

                    singular_value_curr = sess.run([singular_value], feed_dict={
                        x: _test_images, pred: False})
                    test_singular_value_list[_iter][I] = singular_value_curr[0]

                    print('step %d, test accuracy %g, ' % (i, test_accuracy))
                    print '---------------------'
                    I = I + 1
                """
                if _iter == 0 and (i % 5000 == 0 or i == FLAGS.step - 1):
                    samples = mnist.test.images[:18]
                    layer_samples = sess.run([h_last],
                                  feed_dict={x: samples, pred: False})
                    fig = plot(samples, layer_samples[0])
                    plt.savefig(save_path+'layer_samples_{}.png'.format(i))
                    plt.close()
                """


    #np.save(save_path+'train_accuracy.npy', train_accuracy_list)
    np.save(save_path+'train_accuracy_batch.npy', train_accuracy_list_batch)
    np.save(save_path+'test_accuracy.npy', test_accuracy_list)
    np.save(save_path+'cross_entropy.npy', cross_entropy_list)
    np.save(save_path+'train_singular_value.npy', train_singular_value_list)
    np.save(save_path+'test_singular_value.npy', test_singular_value_list)


if __name__ == '__main__':

    tf.app.run(main=main)
