#coding:utf-8
import argparse
import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.0005, "学習率")
tf.app.flags.DEFINE_integer('iteration', 5, "学習反復回数")
tf.app.flags.DEFINE_integer('step', 500000, "学習数")
tf.app.flags.DEFINE_integer('batch_size', 25, "バッチサイズ")
tf.app.flags.DEFINE_integer('layer_size', 5, "レイヤー数")
tf.app.flags.DEFINE_integer('entropy_num', 100, "entropy")
tf.app.flags.DEFINE_float('gpu_memory', 0.15, "gpuメモリ使用割合")
tf.app.flags.DEFINE_string('data_dir', './../../../../data/mnist/', "mnist保存先")
tf.app.flags.DEFINE_string('save_data_path', './../../../../data/classification/mnist/layer_cahnge/', "データ保存先")


def deepnn(x, phase_train):
    # 1 layer
    W_1 = weight_variable([28*28, 28*28])
    b_1 = bias_variable([28*28])
    h_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
    x, y = h_1.get_shape().to_list()
    jacobian = tf.reshape((1.0 - h_1) * h_1, [x, 1, y]) * W_1

    if FLAGS.layer_size == 1:
        # output
        W_out = weight_variable([28*28, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_1, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_1

    # 2 layer
    W_2 = weight_variable([28*28, 28*28])
    b_2 = bias_variable([28*28])
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)
    x, y = h_2.get_shape().to_list()
    jacobian = tf.matmul(tf.reshape((1.0 - h_2) * h_2, [x, 1, y]) * W_2, jacobian)

    if FLAGS.layer_size == 2:
        # output
        W_out = weight_variable([28*28, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_2, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_2

    # 3 layer
    W_3 = weight_variable([28*28, 28*28])
    b_3 = bias_variable([28*28])
    h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)
    x, y = h_3.get_shape().to_list()
    jacobian = tf.matmul(tf.reshape((1.0 - h_3) * h_3, [x, 1, y]) * W_3, jacobian)

    if FLAGS.layer_size == 3:
        # output
        W_out = weight_variable([28*28, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_3, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_3

    # 4 layer
    W_4 = weight_variable([28*28, 28*28])
    b_4 = bias_variable([28*28])
    h_4 = tf.nn.sigmoid(tf.matmul(h_3, W_4) + b_4)
    x, y = h_4.get_shape().to_list()
    jacobian = tf.matmul(tf.reshape((1.0 - h_4) * h_4, [x, 1, y]) * W_4, jacobian)

    if FLAGS.layer_size == 4:
        # output
        W_out = weight_variable([28*28, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_4, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_4

    # 5 layer
    W_5 = weight_variable([28*28, 28*28])
    b_5 = bias_variable([28*28])
    h_5 = tf.nn.sigmoid(tf.matmul(h_4, W_5) + b_5)
    x, y = h_5.get_shape().to_list()
    jacobian = tf.matmul(tf.reshape((1.0 - h_5) * h_5, [x, 1, y]) * W_5, jacobian)

    if FLAGS.layer_size == 5:
        # output
        W_out = weight_variable([28*28, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_5, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_5

    # 6 layer
    W_6 = weight_variable([28*28, 28*28])
    b_6 = bias_variable([28*28])
    h_6 = tf.nn.sigmoid(tf.matmul(h_5, W_6) + b_6)
    x, y = h_6.get_shape().to_list()
    jacobian = tf.matmul(tf.reshape((1.0 - h_6) * h_6, [x, 1, y]) * W_6, jacobian)

    if FLAGS.layer_size == 6:
        # output
        W_out = weight_variable([28*28, 10])
        b_out = bias_variable([10])

        y_out = tf.matmul(h_6, W_out) + b_out
        entropy_all = compute_entropy_with_svd(jacobian)
        return y_out, entropy_all, h_6

    # 7 layer
    W_7 = weight_variable([28*28, 28*28])
    b_7 = bias_variable([28*28])
    h_7 = tf.nn.sigmoid(tf.matmul(h_6, W_7) + b_7)
    x, y = h_7.get_shape().to_list()
    jacobian = tf.matmul(tf.reshape((1.0 - h_7) * h_7, [x, 1, y]) * W_7, jacobian)

    # output
    W_out = weight_variable([28*28, 10])
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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        ax = plt.subplot(gs[2*i + 1])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(layer_samples[i].reshape(28, 28), cmap='Greys_r')

    return fig


def main(argv):
    # data preparation
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # placeholder
    x = tf.placeholder(tf.float32, [None, 28*28])
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

    save_path = FLAGS.save_data_path + 'sigmoid_layer_{}_batch_{}_alpha_{}/'.format(
        FLAGS.layer_size, FLAGS.batch_size, FLAGS.learning_rate)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = tf.ConfigProto(
             gpu_options=tf.GPUOptions(
             per_process_gpu_memory_fraction=FLAGS.gpu_memory # 最大値の50%まで
             )
    )

    for _iter in range(FLAGS.iteration):
        print '=== iteration {} ==='.format(_iter)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(FLAGS.step):

                batch = mnist.train.next_batch(FLAGS.batch_size)

                _, cross_entropy_curr, train_accuracy = sess.run([
                    train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y_: batch[1], phase_train: True})

                train_accuracy_list_batch[_iter][i] = train_accuracy
                cross_entropy_list[_iter][i] = cross_entropy_curr

                if i % 5000 == 0 or i == FLAGS.step - 1:

                    test_accuracy = accuracy.eval(feed_dict={
                        x: mnist.test.images, y_: mnist.test.labels, phase_train: False})

                    test_accuracy_list[_iter][i] = test_accuracy

                    A = np.random.choice(len(mnist.test.images), FLAGS.entropy_num)
                    entropy_curr = entropy_all.eval(feed_dict={
                        x: mnist.test.images[A], phase_train: False})
                    entropy_list[_iter][i] = entropy_curr
                    """
                    true_entropy = 0.0
                    for i in range(100):
                        entropy_curr = entropy_all.eval(feed_dict={
                            x: mnist.test.images[i*FLAGS.entropy_num:(i+1)*FLAGS.entropy_num], phase_train: False})
                        true_entropy = true_entropy + entropy_curr
                    entropy_list[_iter][i] = true_entropy / 100.0
                    """

                    print('step %d, test accuracy %g, entropy %g' % (i, test_accuracy, entropy_curr))
                    print '---------------------'

                if _iter == 0 and (i % 10000 == 0 or i == FLAGS.step - 1):
                    samples = mnist.test.images[:18]
                    layer_samples = sess.run([h_last],
                                  feed_dict={x: samples, phase_train: False})
                    fig = plot(samples, layer_samples[0])
                    plt.savefig(save_path+'layer_samples_{}.png'.format(i))
                    plt.close()

    #np.save(save_path+'train_accuracy.npy', train_accuracy_list)
    np.save(save_path+'train_accuracy_batch.npy', train_accuracy_list_batch)
    np.save(save_path+'test_accuracy.npy', test_accuracy_list)
    np.save(save_path+'cross_entropy.npy', cross_entropy_list)
    np.save(save_path+'entropy.npy', entropy_list)


if __name__ == '__main__':

    tf.app.run(main=main)
