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
tf.app.flags.DEFINE_float('learning_rate', 0.01, "学習率")
tf.app.flags.DEFINE_integer('iteration', 1, "学習反復回数")
tf.app.flags.DEFINE_integer('step', 500000, "学習数")
tf.app.flags.DEFINE_integer('batch_size', 25, "バッチサイズ")
tf.app.flags.DEFINE_integer('layer_size', 5, "レイヤー数")
tf.app.flags.DEFINE_integer('entropy_num', 25, "entropy")
tf.app.flags.DEFINE_float('gpu_memory', 0.1, "gpuメモリ使用割合")
tf.app.flags.DEFINE_string('data_dir', './../../../../data/mnist/', "mnist保存先")
tf.app.flags.DEFINE_string('save_data_path', './../../../../data/classification/mnist/batch_change/', "データ保存先")


def deepnn(x, pred):
    # 1 layer
    with tf.variable_scope('classifier') as scope:
        W_1 = weight_variable([28*28, 28*28], 'W_1')
        b_1 = bias_variable([28*28], 'b_1')
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
    jacobian = W_1

    if FLAGS.layer_size == 1:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([28*28, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_1, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_1, singular_value

    # 2 layer
    with tf.variable_scope('classifier') as scope:
        W_2 = weight_variable([28*28, 28*28], 'W_2')
        b_2 = bias_variable([28*28], 'b_2')
    jacobian = tf.matmul(W_2, jacobian)

    if FLAGS.layer_size == 2:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([28*28, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_2, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_2, singular_value

    # 3 layer
    with tf.variable_scope('classifier') as scope:
        W_3 = weight_variable([28*28, 28*28], 'W_3')
        b_3 = bias_variable([28*28], 'b_3')
    h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)
    jacobian = tf.matmul(W_3, jacobian)

    if FLAGS.layer_size == 3:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([28*28, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_3, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_3, singular_value

    # 4 layer
    with tf.variable_scope('classifier') as scope:
        W_4 = weight_variable([28*28, 28*28], 'W_4')
        b_4 = bias_variable([28*28], 'b_4')
    h_4 = tf.nn.relu(tf.matmul(h_3, W_4) + b_4)
    jacobian = tf.matmul(W_4, jacobian)

    if FLAGS.layer_size == 4:
        # output
        with tf.variable_scope('classifier') as scope:
            W_out = weight_variable([28*28, 10], 'W_out')
            b_out = bias_variable([10], 'b_out')

        y_out = tf.matmul(h_4, W_out) + b_out
        singular_value = compute_svd(jacobian)
        return y_out, h_4, singular_value

    # 5 layer
    with tf.variable_scope('classifier') as scope:
        W_5 = weight_variable([28*28, 28*28], 'W_5')
        b_5 = bias_variable([28*28], 'b_5')
    h_5 = tf.nn.relu(tf.matmul(h_4, W_5) + b_5)
    jacobian = tf.matmul(W_5, jacobian)

    # output
    with tf.variable_scope('classifier') as scope:
        W_out = weight_variable([28*28, 10], 'W_out')
        b_out = bias_variable([10], 'b_out')

    y_out = tf.matmul(h_5, W_out) + b_out
    singular_value = compute_svd(jacobian)
    return y_out, h_5, singular_value


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
        train_singular_value_list = np.zeros((FLAGS.iteration, int(FLAGS.step / 5000.0) + 1, FLAGS.batch_size, 28*28), dtype=np.float32)
        test_singular_value_list = np.zeros((FLAGS.iteration, int(FLAGS.step / 5000.0) + 1, FLAGS.entropy_num, 28*28), dtype=np.float32)

    save_path = FLAGS.save_data_path + 'relu_up_layer_{}_batch_{}_alpha_{}/'.format(
        FLAGS.layer_size, FLAGS.batch_size, FLAGS.learning_rate)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = tf.ConfigProto(
             gpu_options=tf.GPUOptions(
             per_process_gpu_memory_fraction=FLAGS.gpu_memory # 最大値の50%まで
             )
    )

    I = 0
    for _iter in range(FLAGS.iteration):
        print '=== iteration {} ==='.format(_iter)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={pred: False})

            for i in range(FLAGS.step):

                batch = mnist.train.next_batch(FLAGS.batch_size)

                _, cross_entropy_curr, train_accuracy = sess.run([
                    train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y_: batch[1], pred: True})

                train_accuracy_list_batch[_iter][i] = train_accuracy
                cross_entropy_list[_iter][i] = cross_entropy_curr

                if i % 5000 == 0 or i == FLAGS.step - 1:
                    singular_value_curr = sess.run([singular_value], feed_dict={
                        x: batch[0], y_: batch[1], pred: False})
                    train_singular_value_list[_iter][I] = singular_value_curr

                    test_accuracy = accuracy.eval(feed_dict={
                        x: mnist.test.images, y_: mnist.test.labels, pred: False})

                    test_accuracy_list[_iter][i] = test_accuracy
                    A = np.random.choice(len(mnist.test.images), FLAGS.entropy_num)
                    singular_value_curr = sess.run([singular_value], feed_dict={
                        x: mnist.test.images[A], y_: mnist.test.labels[A], pred: False})
                    test_singular_value_list[_iter][I] = singular_value_curr

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
