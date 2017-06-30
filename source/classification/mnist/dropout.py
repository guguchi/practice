# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.
    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). keep_prob is a scalar placeholder for the probability of
        dropout.
        """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = x
    keep_prob = tf.placeholder(tf.float32)

    # 1 layer
    W_1 = weight_variable([784, 784])
    b_1 = bias_variable([784])
    h_1 = tf.nn.relu(tf.matmul(x_image, W_1) + b_1)
    h_1_out = tf.nn.dropout(h_1, keep_prob)

    # 2 layer
    W_2 = weight_variable([784, 784])
    b_2 = bias_variable([784])
    h_2 = tf.nn.relu(tf.matmul(h_1_out, W_2) + b_2)
    h_2_out = tf.nn.dropout(h_2, keep_prob)

    # 3 layer
    W_3 = weight_variable([784, 784])
    b_3 = bias_variable([784])
    h_3 = tf.nn.relu(tf.matmul(h_2_out, W_3) + b_3)
    h_3_out = tf.nn.dropout(h_3, keep_prob)

    # 4 layer
    W_4 = weight_variable([784, 784])
    b_4 = bias_variable([784])
    h_4 = tf.nn.relu(tf.matmul(h_3_out, W_4) + b_4)
    h_4_out = tf.nn.dropout(h_4, keep_prob)

    # 5 layer
    W_5 = weight_variable([784, 784])
    b_5 = bias_variable([784])
    h_5 = tf.nn.relu(tf.matmul(h_4_out, W_5) + b_5)
    h_5_out = tf.nn.dropout(h_5, keep_prob)

    # output
    W_out = weight_variable([784, 10])
    b_out = bias_variable([10])

    y_out = tf.matmul(h_5_out, W_out) + b_out
    return y_out, keep_prob


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    _dropout = 0.5
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_out, keep_prob = deepnn(x)

    # objective
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_accuracy_list = []
    train_accuracy_list = []
    cross_entropy_list = []
    save_data_path = '/home/ishii/Desktop/research/practice/data/classification/dropout/0630/'
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1

    for _iter in range(10):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(20000):
                batch = mnist.train.next_batch(50)

                _, cross_entropy_curr, train_accuracy = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: _dropout})
                train_accuracy_list.append(train_accuracy)
                cross_entropy_list.append(cross_entropy_curr)

                if i % 100 == 0:
                    test_accuracy = accuracy.eval(feed_dict={
                        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                    print('step %d, test accuracy %g' % (i, test_accuracy))
                    test_accuracy_list.append(test_accuracy)

            test_accuracy = accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print('test accuracy %g' % test_accuracy)
            test_accuracy_list.append(test_accuracy)

            np.save(save_data_path+'test_accuracy_vanilla_dropout_{}_{}.npy'.format(_dropout, _iter), test_accuracy_list)
            np.save(save_data_path+'train_accuracy_vanilla_dropout_{}_{}.npy'.format(_dropout, _iter), train_accuracy_list)
            np.save(save_data_path+'cross_entropy_vanilla_dropout_{}_{}.npy'.format(_dropout, _iter), cross_entropy_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
