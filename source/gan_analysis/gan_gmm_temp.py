# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from gmm_sampler import *


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n, z_range):
    return np.random.uniform(-z_range, z_range, size=[m, n])


class Generator:

    def __init__(self, z_size, x_size, depths=[10, 8, 6, 4]):
        G_W1 = tf.Variable(xavier_init([z_size, depths[0]]), name='G_W1')
        G_b1 = tf.Variable(tf.zeros(shape=[depths[0]]), name='G_b1')

        G_W2 = tf.Variable(xavier_init([depths[0], depths[1]]), name='G_W2')
        G_b2 = tf.Variable(tf.zeros(shape=[depths[1]]), name='G_b2')

        G_W3 = tf.Variable(xavier_init([depths[1], depths[2]]), name='G_W3')
        G_b3 = tf.Variable(tf.zeros(shape=[depths[2]]), name='G_b3')

        G_W4 = tf.Variable(xavier_init([depths[2], depths[3]]), name='G_W4')
        G_b4 = tf.Variable(tf.zeros(shape=[depths[3]]), name='G_b4')

        G_W5 = tf.Variable(xavier_init([depths[3], x_size]), name='G_W5')
        G_b5 = tf.Variable(tf.zeros(shape=[x_size]), name='G_b5')

        self.W1 = G_W1
        self.W2 = G_W2
        self.W3 = G_W3
        self.W4 = G_W4
        self.W5 = G_W5

        self.b1 = G_b1
        self.b2 = G_b2
        self.b3 = G_b3
        self.b4 = G_b4
        self.b5 = G_b5

        self.theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5,
                        G_b1, G_b2, G_b3, G_b4, G_b5]

    def outputs(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.W1) + self.b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, self.W2) + self.b2)
        G_h3 = tf.nn.relu(tf.matmul(G_h2, self.W3) + self.b3)
        G_h4 = tf.nn.relu(tf.matmul(G_h3, self.W4) + self.b4)
        G_log_prob = tf.matmul(G_h4, self.W5) + self.b5
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob


class Discriminator:

    def __init__(self, x_size, depths=[10, 8, 6, 4]):
        D_W1 = tf.Variable(xavier_init([x_size, depths[0]]), name='D_W1')
        D_b1 = tf.Variable(tf.zeros(shape=[depths[0]]), name='D_b1')

        D_W2 = tf.Variable(xavier_init([depths[0], depths[1]]), name='D_W2')
        D_b2 = tf.Variable(tf.zeros(shape=[depths[1]]), name='D_b2')

        D_W3 = tf.Variable(xavier_init([depths[1], depths[2]]), name='D_W3')
        D_b3 = tf.Variable(tf.zeros(shape=[depths[2]]), name='D_b3')

        D_W4 = tf.Variable(xavier_init([depths[2], depths[3]]), name='D_W4')
        D_b4 = tf.Variable(tf.zeros(shape=[depths[3]]), name='D_b4')

        D_W5 = tf.Variable(xavier_init([depths[3], 1]), name='D_W5')
        D_b5 = tf.Variable(tf.zeros(shape=[1]), name='D_b5')

        self.W1 = D_W1
        self.W2 = D_W2
        self.W3 = D_W3
        self.W4 = D_W4
        self.W5 = D_W5

        self.b1 = D_b1
        self.b2 = D_b2
        self.b3 = D_b3
        self.b4 = D_b4
        self.b5 = D_b5

        self.theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5,
                        D_b1, D_b2, D_b3, D_b4, D_b5]

    def outputs(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.W2) + self.b2)
        D_h3 = tf.nn.relu(tf.matmul(D_h2, self.W3) + self.b3)
        D_h4 = tf.nn.relu(tf.matmul(D_h3, self.W4) + self.b4)
        D_logit = tf.matmul(D_h4, self.W5) + self.b5
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob


def train_gan(z_size=1, z_range=1.0, x_size=2, g_depths=[10, 8, 6, 4], d_depths=[10, 8, 6, 4],
              learning_rate_D=0.001, learning_rate_G=0.001,
              step=1000, mb_size = 128, num_cluster=8, scale=2, std=0.2):

    generator = Generator(z_size, x_size, g_depths)
    discriminator = Discriminator(x_size, d_depths)

    X = tf.placeholder(tf.float32, shape=[None, x_size])
    Z = tf.placeholder(tf.float32, shape=[None, z_size])

    G_sample = generator.outputs(Z)
    D_real = discriminator.outputs(X)
    D_fake = discriminator.outputs(G_sample)

    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))

    D_solver = tf.train.AdamOptimizer(learning_rate_D).minimize(D_loss)
    G_solver = tf.train.AdamOptimizer(learning_rate_G).minimize(G_loss)

    saver = tf.train.Saver()
    D_loss_list = []
    G_loss_list = []

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('../../data/gan_analysis/'):
        os.makedirs('../../data/gan_analysis/')

    for it in range(step):
        print 'step:{}'.format(it)

        X_mb = gaussian_mixture_circle(mb_size, num_cluster, scale, std)

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, z_size, z_range)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, z_size, z_range)})

        saver.save(sess, '../../data/gan_analysis/model_gan.ckpt', global_step=10)
        D_loss_list.append(D_loss_curr)
        G_loss_list.append(G_loss_curr)

    np.save('../../data/gan_analysis/d_loss_gan.npy', np.array(D_loss_list))
    np.save('../../data/gan_analysis/g_loss_gan.npy', np.array(G_loss_list))


if __name__ == '__main__':

    train_gan(z_size=1, x_size=2, g_depths=[10, 8, 6, 4], d_depths=[10, 8, 6, 4],
              learning_rate_D=0.001, learning_rate_G=0.001,
              step=20, mb_size = 128, num_cluster=8, scale=2, std=0.2)
