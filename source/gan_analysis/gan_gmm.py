# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from gmm_sampler import *


def gan_gmm(z_size=1, z_range=1.0, learning_rate_D=0.001, learning_rate_G=0.001,
            step=1000000, mb_size = 128, num_cluster=8, scale=2, std=0.2):

    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)


    X = tf.placeholder(tf.float32, shape=[None, 2])

    D_W1 = tf.Variable(xavier_init([2, 10]))
    D_b1 = tf.Variable(tf.zeros(shape=[10]))

    D_W2 = tf.Variable(xavier_init([10, 8]))
    D_b2 = tf.Variable(tf.zeros(shape=[8]))

    D_W3 = tf.Variable(xavier_init([8, 6]))
    D_b3 = tf.Variable(tf.zeros(shape=[6]))

    D_W4 = tf.Variable(xavier_init([6, 3]))
    D_b4 = tf.Variable(tf.zeros(shape=[3]))

    D_W5 = tf.Variable(xavier_init([3, 1]))
    D_b5 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5, D_b1, D_b2, D_b3, D_b4, D_b5]

    def discriminator(x):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
        D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
        D_logit = tf.matmul(D_h4, D_W5) + D_b5
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob


    Z = tf.placeholder(tf.float32, shape=[None, z_size])

    G_W1 = tf.Variable(xavier_init([z_size, 8]))
    G_b1 = tf.Variable(tf.zeros(shape=[8]))

    G_W2 = tf.Variable(xavier_init([8, 6]))
    G_b2 = tf.Variable(tf.zeros(shape=[6]))

    G_W3 = tf.Variable(xavier_init([6, 4]))
    G_b3 = tf.Variable(tf.zeros(shape=[4]))

    G_W4 = tf.Variable(xavier_init([4, 3]))
    G_b4 = tf.Variable(tf.zeros(shape=[3]))

    G_W5 = tf.Variable(xavier_init([3, 2]))
    G_b5 = tf.Variable(tf.zeros(shape=[2]))

    theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2, G_b3, G_b4, G_b5]


    def sample_Z(m, n):
        return np.random.uniform(-z_range, z_range, size=[m, n])


    def generator(z):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
        G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
        G_log_prob = tf.matmul(G_h4, G_W5) + G_b5
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob


    G_sample = generator(Z)
    D_real = discriminator(X)
    D_fake = discriminator(G_sample)

    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))

    D_solver = tf.train.AdamOptimizer(learning_rate_D).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate_G).minimize(G_loss, var_list=theta_G)

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

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, z_size)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, z_size)})

        saver.save(sess, '../../data/gan_analysis/model_gan.ckpt', global_step=100)
        D_loss_list.append(D_loss_curr)
        G_loss_list.append(G_loss_curr)

    np.save('../../data/gan_analysis/d_loss_gan.npy', np.array(D_loss_list))
    np.save('../../data/gan_analysis/g_loss_gan.npy', np.array(G_loss_list))



gan_gmm(z_size=1, z_range=1.0, learning_rate_D=0.001, learning_rate_G=0.001,
            step=1000000, mb_size = 128, num_cluster=8, scale=2, std=0.2)
