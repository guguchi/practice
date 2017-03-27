# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
from gmm_sampler import *


class VanillaGAN(object):

    def __init__(self, sess, x_size, z_size, z_range, d_depths, g_depths, mb_size):
        self.sess = sess
        self.x_size = x_size
        self.z_size = z_size
        self.z_range = z_range
        self.d_depths = d_depths
        self.g_depths = g_depths
        self.mb_size = mb_size

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def sample_Z(self, m, n):
        return np.random.uniform(-self.z_range, self.z_range, size=[m, n])

    def discriminator(self, x, theta_D):
        [D_W5, D_W4, D_W3, D_W2, D_W1, D_b5, D_b4, D_b3, D_b2, D_b1] = theta_D
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
        D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
        D_logit = tf.matmul(D_h4, D_W5) + D_b5
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    def generator(self, z, theta_G):
        [G_W5, G_W4, G_W3, G_W2, G_W1, G_b5, G_b4, G_b3, G_b2, G_b1] = theta_G
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
        G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
        G_log_prob = tf.matmul(G_h4, G_W5) + G_b5
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_size])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_size])

        self.D_W1 = tf.Variable(self.xavier_init([self.x_size, self.d_depths[0]]), name='D_W1')
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.d_depths[0]]), name='D_b1')

        self.D_W2 = tf.Variable(self.xavier_init([self.d_depths[0], self.d_depths[1]]), name='D_W2')
        self.D_b2 = tf.Variable(tf.zeros(shape=[self.d_depths[1]]), name='D_b2')

        self.D_W3 = tf.Variable(self.xavier_init([self.d_depths[1], self.d_depths[2]]), name='D_W3')
        self.D_b3 = tf.Variable(tf.zeros(shape=[self.d_depths[2]]), name='D_b3')

        self.D_W4 = tf.Variable(self.xavier_init([self.d_depths[2], self.d_depths[3]]), name='D_W4')
        self.D_b4 = tf.Variable(tf.zeros(shape=[self.d_depths[3]]), name='D_b4')

        self.D_W5 = tf.Variable(self.xavier_init([self.d_depths[3], 1]), name='D_W5')
        self.D_b5 = tf.Variable(tf.zeros(shape=[1]), name='D_b5')

        self.G_W1 = tf.Variable(self.xavier_init([self.z_size, self.g_depths[0]]), name='G_W1')
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.g_depths[0]]), name='G_b1')

        self.G_W2 = tf.Variable(self.xavier_init([self.g_depths[0], self.g_depths[1]]), name='G_W2')
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.g_depths[1]]), name='G_b2')

        self.G_W3 = tf.Variable(self.xavier_init([self.g_depths[1], self.g_depths[2]]), name='G_W3')
        self.G_b3 = tf.Variable(tf.zeros(shape=[self.g_depths[2]]), name='G_b3')

        self.G_W4 = tf.Variable(self.xavier_init([self.g_depths[2], self.g_depths[3]]), name='G_W4')
        self.G_b4 = tf.Variable(tf.zeros(shape=[self.g_depths[3]]), name='G_b4')

        self.G_W5 = tf.Variable(self.xavier_init([self.g_depths[3], self.x_size]), name='G_W5')
        self.G_b5 = tf.Variable(tf.zeros(shape=[self.x_size]), name='G_b5')

        self.theta_D = [self.D_W5, self.D_W4, self.D_W3, self.D_W2, self.D_W1,
                        self.D_b5, self.D_b4, self.D_b3, self.D_b2, self.D_b1]
        self.theta_G = [self.G_W5, self.G_W4, self.G_W3, self.G_W2, self.G_W1,
                        self.G_b5, self.G_b4, self.G_b3, self.G_b2, self.G_b1]

        self.G_sample = self.generator(self.Z, self.theta_G)
        self.D_real = self.discriminator(self.X, self.theta_D)
        self.D_fake = self.discriminator(self.G_sample, self.theta_D)

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1.0 - self.D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake))

        self.saver = tf.train.Saver()

    def train_local(self, step, learning_rate_D, learning_rate_G,
                    num_cluster, scale, std, save_path):
        self.build_model()
        D_solver = tf.train.AdamOptimizer(learning_rate_D).minimize(self.D_loss,
                                                          var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer(learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)
        D_loss_list = []
        G_loss_list = []

        sess = tf.Session(max_to_keep=2500)
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for it in range(step):

            X_mb = gaussian_mixture_circle(self.mb_size, num_cluster, scale, std)

            _, D_loss_curr = sess.run([D_solver, self.D_loss], feed_dict={
                 self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size)})
            _, G_loss_curr = sess.run([G_solver, self.G_loss], feed_dict={
                              self.Z: self.sample_Z(self.mb_size, self.z_size)})

            if it % 1000 == 0 or it == step-1:
                self.saver.save(sess, save_path+'model.ckpt', global_step=it)
            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)

    def train(self, args):
        time_start = time.time()
        self.build_model()
        D_solver = tf.train.AdamOptimizer(args.learning_rate_D).minimize(self.D_loss,
                                                          var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer(args.learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)
        D_loss_list = []
        G_loss_list = []

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        for it in range(args.step):

            X_mb = gaussian_mixture_circle(self.mb_size, args.num_cluster,
                                           args.scale, args.std)

            _, D_loss_curr = sess.run([D_solver, self.D_loss], feed_dict={
                self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size)})
            _, G_loss_curr = sess.run([G_solver, self.G_loss], feed_dict={
                              self.Z: self.sample_Z(self.mb_size, self.z_size)})

            if it % 1000 == 0 or it == args.step-1:
                self.saver.save(sess,
                                args.save_path+'model.ckpt'.format(
                                args.num_cluster, args.learning_rate_D,
                                args.learning_rate_G), global_step=it)
            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)

        np.save(args.save_path+'d_loss_list.npy', D_loss_list)
        np.save(args.save_path+'g_loss_list.npy', G_loss_list)
        elapsed_time = time.time()-time_start
        print ("birth:{}".format(elapsed_time)) + "[sec]"
        print '---'

    def restore(self):
        self.build_model()

        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state('../../data/gan_analysis/')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            saver.restore(self.sess, '../../data/gan_analysis/model.ckpt-29')

        else:
            init = tf.initialize_all_variables()
            self.sess.run(init)
