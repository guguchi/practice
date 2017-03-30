# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from gmm_sampler import *


class WassersteinGAN(object):

    def __init__(self, sess, x_size, z_size, z_range, d_depths, g_depths,
                 mb_size, clip_value):
        self.sess = sess
        self.x_size = x_size
        self.z_size = z_size
        self.z_range = z_range
        self.d_depths = d_depths
        self.g_depths = g_depths
        self.mb_size = mb_size
        self.clip_value = clip_value

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
        return D_logit

    def generator(self, z, theta_G):
        [G_W5, G_W4, G_W3, G_W2, G_W1, G_b5, G_b4, G_b3, G_b2, G_b1] = theta_G
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
        G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
        G_log_prob = tf.matmul(G_h4, G_W5) + G_b5
        return G_log_prob

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

        self.D_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake)
        self.G_loss = -tf.reduce_mean(self.D_fake)

        self.saver = tf.train.Saver(max_to_keep=2500)

    def train_local(self, g_iteration, d_iteration, learning_rate_D, learning_rate_G,
                    num_cluster, scale, std, sample_size, save_path):
        self.build_model()
        D_solver = tf.train.RMSPropOptimizer(learning_rate_D).minimize(-self.D_loss,
                                                          var_list=self.theta_D)
        G_solver = tf.train.RMSPropOptimizer(learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)

        clip_D = [p.assign(tf.clip_by_value(p, -self.clip_value,
                                        self.clip_value)) for p in self.theta_D]

        D_loss_list = []
        G_loss_list = []

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for g_iter in range(g_iteration):
            print 'g_iter:{}'.format(g_iter)
            for d_iter in range(d_iteration):
                X_mb = gaussian_mixture_circle(self.mb_size, num_cluster, scale, std)

                _, D_loss_curr, _ = self.sess.run(
                        [D_solver, self.D_loss, clip_D],
                        feed_dict={self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size)}
                )

            _, G_loss_curr = self.sess.run([G_solver, self.G_loss],
                   feed_dict={self.Z: self.sample_Z(self.mb_size, self.z_size)}
            )

            if g_iter % 1000 == 0 or g_iter == g_iteration-1:
                self.saver.save(self.sess, save_path+'model.ckpt', global_step=g_iter)

                sample = self.sess.run([self.G_sample],
                            feed_dict={self.Z: self.sample_Z(sample_size, self.z_size)}
                )

                X_mb = gaussian_mixture_circle(sample_size, num_cluster, scale, std)

                plt.plot(sample[0].T[0], sample[0].T[1], 'o', label='sampler')
                plt.plot(X_mb.T[0], X_mb.T[1], 'o', label='true')
                plt.legend()
                plt.savefig(save_path+'step_{}.png'.format(g_iter))
                plt.close()

            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)


    def train(self, args):
        self.build_model()
        D_solver = tf.train.RMSPropOptimizer(args.learning_rate_D).minimize(-self.D_loss,
                                                          var_list=self.theta_D)
        G_solver = tf.train.RMSPropOptimizer(args.learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)

        clip_D = [p.assign(tf.clip_by_value(p, -self.clip_value,
                                        self.clip_value)) for p in self.theta_D]

        D_loss_list = []
        G_loss_list = []

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(args.save_data_path):
            os.makedirs(args.save_data_path)

        if not os.path.exists(args.save_fig_path):
            os.makedirs(args.save_fig_path)

        for g_iter in range(args.g_iteration):
            for d_iter in range(args.d_iteration):
                X_mb = gaussian_mixture_circle(self.mb_size, args.num_cluster, args.scale, args.std)

                _, D_loss_curr, _ = self.sess.run(
                        [D_solver, self.D_loss, clip_D],
                        feed_dict={self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size)}
                )

            _, G_loss_curr = self.sess.run([G_solver, self.G_loss],
                   feed_dict={self.Z: self.sample_Z(self.mb_size, self.z_size)}
            )

            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)

            if g_iter % 1000 == 0 or g_iter == args.g_iteration-1:
                self.saver.save(self.sess, args.save_data_path+'model.ckpt', global_step=g_iter)

                sample = self.sess.run([self.G_sample],
                            feed_dict={self.Z: self.sample_Z(args.sample_size, self.z_size)}
                )

                X_mb = gaussian_mixture_circle(args.sample_size, args.num_cluster, args.scale, args.std)

                plt.plot(sample[0].T[0], sample[0].T[1], 'o', label='sampler')
                plt.plot(X_mb.T[0], X_mb.T[1], 'o', label='true')
                plt.legend()
                plt.savefig(args.save_fig_path+'step_{}.png'.format(g_iter))
                plt.close()

        plt.plot(np.arange(len(D_loss_list)), D_loss_list, label='D_loss')
        plt.plot(np.arange(len(D_loss_list)), G_loss_list, label='G_loss')
        plt.legend()
        plt.savefig(args.save_fig_path+'loss.png')
        plt.close()
        np.savetxt(args.save_data_path+'d_loss.out', D_loss_list)
        np.savetxt(args.save_data_path+'g_loss.out', G_loss_list)
