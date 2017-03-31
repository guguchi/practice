# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
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

    def xavier_init(self, in_dim):
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return xavier_stddev

    def get_weights(self, name, size, stddev):
        var = tf.get_variable(
            name = name,
            shape = size,
            initializer = tf.truncated_normal_initializer(stddev = stddev))
        return var

    def get_biases(self, name, size, value):
        var = tf.get_variable(
            name = name,
            shape = size,
            initializer = tf.constant_initializer(value = value))
        return var

    def sample_Z(self, m, n):
        return np.random.uniform(-self.z_range, self.z_range, size=[m, n])

    def discriminator(self, x, scope_reuse):
        N = len(self.d_depths)
        with tf.variable_scope("discriminator") as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = x
            for index in range(N - 2):
                W = self.get_weights(
                        name = 'D_W{}'.format(index+1),
                        size = [self.d_depths[index], self.d_depths[index+1]],
                        stddev = self.clip_value / 2.
                )
                b = self.get_biases(
                        name = 'D_b{}'.format(index+1),
                        size = [self.d_depths[index+1]],
                        value = 0.0
                )
                h = tf.nn.relu(tf.matmul(h, W) + b)

            W = self.get_weights(
                    name = 'D_W{}'.format(N-1),
                    size = [self.d_depths[N-2], self.d_depths[N-1]],
                    stddev = self.clip_value / 2.
            )
            b = self.get_biases(
                    name = 'D_b{}'.format(N-1),
                    size = [self.d_depths[N-1]],
                    value = 0.0
            )
            out = tf.matmul(h, W) + b
        return out

    def generator(self, z):
        N = len(self.g_depths)
        with tf.variable_scope("generator") as scope:
            h = z
            for index in range(N - 2):
                W = self.get_weights(
                        name = 'G_W{}'.format(index+1),
                        size = [self.g_depths[index], self.g_depths[index+1]],
                        stddev = self.xavier_init(self.g_depths[index])
                )
                b = self.get_biases(
                        name = 'G_b{}'.format(index+1),
                        size = [self.g_depths[index+1]],
                        value = 0.0
                )
                h = tf.nn.relu(tf.matmul(h, W) + b)

            W = self.get_weights(
                    name = 'G_W{}'.format(N-1),
                    size = [self.g_depths[N-2], self.g_depths[N-1]],
                    stddev = self.xavier_init(self.g_depths[N-2])
            )
            b = self.get_biases(
                    name = 'G_b{}'.format(N-1),
                    size = [self.g_depths[N-1]],
                    value = 0.0
            )
            out = tf.matmul(h, W) + b
        return out

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_size])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_size])

        self.G_sample = self.generator(self.Z)
        self.D_real = self.discriminator(self.X, scope_reuse = False)
        self.D_fake = self.discriminator(self.G_sample, scope_reuse = True)

        train_variables = tf.trainable_variables()
        self.theta_G = [v for v in train_variables if v.name.startswith("generator")]
        self.theta_D = [v for v in train_variables if v.name.startswith("discriminator")]

        self.D_loss = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        self.G_loss = -tf.reduce_mean(self.D_fake)

        self.saver = tf.train.Saver(max_to_keep=2500)

    def train_local(self, g_iteration, d_iteration, learning_rate_D, learning_rate_G,
                    num_cluster, scale, std, sample_size, save_path):
        self.build_model()
        D_solver = tf.train.RMSPropOptimizer(learning_rate_D).minimize(self.D_loss,
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
            print D_loss_curr
            print G_loss_curr

            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)

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

            if g_iter % 1000 == 0:
                self.saver.save(self.sess, args.save_data_path+'model.ckpt', global_step=g_iter)

                sample = self.sess.run([self.G_sample],
                            feed_dict={self.Z: self.sample_Z(args.sample_size, self.z_size)}
                )

                X_mb = gaussian_mixture_circle(args.sample_size, args.num_cluster, args.scale, args.std)

                plt.plot(sample[0].T[0], sample[0].T[1], 'o', ms = 1, label='sampler')
                plt.plot(X_mb.T[0], X_mb.T[1], 'o', ms = 1, label='true')
                plt.legend()
                plt.savefig(args.save_fig_path+'step_{}.png'.format(g_iter))
                plt.close()

            if g_iter == args.g_iteration-1:
                self.saver.save(self.sess, args.save_data_path+'model.ckpt', global_step=g_iter)

                sample = self.sess.run([self.G_sample],
                            feed_dict={self.Z: self.sample_Z(args.sample_size, self.z_size)}
                )

                X_mb = gaussian_mixture_circle(args.sample_size, args.num_cluster, args.scale, args.std)

                plt.plot(sample[0].T[0], sample[0].T[1], 'o', ms = 1, label='sampler')
                plt.plot(X_mb.T[0], X_mb.T[1], 'o', ms = 1, label='true')
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
