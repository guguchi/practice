# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
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
                        stddev = self.xavier_init(self.g_depths[index])
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
                    stddev = self.xavier_init(self.g_depths[N-2])
            )
            b = self.get_biases(
                    name = 'D_b{}'.format(N-1),
                    size = [self.d_depths[N-1]],
                    value = 0.0
            )
            out = tf.nn.sigmoid(tf.matmul(h, W) + b)
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

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1.0 - self.D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake))

        self.saver = tf.train.Saver(max_to_keep=2500)

    def train_local(self, step, learning_rate_D, learning_rate_G,
                    num_cluster, scale, std, save_path):
        self.build_model()
        D_solver = tf.train.AdamOptimizer(learning_rate_D).minimize(self.D_loss,
                                                          var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer(learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)
        D_loss_list = []
        G_loss_list = []

        #sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for it in range(step):

            X_mb = gaussian_mixture_circle(self.mb_size, num_cluster, scale, std)

            _, D_loss_curr = self.sess.run([D_solver, self.D_loss], feed_dict={
                 self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size)})
            _, G_loss_curr = self.sess.run([G_solver, self.G_loss], feed_dict={
                              self.Z: self.sample_Z(self.mb_size, self.z_size)})

            print self.sess.run([self.theta_G])
            print D_loss_curr
            print G_loss_curr

            if it % 1000 == 0 or it == step-1:
                self.saver.save(self.sess, save_path+'model.ckpt', global_step=it)
            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)

    def train(self, args):
        self.build_model()
        D_solver = tf.train.AdamOptimizer(args.learning_rate_D).minimize(self.D_loss,
                                                          var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer(args.learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)
        D_loss_list = []
        G_loss_list = []

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(args.save_data_path):
            os.makedirs(args.save_data_path)

        if not os.path.exists(args.save_fig_path):
            os.makedirs(args.save_fig_path)

        for it in range(args.step):

            X_mb = gaussian_mixture_circle(self.mb_size, args.num_cluster,
                                           args.scale, args.std)

            _, D_loss_curr = self.sess.run([D_solver, self.D_loss], feed_dict={
                self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size)})
            _, G_loss_curr = self.sess.run([G_solver, self.G_loss], feed_dict={
                              self.Z: self.sample_Z(self.mb_size, self.z_size)})

            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)

            if it % 1000 == 0:
                self.saver.save(self.sess,
                                args.save_data_path+'model.ckpt'.format(
                                args.num_cluster, args.learning_rate_D,
                                args.learning_rate_G), global_step=it)

                sample = self.sess.run([self.G_sample], feed_dict={self.Z: self.sample_Z(args.sample_size, self.z_size)})

                X_mb = gaussian_mixture_circle(args.sample_size, args.num_cluster, args.scale, args.std)

                plt.plot(sample[0].T[0], sample[0].T[1], 'o', label='sampler')
                plt.plot(X_mb.T[0], X_mb.T[1], 'o', label='true')
                plt.legend()
                plt.savefig(args.save_fig_path+'step_{}.png'.format(it))
                plt.close()

            if it == args.step-1:
                self.saver.save(self.sess,
                                args.save_data_path+'model.ckpt'.format(
                                args.num_cluster, args.learning_rate_D,
                                args.learning_rate_G), global_step=it)

                sample = self.sess.run([self.G_sample], feed_dict={self.Z: self.sample_Z(args.sample_size, self.z_size)})

                X_mb = gaussian_mixture_circle(args.sample_size, args.num_cluster, args.scale, args.std)

                plt.plot(sample[0].T[0], sample[0].T[1], 'o', label='sampler')
                plt.plot(X_mb.T[0], X_mb.T[1], 'o', label='true')
                plt.legend()
                plt.savefig(args.save_fig_path+'step_{}.png'.format(it))
                plt.close()

                plt.plot(np.arange(len(D_loss_curr)), D_loss_curr, label='D')
                plt.plot(np.arange(len(G_loss_curr)), G_loss_curr, label='G')
                plt.legend()
                plt.savefig(args.save_fig_path+'loss.png')
                plt.close()

        np.save(args.save_data_path+'d_loss_list.npy', D_loss_list)
        np.save(args.save_data_path+'g_loss_list.npy', G_loss_list)

    def sample_generator(self):
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_size])

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

        self.theta_G = [self.G_W5, self.G_W4, self.G_W3, self.G_W2, self.G_W1,
                        self.G_b5, self.G_b4, self.G_b3, self.G_b2, self.G_b1]

        G_sample = self.generator(self.Z, self.theta_G)

        self.saver = tf.train.Saver()

        return G_sample


    def restore_local(self, path, step, sample_size, num_cluster, scale, std):
        G_sample = self.sample_generator()

        ckpt = tf.train.get_checkpoint_state('../../data/gan_analysis/'+path)
        if ckpt:
            self.saver.restore(self.sess, '../../data/gan_analysis/'+path+'model.ckpt-{}'.format(step))
            sample = self.sess.run([G_sample], feed_dict={self.Z: self.sample_Z(sample_size, self.z_size)})

            X_mb = gaussian_mixture_circle(sample_size, num_cluster, scale, std)

            plt.plot(sample[0].T[0], sample[0].T[1], 'o', label='sampler')
            plt.plot(X_mb.T[0], X_mb.T[1], 'o', label='true')
            plt.legend()
            plt.show()

        else:
            print 'nothing model.ckpt'

    def restore(self, args):
        G_sample = self.sample_generator()

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        ckpt = tf.train.get_checkpoint_state(args.data_path)
        if ckpt:
            self.saver.restore(self.sess, args.data_path+'model.ckpt-{}'.format(args.step))
            sample = self.sess.run([G_sample], feed_dict={self.Z: self.sample_Z(args.sample_size, self.z_size)})

            X_mb = gaussian_mixture_circle(args.sample_size, args.num_cluster, args.scale, args.std)

            plt.plot(sample[0].T[0], sample[0].T[1], 'o', label='sampler')
            plt.plot(X_mb.T[0], X_mb.T[1], 'o', label='true')
            plt.legend()
            plt.savefig(args.save_path+'gmm_{}.png'.format(args.step))
            plt.close()

        else:
            print 'nothing model.ckpt'
