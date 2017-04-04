# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from ops import batch_norm, batch_norm_wrapper


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

    def lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def sample_Z(self, m, n):
        return np.random.uniform(-self.z_range, self.z_range, size=[m, n])

    def discriminator(self, x, phase_train, scope_reuse):
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
                h = self.lrelu(tf.matmul(h, W) + b)

            W = self.get_weights(
                    name = 'D_W{}'.format(N-1),
                    size = [self.d_depths[N-2], self.d_depths[N-1]],
                    stddev = self.xavier_init(self.g_depths[N-2])
            )
            b = self.get_biases(
                    name = 'D_b{}'.format(N-1),
                    size = [self.g_depths[N-1]],
                    value = 0.0
            )
            out = tf.nn.sigmoid(tf.matmul(h, W) + b)
        return out

    def generator(self, z, phase_train):
        N = len(self.g_depths)
        scope_reuse = False
        with tf.variable_scope("generator") as scope:
            h = z
            for index in range(N - 2):
                W = self.get_weights(
                        name = 'G_W{}'.format(index+1),
                        size = [self.g_depths[index], self.g_depths[index+1]],
                        stddev = self.xavier_init(self.g_depths[index])
                )
                h = self.lrelu(batch_norm_wrapper(tf.matmul(h, W), phase_train,
                               'g_bn_{}'.format(index), scope_reuse))

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
            out = tf.nn.sigmoid(tf.matmul(h, W) + b)
        return out

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_size])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_size])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        self.G_sample = self.generator(self.Z, self.phase_train)
        self.D_real = self.discriminator(self.X, self.phase_train, scope_reuse = False)
        self.D_fake = self.discriminator(self.G_sample, self.phase_train, scope_reuse = True)

        train_variables = tf.trainable_variables()
        self.theta_G = [v for v in train_variables if v.name.startswith("generator")]
        self.theta_D = [v for v in train_variables if v.name.startswith("discriminator")]

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1.0 - self.D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake))

        self.saver = tf.train.Saver(max_to_keep=2500)

    def train_local(self, step, learning_rate_D, learning_rate_G, save_path):
        self.build_model()
        D_solver = tf.train.AdamOptimizer(learning_rate_D).minimize(self.D_loss,
                                                          var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer(learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)
        D_loss_list = []
        G_loss_list = []

        mnist = input_data.read_data_sets('../../data/minst/MNIST_data', one_hot=True)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for it in range(step):

            X_mb, _ = mnist.train.next_batch(self.mb_size)

            _, D_loss_curr = self.sess.run([D_solver, self.D_loss], feed_dict={
                 self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                 self.phase_train: True})
            _, G_loss_curr = self.sess.run([G_solver, self.G_loss], feed_dict={
                              self.Z: self.sample_Z(self.mb_size, self.z_size),
                              self.phase_train: True})

            print self.sess.run([self.theta_G])
            print D_loss_curr
            print G_loss_curr

            if it % 100 == 0 or it == step-1:
                self.saver.save(self.sess, save_path+'model.ckpt', global_step=it)

                samples = self.sess.run([self.G_sample],
                              feed_dict={self.Z: self.sample_Z(36, self.z_size), self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(save_path+'{}.png'.format(it), bbox_inches='tight')
                plt.close(fig)

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

        mnist = input_data.read_data_sets(args.mnist_data_path, one_hot=True)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(args.save_data_path):
            os.makedirs(args.save_data_path)

        if not os.path.exists(args.save_fig_path):
            os.makedirs(args.save_fig_path)

        for it in range(args.step):

            X_mb, _ = mnist.train.next_batch(self.mb_size)

            _, D_loss_curr = self.sess.run([D_solver, self.D_loss], feed_dict={
                self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                self.phase_train: True})
            _, G_loss_curr = self.sess.run([G_solver, self.G_loss], feed_dict={
                              self.Z: self.sample_Z(self.mb_size, self.z_size),
                              self.phase_train: True})

            D_loss_list.append(D_loss_curr)
            G_loss_list.append(G_loss_curr)

            if it % 2000 == 0:
                self.saver.save(self.sess,
                                args.save_data_path+'model.ckpt', global_step=it)

                samples = self.sess.run([self.G_sample],
                             feed_dict={self.Z: self.sample_Z(36, self.z_size),
                             self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(args.save_fig_path+'step_{}.png'.format(it))
                plt.close()

            if it == args.step-1:
                self.saver.save(self.sess,
                                args.save_data_path+'model.ckpt', global_step=it)

                samples = self.sess.run([self.G_sample],
                             feed_dict={self.Z: self.sample_Z(36, self.z_size),
                             self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(args.save_fig_path+'step_{}.png'.format(it))
                plt.close()

                plt.plot(np.arange(len(D_loss_list)), D_loss_list, label='D')
                plt.plot(np.arange(len(G_loss_list)), G_loss_list, label='G')
                plt.legend()
                plt.savefig(args.save_fig_path+'loss.png')
                plt.close()

        np.save(args.save_data_path+'d_loss_list.npy', D_loss_list)
        np.save(args.save_data_path+'g_loss_list.npy', G_loss_list)

    def plot(self, samples):
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace=0.05, hspace=0.05)

        for (i, sample) in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig
