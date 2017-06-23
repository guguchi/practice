# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from ops import batch_norm_wrapper


class AGE(object):

    def __init__(self, sess, x_size, z_size, z_range, e_depths, g_depths, mb_size,
                 lam, mu):
        self.sess = sess
        self.x_size = x_size
        self.z_size = z_size
        self.z_range = z_range
        self.e_depths = e_depths
        self.g_depths = g_depths
        self.mb_size = mb_size
        self.lam = lam
        self.mu = mu

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

    def encoder(self, x, phase_train, scope_reuse):
        N_e = len(self.e_depths)
        with tf.variable_scope('encoder') as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = x
            for index in range(N_e - 2):
                W = self.get_weights(
                        name = 'E_W{}'.format(index+1),
                        size = [self.e_depths[index], self.e_depths[index+1]],
                        stddev = self.xavier_init(self.e_depths[index])
                )
                """
                b = self.get_biases(
                        name = 'E_b{}'.format(index+1),
                        size = [self.g_depths[index+1]],
                        value = 0.0
                )
                """
                h = self.lrelu(batch_norm_wrapper(tf.matmul(h, W), phase_train,
                               'e_bn_{}'.format(index), scope_reuse))
                #h = tf.matmul(h, W) + b

            W = self.get_weights(
                    name = 'E_W{}'.format(N_e-1),
                    size = [self.e_depths[N_e-2], self.e_depths[N_e-1]],
                    stddev = self.xavier_init(self.e_depths[N_e-2])
            )
            b = self.get_biases(
                    name = 'E_b{}'.format(N_e-1),
                    size = [self.e_depths[N_e-1]],
                    value = 0.0
            )
            out = tf.matmul(h, W) + b
        return out

    def generator(self, z, phase_train, scope_reuse):
        N_g = len(self.g_depths)
        with tf.variable_scope('generator') as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = z
            for index in range(N_g - 2):
                W = self.get_weights(
                        name = 'G_W{}'.format(index+1),
                        size = [self.g_depths[index], self.g_depths[index+1]],
                        stddev = self.xavier_init(self.g_depths[index])
                )
                """
                b = self.get_biases(
                        name = 'G_b{}'.format(index+1),
                        size = [self.g_depths[index+1]],
                        value = 0.0
                )
                """
                h = self.lrelu(batch_norm_wrapper(tf.matmul(h, W), phase_train,
                               'g_bn_{}'.format(index), scope_reuse))
                #h = tf.matmul(h, W) + b

            W = self.get_weights(
                    name = 'G_W{}'.format(N_g-1),
                    size = [self.g_depths[N_g-2], self.g_depths[N_g-1]],
                    stddev = self.xavier_init(self.g_depths[N_g-2])
            )
            b = self.get_biases(
                    name = 'G_b{}'.format(N_g-1),
                    size = [self.g_depths[N_g-1]],
                    value = 0.0
            )
            out = tf.nn.sigmoid(tf.matmul(h, W) + b)
        return out

    def kl_divergence(self, x):
        x_mean = tf.reduce_mean(x, axis=0)
        x_var = tf.reduce_mean((x-x_mean)**2.0, axis=0)#tf.reshape(x_mean, [self.z_size, 1])
        kl_div = 0.5*tf.reduce_sum(x_mean**2.0+x_var**2.0-2.0*tf.log(x_var))
        return kl_div

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_size])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_size])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        self.e_true = self.encoder(self.X, self.phase_train, False)
        self.g_false = self.generator(self.Z, self.phase_train, False)
        self.e_false = self.encoder(self.g_false, self.phase_train, True)
        self.g_true = self.generator(self.e_true, self.phase_train, True)

        self.V2_loss = self.kl_divergence(self.e_false)-self.kl_divergence(self.e_true)

        self.reconst_x_loss = tf.reduce_sum(tf.abs(self.g_true-self.X), axis=1)
        self.reconst_z_loss = tf.reduce_sum(tf.abs(self.e_false-self.Z), axis=1)

        train_variables = tf.trainable_variables()
        self.theta_G = [v for v in train_variables if v.name.startswith('generator')]
        self.theta_E = [v for v in train_variables if v.name.startswith('encoder')]

        self.G_loss = tf.reduce_mean(self.V2_loss)+tf.reduce_mean(self.lam*self.reconst_z_loss)
        self.E_loss = -tf.reduce_mean(self.V2_loss)+tf.reduce_mean(self.mu*self.reconst_x_loss)
        self.LOSS = tf.reduce_mean(self.V2_loss)
        self.AE_loss = tf.reduce_mean(self.reconst_x_loss)

        self.saver = tf.train.Saver(max_to_keep=2500)

    def train_local(self, step, learning_rate_D, learning_rate_G, save_path):
        self.build_model()
        G_solver = tf.train.AdamOptimizer(learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)
        E_solver = tf.train.AdamOptimizer(learning_rate_D).minimize(self.E_loss,
                                                          var_list=self.theta_E)
        AE_solver = tf.train.AdamOptimizer(learning_rate_D).minimize(self.AE_loss,
                                                          var_list=self.theta_E+self.theta_G)

        E_loss_list = []
        G_loss_list = []

        mnist = input_data.read_data_sets('../../data/minst/MNIST_data', one_hot=True)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for it in range(3000):
            print 'step:{}'.format(it)
            X_mb, _ = mnist.train.next_batch(self.mb_size)

            _, AE_loss_curr= self.sess.run([AE_solver, self.AE_loss], feed_dict={
                                            self.X: X_mb, self.phase_train: True})
            print AE_loss_curr
            if it % 500 == 0 or it == step-1:
                self.saver.save(self.sess, save_path+'model.ckpt', global_step=it)

                samples = self.sess.run([self.g_true],
                              feed_dict={self.X: mnist.train.next_batch(36)[0], self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(save_path+'pre_x_{}.png'.format(it), bbox_inches='tight')
                plt.close(fig)

        for it in range(step):
            print 'step:{}'.format(it)

            X_mb, _ = mnist.train.next_batch(self.mb_size)
            
            _, G_loss_curr = self.sess.run([G_solver, self.G_loss], feed_dict={
                self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                self.phase_train: True})
            _, E_loss_curr, LOSS= self.sess.run([E_solver, self.E_loss, self.LOSS], feed_dict={
                self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                self.phase_train: True})
            """
            for i in range(3):
                _, G_loss_curr = self.sess.run([G_solver, self.G_loss], feed_dict={
                    self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                    self.phase_train: True})
            for i in range(3):
                _, E_loss_curr, LOSS= self.sess.run([E_solver, self.E_loss, self.LOSS], feed_dict={
                    self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                    self.phase_train: True})
            """

            print G_loss_curr
            print E_loss_curr
            print LOSS
            print '----'
            #_, AE_loss_curr= self.sess.run([AE_solver, self.AE_loss], feed_dict={
            #    self.X: X_mb, self.phase_train: True})
            if it % 500 == 0 or it == step-1:
                self.saver.save(self.sess, save_path+'model.ckpt', global_step=it)

                samples = self.sess.run([self.g_false],
                              feed_dict={self.Z: self.sample_Z(36, self.z_size), self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(save_path+'z_{}.png'.format(it), bbox_inches='tight')
                plt.close(fig)

                samples = self.sess.run([self.g_true],
                              feed_dict={self.X: mnist.train.next_batch(36)[0], self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(save_path+'x_{}.png'.format(it), bbox_inches='tight')
                plt.close(fig)

            #E_loss_list.append(E_loss_curr)
            #G_loss_list.append(G_loss_curr)


    def train(self, args):
        self.build_model()
        G_solver = tf.train.AdamOptimizer(args.learning_rate_G).minimize(self.G_loss,
                                                          var_list=self.theta_G)
        E_solver = tf.train.AdamOptimizer(args.learning_rate_D).minimize(self.E_loss,
                                                          var_list=self.theta_E)

        E_loss_list = []
        G_loss_list = []

        mnist = input_data.read_data_sets(args.mnist_data_path, one_hot=True)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(args.save_data_path):
            os.makedirs(args.save_data_path)

        if not os.path.exists(args.save_fig_path):
            os.makedirs(args.save_fig_path)

        for it in range(args.step):

            X_mb, _ = mnist.train.next_batch(self.mb_size)

            _, G_loss_curr, = self.sess.run([G_solver, self.G_loss],
                 feed_dict={self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                 self.phase_train: True})

            _, E_loss_curr, = self.sess.run([E_solver, self.E_loss],
                 feed_dict={self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.z_size),
                 self.phase_train: True})

            E_loss_list.append(E_loss_curr)
            G_loss_list.append(G_loss_curr)

            if it % 2000 == 0:
                self.saver.save(self.sess,
                                args.save_data_path+'model.ckpt', global_step=it)

                samples = self.sess.run([self.g_false],
                             feed_dict={self.Z: self.sample_Z(36, self.z_size),
                             self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(args.save_fig_path+'step_{}_z.png'.format(it))
                plt.close()

                samples = self.sess.run([self.g_true],
                             feed_dict={self.X: mnist.train.next_batch(36)[0],
                             self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(args.save_fig_path+'step_{}_x.png'.format(it))
                plt.close()


            if it == args.step-1:
                self.saver.save(self.sess,
                                args.save_data_path+'model.ckpt', global_step=it)

                samples = self.sess.run([self.g_false],
                             feed_dict={self.Z: self.sample_Z(36, self.z_size),
                             self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(args.save_fig_path+'step_{}_z.png'.format(it))
                plt.close()

                samples = self.sess.run([self.g_true],
                             feed_dict={self.X: mnist.train.next_batch(36)[0],
                             self.phase_train: False})

                fig = self.plot(samples[0])
                plt.savefig(args.save_fig_path+'step_{}_x.png'.format(it))
                plt.close()

                plt.plot(np.arange(len(m_loss_list)), m_loss_list, label='m_global')
                plt.legend()
                plt.savefig(args.save_fig_path+'loss.png')
                plt.close()

        np.save(args.save_data_path+'e_loss_list.npy', E_loss_list)
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
