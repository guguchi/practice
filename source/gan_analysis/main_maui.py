import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vanilla_gan import *
import tensorflow as tf


def main(_):

    x_size = 2
    z_range = 1.0
    d_depths = [25, 30, 30, 25]
    g_depths = [25, 20, 20, 10]
    mb_size = 128
    root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis'

    #z_size
    learning_rate_D_list = [0.0001, 0.00005, 0.00001]
    learning_rate_G_list = [0.0001, 0.00005, 0.00001]
    num_cluster = 8
    scale = 2
    std = 0.2
    z_size_list = [100, 50, 10, 2, 1]
    species = 'z_zize'

    sample_size = 100000

    for learning_rate_D in learning_rate_D_list:
        for learning_rate_G in learning_rate_G_list:
            for z_size in z_size_list:
                path = '/20170329/'+species+'/d_{}_g_{}_cl_{}_std_0.2_z_{}/'.format(
                                      learning_rate_D, learning_rate_G, num_cluster, z_size)

                D_loss_list = np.load(root+path+'d_loss_list.npy')
                G_loss_list = np.load(root+path+'g_loss_list.npy')
                plt.plot(np.arange(len(D_loss_list)), D_loss_list, label='D')
                plt.plot(np.arange(len(G_loss_list)), G_loss_list, label='G')
                plt.legend()
                plt.savefig(root+path+'loss.png')
                plt.close()

                for step in range(0, 300000, 1000):
                    sess = tf.Session()

                    model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
                    model.restore_maui(root+path, step, sample_size, num_cluster, scale, std)

                    sess.close()

    #num_cluster
    z_size = 25
    num_cluster_list = [8, 4, 2, 1]

    for learning_rate_D in learning_rate_D_list:
        for learning_rate_G in learning_rate_G_list:
            for num_cluster in num_cluster_list:
                path = '/20170329/'+species+'/d_{}_g_{}_cl_{}_std_0.2_z_{}/'.format(
                                      learning_rate_D, learning_rate_G, num_cluster, z_size)

                D_loss_list = np.load(root+path+'d_loss_list.npy')
                G_loss_list = np.load(root+path+'g_loss_list.npy')
                plt.plot(np.arange(len(D_loss_list)), D_loss_list, label='D')
                plt.plot(np.arange(len(G_loss_list)), G_loss_list, label='G')
                plt.legend()
                plt.savefig(root+path+'loss.png')
                plt.close()

                for step in range(0, 300000, 1000):
                    sess = tf.Session()

                    model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
                    model.restore_maui(root+path, step, sample_size, num_cluster, scale, std)

                    sess.close()


if __name__ == '__main__':
    tf.app.run()
