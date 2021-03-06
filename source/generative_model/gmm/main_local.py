import os
from vanilla_gan import *
from wasserstein_gan import *
import tensorflow as tf


def main(_):

    x_size = 2
    z_size = 25
    z_range = 10.0
    d_depths = [x_size, 50, 50, 50, 50, 1]
    g_depths = [z_size, 50, 50, 50, 50, x_size]
    mb_size = 128
    phase = 'train'#'test'#
    species =  'wesserstein_gan'#'vanilla_gan'#

    learning_rate_D = 0.00001
    learning_rate_G = 0.00001
    num_cluster = 8
    scale = 2
    std = 0.2
    save_path = '../../data/gan_analysis/20170330/d_{}_g_{}_cl_{}_std_0.2_z_{}/'.format(
                          learning_rate_D, learning_rate_G, num_cluster, z_size)
    step = 10000
    sample_size = 10000

    with tf.Session() as sess:

        if species == 'vanilla_gan':
            if phase == 'train':
                model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
                model.train_local(step, learning_rate_D, learning_rate_G, num_cluster,
                                  scale, std, save_path)

        if species == 'wesserstein_gan':
            if phase == 'train':
                clip_value = 0.01
                g_iteration = 3000
                d_iteration = 100
                model = WassersteinGAN(sess, x_size, z_size, z_range, d_depths, g_depths,
                                       mb_size, clip_value)
                model.train_local(g_iteration, d_iteration, learning_rate_D, learning_rate_G,
                            num_cluster, scale, std, sample_size, save_path)

if __name__ == '__main__':
    tf.app.run()
