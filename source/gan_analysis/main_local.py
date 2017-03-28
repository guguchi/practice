import os
from vanilla_gan import *
import tensorflow as tf


def main(_):

    x_size = 2
    z_size = 100
    z_range = 1.0
    d_depths = [50, 50, 50, 50]
    g_depths = [50, 50, 50, 50]
    mb_size = 128
    phase = 'test'#'train'#

    learning_rate_D = 0.0004
    learning_rate_G = 0.0004
    num_cluster = 8
    path = '/20170327/z_size/d_{}_g_{}_cl_{}_std_0.2_z_{}/'.format(
                          learning_rate_D, learning_rate_G, num_cluster, z_size)
    step = 10000
    sample_size = 900000#<1000000

    with tf.Session() as sess:

        if phase == 'train':
            model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
            model.train(step = 30, learning_rate_D = 0.001, learning_rate_G = 0.001,
                        num_cluster = 8, scale = 2, std = 0.2)
        else:
            model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
            model.restore_local(path, step, sample_size, num_cluster = 8, scale = 2, std = 0.2)





if __name__ == '__main__':
    tf.app.run()
