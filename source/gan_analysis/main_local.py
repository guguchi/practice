import argparse
import os

from vanilla_gan import *
import tensorflow as tf


def main(_):

    x_size = 2
    z_size = 1
    z_range = 1.0
    d_depths = [10, 8, 6, 4]
    g_depths = [10, 8, 6, 4]
    mb_size = 128
    phase = 'test'#'train'#

    with tf.Session() as sess:

        if phase == 'train':
            model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
            model.train(step = 30, learning_rate_D = 0.001, learning_rate_G = 0.001,
                        num_cluster = 8, scale = 2, std = 0.2)
        else:
            model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
            model.restore()


if __name__ == '__main__':
    tf.app.run()
