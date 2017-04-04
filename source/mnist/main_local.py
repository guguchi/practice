import os
from vanilla_gan import *
from ls_gan import *
from w_gan import *
import tensorflow as tf


def main(_):

    x_size = 784
    z_size = 25
    z_range = 10.0
    d_depths = [x_size, 250, 250, 250, 250, 1]
    g_depths = [z_size, 250, 250, 250, 250, x_size]
    mb_size = 128
    phase = 'train'#'test'#
    species =  'w_gan'#'ls_gan'#'vanilla_gan'#

    learning_rate_D = 0.00001
    learning_rate_G = 0.00001
    save_path = '../../data/mnist/20170405/'+species+'/d_{}_g_{}/'.format(
                          learning_rate_D, learning_rate_G)
    step = 10000
    g_iteration = 10000
    d_iteration = 5
    clip_value = 0.01

    with tf.Session() as sess:

        if species == 'vanilla_gan':
            if phase == 'train':
                model = VanillaGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
                model.train_local(step, learning_rate_D, learning_rate_G, save_path)

        if species == 'ls_gan':
            if phase == 'train':
                model = LeastSquareGAN(sess, x_size, z_size, z_range, d_depths, g_depths, mb_size)
                model.train_local(step, learning_rate_D, learning_rate_G, save_path)

        if species == 'w_gan':
            if phase == 'train':
                model = WassersteinGAN(sess, x_size, z_size, z_range, d_depths,
                                       g_depths, mb_size, clip_value)
                model.train_local(g_iteration, d_iteration, learning_rate_D,
                                  learning_rate_G, save_path)


if __name__ == '__main__':
    tf.app.run()
