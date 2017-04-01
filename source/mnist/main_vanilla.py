import argparse
import os

from vanilla_gan import *
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('save_data_path', type=str)
parser.add_argument('save_fig_path', type=str)
parser.add_argument('mnist_data_path', type=str)
parser.add_argument('step', type=int)
parser.add_argument('learning_rate_D', type=float)
parser.add_argument('learning_rate_G', type=float)
parser.add_argument('z_size', type=int)
parser.add_argument('z_range', type=float)

args = parser.parse_args()

def main(_):

    x_size = 784
    d_depths = [x_size, 64*4*4, 64*5*5, 64*4*4, 64*3*3, 1]
    g_depths = [args.z_size, 64*2*2, 64*3*3, 64*4*4, 64*5*5, x_size]
    mb_size = 128

    with tf.Session() as sess:

        model = VanillaGAN(sess, x_size, args.z_size, arhs.z_range, d_depths,
                           g_depths, mb_size)
        model.train(args)


if __name__ == '__main__':
    tf.app.run()
