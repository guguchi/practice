import argparse
import os

from vanilla_gan import *
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('save_path', type=str)
parser.add_argument('step', type=int)
parser.add_argument('learning_rate_D', type=float)
parser.add_argument('learning_rate_G', type=float)
parser.add_argument('num_cluster', type=int)
parser.add_argument('scale', type=float)
parser.add_argument('std', type=float)
parser.add_argument('z_size', type=int)

args = parser.parse_args()

def main(_):

    x_size = 2
    z_range = 1.0
    d_depths = [50, 50, 50, 50]
    g_depths = [50, 50, 50, 50]
    mb_size = 128

    with tf.Session() as sess:

        if species = 'vanilla_gan':
            model = VanillaGAN(sess, x_size, args.z_size, z_range, d_depths, g_depths, mb_size)
            model.train(args)


if __name__ == '__main__':
    tf.app.run()
