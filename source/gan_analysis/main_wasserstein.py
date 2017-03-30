import argparse
import os

from wasserstein_gan import *
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('save_data_path', type=str)
parser.add_argument('save_fig_path', type=str)
parser.add_argument('g_iteration', type=int)
parser.add_argument('d_iteration', type=int)
parser.add_argument('learning_rate_D', type=float)
parser.add_argument('learning_rate_G', type=float)
parser.add_argument('num_cluster', type=int)
parser.add_argument('scale', type=float)
parser.add_argument('std', type=float)
parser.add_argument('z_size', type=int)
parser.add_argument('sample_size', type=int)

args = parser.parse_args()

def main(_):

    x_size = 2
    z_range = 2.0
    d_depths = [10, 10, 10, 10]
    g_depths = [10, 10, 10, 10]
    mb_size = 128
    clip_value = 0.01

    with tf.Session() as sess:

        model = WassersteinGAN(sess, x_size, args.z_size, z_range, d_depths,
                                   g_depths, mb_size, clip_value)
        model.train(args)

if __name__ == '__main__':
    tf.app.run()
