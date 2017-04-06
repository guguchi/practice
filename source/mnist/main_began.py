import argparse
import os

from be_gan import *
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
parser.add_argument('gamma', type=float)
parser.add_argument('h_size', type=int)
parser.add_argument('_lambda', type=float)

args = parser.parse_args()

def main(_):

    x_size = 784
    d_depths = [x_size, 64*4*4, 64*5*5, 64*4*4, 64*3*3, args.h_size,
                64*3*3, 64*4*4, 64*5*5, 64*4*4, x_size]
    g_depths = [args.z_size, 64*3*3, 64*4*4, 64*5*5, 64*4*4, x_size]
    mb_size = 64

    with tf.Session() as sess:

        model = BEGAN(sess, x_size, args.z_size, args.z_range, d_depths,
                      g_depths, mb_size, args.gamma)
        model.train(args)


if __name__ == '__main__':
    tf.app.run()
