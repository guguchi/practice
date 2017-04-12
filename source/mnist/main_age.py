import argparse
import os

from age import *
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
parser.add_argument('lam', type=float)
parser.add_argument('mu', type=float)

args = parser.parse_args()

def main(_):

    x_size = 784
    e_depths = [x_size, 64*4*4, 64*5*5, 64*4*4, 64*3*3, args.z_size]
    g_depths = [args.z_size, 64*3*3, 64*4*4, 64*5*5, 64*4*4, x_size]
    mb_size = 64

    with tf.Session() as sess:

        model = AGE(sess, x_size, args.z_size, args.z_range, e_depths, g_depths, mb_size,
                    args.lam, args.mu)
        model.train(args)


if __name__ == '__main__':
    tf.app.run()
