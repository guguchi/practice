import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vanilla_gan import *
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('save_path', type=str)
parser.add_argument('data_path', type=str)
parser.add_argument('step', type=int)
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
    z_range = 1.0
    d_depths = [25, 30, 30, 25]
    g_depths = [25, 20, 20, 10]
    mb_size = 128
    species = 'test'

    with tf.Session() as sess:

        if species == 'test':
            model = VanillaGAN(sess, x_size, args.z_size, z_range, d_depths, g_depths, mb_size)
            model.restore(args)



if __name__ == '__main__':
    tf.app.run()
