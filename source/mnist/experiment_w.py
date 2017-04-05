# coding: utf-8
#!/usr/bin/python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from slurm import slurm_tools

g_iteration = 100000
d_iteration = 3
learning_rate_D_list = [0.0001, 0.00005]
learning_rate_G_list = [0.0001, 0.00005]
z_size_list = [200, 100, 50, 10, 5, 2]
z_range = 200.0
clip_value_list = [0.005]
date = '20170406/'
species = 'w_gan/'
save_data_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'file/'
save_fig_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'fig/'
mnist_data_path = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/MNIST_data'


for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for z_size in z_size_list:
            for clip_value in clip_value_list:
                save_data_path = save_data_root+'d_{}_g_{}_sz{}_cl_{}/'.format(
                      learning_rate_D, learning_rate_G, z_size, clip_value)

                save_fig_path = save_fig_root+'d_{}_g_{}_sz{}_cl_{}/'.format(
                      learning_rate_D, learning_rate_G, z_size, clip_value)

                SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/mnist/main_w.py {} {} {} {} {} {} {} {} {} {}'.format(
                                save_data_path, save_fig_path, mnist_data_path,
                                g_iteration, d_iteration, learning_rate_D, learning_rate_G,
                                z_size, z_range, clip_value)]

                res, success=slurm_tools.slurm_submit(SLURM_commands, mem=25000, gres='gpu:1')

                print success
                print res
