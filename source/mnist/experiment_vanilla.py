# coding: utf-8
#!/usr/bin/python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from slurm import slurm_tools

step = 100000
learning_rate_D_list = [0.0002]#, 0.0001, 0.00005]
learning_rate_G_list = [0.0002]#, 0.0001, 0.00005]
z_size_list = [200]#, 100, 50, 10, 5]
z_range_list = [100.0]
date = '20170403/'
species = 'vanilla_gan/'
save_data_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'file/'
save_fig_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'fig/'
mnist_data_path = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/MNIST_data'

for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for z_size in z_size_list:
            for z_range in z_range_list:
                save_data_path = save_data_root+'d_{}_g_{}_sz{}_rng_{}/'.format(
                      learning_rate_D, learning_rate_G, z_size, z_range)

                save_fig_path = save_fig_root+'d_{}_g_{}_sz{}_rng_{}/'.format(
                      learning_rate_D, learning_rate_G, z_size, z_range)

                SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/mnist/main_vanilla.py {} {} {} {} {} {} {} {} '.format(
                                save_data_path, save_fig_path, mnist_data_path,
                                step, learning_rate_D, learning_rate_G,
                                z_size, z_range)]

                res, success=slurm_tools.slurm_submit(SLURM_commands, mem=10000, gres='gpu:1')

                print success
                print res
