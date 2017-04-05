# coding: utf-8
#!/usr/bin/python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from slurm import slurm_tools

step = 100000
learning_rate_D_list = [0.0001, 0.00005]
learning_rate_G_list = [0.0001, 0.00005]
z_size = 100
z_range = 100.0
gamma_list = [0.3, 0.6, 0.9]
h_size_list = [5, 20]
_lambda = 0.001

date = '20170406/'
species = 'eb_gan/'
save_data_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'file/'
save_fig_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'fig/'
mnist_data_path = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/MNIST_data'


for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for gamma in gamma_list:
            for h_size in h_size_list:
                save_data_path = save_data_root+'d_{}_g_{}_gm_{}_h_{}/'.format(
                      learning_rate_D, learning_rate_G, gamma, h_size)

                save_fig_path = save_fig_root+'d_{}_g_{}_gm_{}_h_{}/'.format(
                      learning_rate_D, learning_rate_G, gamma, h_size)

                SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/mnist/main_began.py {} {} {} {} {} {} {} {} {} {} {}'.format(
                                save_data_path, save_fig_path, mnist_data_path,
                                step, learning_rate_D, learning_rate_G,
                                z_size, z_range, gamma, h_size, _lambda)]

                res, success=slurm_tools.slurm_submit(SLURM_commands, mem=25000, gres='gpu:1')

                print success
                print res
