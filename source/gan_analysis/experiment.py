# coding: utf-8
#!/usr/bin/python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from slurm import slurm_tools

step = 1000000
learning_rate_D_list = [0.001, 0.00075, 0.0005, 0.00025, 0.0001]
learning_rate_G_list = [0.001, 0.00075, 0.0005, 0.00025, 0.0001]
num_cluster_list = [8, 4, 2, 1]
scale = 2
std_list = [0.5, 0.4, 0.3, 0.2, 0.1]
z_size = [500, 100, 50, 10, 2, 1]
save_root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis/20170324/'

for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for num_cluster in num_cluster_list:
            for std in std_list:
                save_path = save_root+'d_{}_g_{}_cl_{}_std{}/'.format(
                             learning_rate_D, learning_rate_G, num_cluster, std)
                SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/gan_analysis/main.py {} {} {} {} {} {} {} {}'.format(
                                save_path, step, learning_rate_D, learning_rate_G,
                                num_cluster, scale, std, z_size)]
            	res, success=slurm_tools.slurm_submit(SLURM_commands, mem=16000, gres='gpu:1')

                print "job number: "+res
