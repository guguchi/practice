# coding: utf-8
#!/usr/bin/python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from slurm import slurm_tools

g_iteration = 100000
d_iteration = 50
learning_rate_D_list = [0.00005, 0.000025, 0.00001]
learning_rate_G_list = [0.00005, 0.000025, 0.00001]
num_cluster_list = [8]
scale = 2
std = 0.2
z_size_list = [50, 20, 10, 2, 1]
clip_value_list = [0.01, 0.05, 0.1, 1.0]
date = '20170401/'
species = 'wasserstein_gan/'
save_data_root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis/'+date+species+'file/'
save_fig_root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis/'+date+species+'fig/'
sample_size = 50000

for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for num_cluster in num_cluster_list:
            for z_size in z_size_list:
                for clip_value in clip_value_list:

                    save_data_path = save_data_root+'d_{}_g_{}_cl_{}_std_{}_z_{}_clip_{}/'.format(
                      learning_rate_D, learning_rate_G, num_cluster, std, z_size, clip_value)

                    save_fig_path = save_fig_root+'d_{}_g_{}_cl_{}_std_{}_z_{}_clip_{}/'.format(
                      learning_rate_D, learning_rate_G, num_cluster, std, z_size, clip_value)

                    SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/gan_analysis/main_wasserstein.py {} {} {} {} {} {} {} {} {} {} {} {}'.format(
                                save_data_path, save_fig_path, g_iteration, d_iteration,
                                learning_rate_D, learning_rate_G,
                                num_cluster, scale, std, z_size, clip_value, sample_size)]

                    res, success=slurm_tools.slurm_submit(SLURM_commands, mem=16000, gres='gpu:1')

                    print success
                    print res
