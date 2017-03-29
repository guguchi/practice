# coding: utf-8
#!/usr/bin/python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from slurm import slurm_tools

step = 300000
learning_rate_D_list = [0.0001, 0.00005, 0.00001]
learning_rate_G_list = [0.0001, 0.00005, 0.00001]
num_cluster_list = [8]#[8, 4, 2, 1]
scale = 2
std_list = [0.2]#[0.5, 0.4, 0.3, 0.2, 0.1]
z_size_list = [100, 50, 10, 2, 1]#[500, 100, 50, 10, 2, 1]#,50 10
date = '20170329/'
concept = 'z_size/'
save_data_root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis/'+date+'file/'+concept
save_fig_root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis/'+date+'fig/'+concept
sample_size = 25000

for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for num_cluster in num_cluster_list:
            for std in std_list:
                for z_size in z_size_list:

                    save_data_path = save_data_root+'d_{}_g_{}_cl_{}_std_{}_z_{}/'.format(
                      learning_rate_D, learning_rate_G, num_cluster, std, z_size)

                    save_fig_path = save_fig_root+'d_{}_g_{}_cl_{}_std_{}_z_{}/'.format(
                      learning_rate_D, learning_rate_G, num_cluster, std, z_size)

                    SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/gan_analysis/main.py {} {} {} {} {} {} {} {} {} {}'.format(
                                save_data_path, save_fig_path, step, learning_rate_D, learning_rate_G,
                                num_cluster, scale, std, z_size, sample_size)]

            	    res, success=slurm_tools.slurm_submit(SLURM_commands, mem=16000, gres='gpu:1')

                    print success
                    print res

step = 300000
learning_rate_D_list = [0.0001, 0.00005, 0.00001]
learning_rate_G_list = [0.0001, 0.00005, 0.00001]
num_cluster_list = [8, 4, 2, 1]#[8, 4, 2, 1]
scale = 2
std_list = [0.2]#[0.5, 0.4, 0.3, 0.2, 0.1]
z_size_list = [25]#[500, 100, 50, 10, 2, 1]#,50 10
date = '20170329/'
concept = 'num_cluster/'
save_data_root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis/'+date+'file/'+concept
save_fig_root = '/home/yamaguchi-s/Desktop/Research/practice/data/gan_analysis/'+date+'fig/'+concept
sample_size = 25000

for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for num_cluster in num_cluster_list:
            for std in std_list:
                for z_size in z_size_list:

                    save_data_path = save_data_root+'d_{}_g_{}_cl_{}_std_{}_z_{}/'.format(
                      learning_rate_D, learning_rate_G, num_cluster, std, z_size)

                    save_fig_path = save_fig_root+'d_{}_g_{}_cl_{}_std_{}_z_{}/'.format(
                      learning_rate_D, learning_rate_G, num_cluster, std, z_size)

                    SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/gan_analysis/main.py {} {} {} {} {} {} {} {}'.format(
                                save_data_path, save_fig_path, step, learning_rate_D, learning_rate_G,
                                num_cluster, scale, std, z_size, sample_size)]

            	    res, success=slurm_tools.slurm_submit(SLURM_commands, mem=16000, gres='gpu:1')

                    print success
                    print res
