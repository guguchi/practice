# coding: utf-8
#!/usr/bin/python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from slurm import slurm_tools

step = 100000
learning_rate_D_list = [0.0001, 0.0002]
learning_rate_G_list = [0.0001, 0.0002]
z_size = 10
z_range_list = [1.0, 10.0, 100.0]
lam_list = [500, 1000, 2000]
mu_list = [10, 100, 1000]
_lambda = 0.001

date = '20170413/'
species = 'age/'
save_data_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'file/'
save_fig_root = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/'+date+species+'fig/'
mnist_data_path = '/home/yamaguchi-s/Desktop/Research/practice/data/mnist/MNIST_data'


for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for z_range in z_range_list:
            for lam in lam_list:
                for mu in mu_list:
                    save_data_path = save_data_root+'d_{}_g_{}_z_{}_lam_{}_mu_{}/'.format(
                          learning_rate_D, learning_rate_G, z_range, lam, mu)

                    save_fig_path = save_fig_root+'d_{}_g_{}_z_{}_lam_{}_mu_{}/'.format(
                          learning_rate_D, learning_rate_G, z_range, lam, mu)

                    SLURM_commands=['python /home/yamaguchi-s/Desktop/Research/practice/source/mnist/main_age.py {} {} {} {} {} {} {} {} {} {}'.format(
                                    save_data_path, save_fig_path, mnist_data_path,
                                    step, learning_rate_D, learning_rate_G,
                                    z_size, z_range, lam, mu)]

                    res, success=slurm_tools.slurm_submit(SLURM_commands, mem=25000, gres='gpu:1')

                    print success
                    print res
