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

for learning_rate_D in learning_rate_D_list:
    for learning_rate_G in learning_rate_G_list:
        for num_cluster in num_cluster_list:
            for std in std_list:
                
