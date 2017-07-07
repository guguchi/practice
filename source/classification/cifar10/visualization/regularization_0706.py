import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


path = '/home/ishii/Desktop/research/practice/data/classification/cifar10/vanilla/0704/batch_50_alpha_0.0005/'
#cross_entropy_list = np.load(path + 'cross_entropy.npy')
test_accuracy_list = np.load(path + 'test_accuracy.npy')
train_accuracy_list = np.load(path + 'train_accuracy_batch.npy')
#plt.plot(np.arange(len(train_accuracy_list[0])), np.mean(train_accuracy_list, axis=0), alpha=0.5, label = 'batch: {}'.format(batch_size))
_range = np.where((test_accuracy_list[0] != 0))[0]
plt.plot(_range, np.mean(test_accuracy_list[:, _range], axis=0), alpha=0.5, label = 'vanilla')

drop_ratio_list = [0.85, 0.9, 0.95, 0.99]

for drop_ratio in drop_ratio_list:
    path = '/home/ishii/Desktop/research/practice/data/classification/cifar10/dropout/0704/drops_{}_batch_50_alpha_0.0005/'.format(drop_ratio)
    #cross_entropy_list = np.load(path + 'cross_entropy.npy')
    test_accuracy_list = np.load(path + 'test_accuracy.npy')
    train_accuracy_list = np.load(path + 'train_accuracy_batch.npy')
    #plt.plot(np.arange(len(train_accuracy_list[0])), np.mean(train_accuracy_list, axis=0), alpha=0.5, label = 'batch: {}'.format(batch_size))
    _range = np.where((test_accuracy_list[0] != 0))[0]
    plt.plot(_range, np.mean(test_accuracy_list[:, _range], axis=0), alpha=0.5, label = 'drop: {}'.format(drop_ratio))

weight_lam = [0.01, 0.05, 0.1, 0.5, 1.0]
for lam in weight_lam:
    path = '/home/ishii/Desktop/research/practice/data/classification/cifar10/weight_decay/0704/lam_{}_batch_50_alpha_0.0005/'.format(lam)
    #cross_entropy_list = np.load(path + 'cross_entropy.npy')
    test_accuracy_list = np.load(path + 'test_accuracy.npy')
    train_accuracy_list = np.load(path + 'train_accuracy_batch.npy')
    #plt.plot(np.arange(len(train_accuracy_list[0])), np.mean(train_accuracy_list, axis=0), alpha=0.5, label = 'batch: {}'.format(batch_size))
    _range = np.where((test_accuracy_list[0] != 0))[0]
    plt.plot(_range, np.mean(test_accuracy_list[:, _range], axis=0), alpha=0.5, label = 'weight_decay: {}'.format(lam))


spectral_lam = [10.0, 20.0, 30.0, 40.0, 50.0]
for lam in spectral_lam:
    path = '/home/ishii/Desktop/research/practice/data/classification/cifar10/spectral_norm/0704/lam_{}_sp_iter_3_batch_50_alpha_0.0005/'.format(lam)
    #cross_entropy_list = np.load(path + 'cross_entropy.npy')
    test_accuracy_list_0 = np.load(path + 'test_accuracy_0.npy')
    train_accuracy_list_0 = np.load(path + 'train_accuracy_batch_0.npy')
    test_accuracy_list_1 = np.load(path + 'test_accuracy_1.npy')
    train_accuracy_list_1 = np.load(path + 'train_accuracy_batch_1.npy')
    test_accuracy_list_2 = np.load(path + 'test_accuracy_2.npy')
    train_accuracy_list_2 = np.load(path + 'train_accuracy_batch_2.npy')
    #plt.plot(np.arange(len(train_accuracy_list[0])), np.mean(train_accuracy_list, axis=0), alpha=0.5, label = 'batch: {}'.format(batch_size))
    test_accuracy_list = (test_accuracy_list_0 + test_accuracy_list_1 + test_accuracy_list_2) / 3.0
    _range = np.where((test_accuracy_list != 0))[0]
    plt.plot(_range, test_accuracy_list[ _range], alpha=0.5, label = 'spectral_norm: {}'.format(lam))

plt.legend()
plt.savefig('regularization_0706_test_accuracy.png')#vanilla_0723_train_accuracy#vanilla_0723_test_accuracy
plt.show()
