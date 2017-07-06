import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


batch_size_list = [5, 10, 25, 50, 100, 200, 500, 1000]

for i, batch_size in enumerate(batch_size_list):
    path = '/home/ishii/Desktop/research/practice/data/classification/cifar10/vanilla/0703/batch_{}_alpha_0.0005/'.format(batch_size)
    #cross_entropy_list = np.load(path + 'cross_entropy.npy')
    test_accuracy_list = np.load(path + 'test_accuracy.npy')
    train_accuracy_list = np.load(path + 'train_accuracy.npy')

    plt.plot(np.arange(len(train_accuracy_list[0])), np.mean(train_accuracy_list, axis=0), alpha=0.5, label = 'batch: {}'.format(batch_size))
    _range = np.where((test_accuracy_list[0] != 0))[0]
    #plt.plot(_range, np.mean(test_accuracy_list[:, _range], axis=0), alpha=0.5, label = 'batch: {}'.format(batch_size))

plt.legend()
plt.savefig('vanilla_0723_train_accuracy.png')#vanilla_0723_train_accuracy#vanilla_0723_test_accuracy
plt.show()
