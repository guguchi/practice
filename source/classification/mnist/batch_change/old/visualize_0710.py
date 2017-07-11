import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

activation = 'relu_up'#'relu'#'sigmoid'#'lrelu'#
layer_size = 5
alpha = 0.01#0.0005#

batch_size_list = [5, 10, 25, 50]

plt.subplot(211)
for batch in batch_size_list:
    path = '/home/ishii/Desktop/research/practice/data/classification/mnist/batch_change/{}_layer_{}_batch_{}_alpha_{}/'.format(activation, layer_size, batch, alpha)
    cross_entropy_list = np.load(path + 'cross_entropy.npy')
    train_accuracy_list = np.load(path + 'train_accuracy_batch.npy')
    test_accuracy_list = np.load(path + 'test_accuracy.npy')
    entropy_list = np.load(path + 'entropy.npy')
    #train_entropy_list = np.load(path + 'entropy_train.npy')

    _range = np.where((test_accuracy_list[0] != 0))[0]
    plt.plot(_range, -np.mean(entropy_list[:, _range], axis=0), alpha=0.5, label = 'entropy: {}'.format(batch))
    #plt.plot(_range, -np.mean(train_entropy_list[:, _range], axis=0), alpha=0.5, label = 'train entropy: {}'.format(batch))
plt.legend()
#plt.ylim(5.9, 6.1)

plt.subplot(212)
for batch in batch_size_list:
    path = '/home/ishii/Desktop/research/practice/data/classification/mnist/batch_change/{}_layer_{}_batch_{}_alpha_{}/'.format(activation, layer_size, batch, alpha)
    cross_entropy_list = np.load(path + 'cross_entropy.npy')
    train_accuracy_list = np.load(path + 'train_accuracy_batch.npy')
    test_accuracy_list = np.load(path + 'test_accuracy.npy')
    entropy_list = np.load(path + 'entropy.npy')
    #train_entropy_list = np.load(path + 'entropy_train.npy')

    _range = np.where((test_accuracy_list[0] != 0))[0]
    plt.plot(_range, (-np.mean(entropy_list[:, _range], axis=0) + np.mean(entropy_list[:, _range[0]])) / layer_size, alpha=0.5, label = 'entropy: {}'.format(batch))
    #plt.plot(_range, (-np.mean(train_entropy_list[:, _range], axis=0) + np.mean(train_entropy_list[:, _range[0]])) / layer_size, alpha=0.5, label = 'train entropy: {}'.format(batch))

plt.title('layer-average increment of entropy ')
#plt.ylim(0.6, 0.85)
plt.legend()
plt.tight_layout()
plt.savefig('mnist_{}_negative_entropy_batch_{}_alpha_{}.png'.format(activation, batch, alpha))
plt.show()


plt.subplot(211)
for batch in batch_size_list:
    path = '/home/ishii/Desktop/research/practice/data/classification/mnist/batch_change/{}_layer_{}_batch_{}_alpha_{}/'.format(activation, layer_size, batch, alpha)
    cross_entropy_list = np.load(path + 'cross_entropy.npy')
    train_accuracy_list = np.load(path + 'train_accuracy_batch.npy')
    test_accuracy_list = np.load(path + 'test_accuracy.npy')
    entropy_list = np.load(path + 'entropy.npy')

    _range = np.where((test_accuracy_list[0] != 0))[0]
    plt.plot(_range, np.mean(test_accuracy_list[:, _range], axis=0), alpha=0.5, label = 'test: {}'.format(batch))

plt.legend()

plt.subplot(212)
for batch in batch_size_list:
    path = '/home/ishii/Desktop/research/practice/data/classification/mnist/batch_change/{}_layer_{}_batch_{}_alpha_{}/'.format(activation, layer_size, batch, alpha)
    cross_entropy_list = np.load(path + 'cross_entropy.npy')
    train_accuracy_list = np.load(path + 'train_accuracy_batch.npy')
    test_accuracy_list = np.load(path + 'test_accuracy.npy')
    entropy_list = np.load(path + 'entropy.npy')

    _range = np.where((test_accuracy_list[0] != 0))[0]
    plt.plot(_range, np.mean(test_accuracy_list[:, _range], axis=0), alpha=0.5, label = 'test: {}'.format(batch))

plt.ylim(0.94, 0.99)
plt.legend()#
plt.savefig('mnist_{}_test_batch_{}_alpha_{}.png'.format(activation, batch, alpha))
plt.show()
