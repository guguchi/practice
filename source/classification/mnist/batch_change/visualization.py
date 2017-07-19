import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import gridspec

activation_list = ['sigmoid', 'lrelu']#[1:2]
layer_size_list = [3, 4, 5]#[1:2]
batch_size_list = [5, 10, 25, 50][:1]
alpha = 0.01

thresh = 0.1 ** 5

# singular value distribution
for batch_size in batch_size_list:
    for activation in activation_list:
        for layer_size in layer_size_list:

            base_path = '/home/ishii/Desktop/research/practice/data/classification/mnist/batch_change/{}_layer_{}_batch_{}_alpha_{}/'.format(
                activation, layer_size, batch_size, alpha)

            test_singular_value = np.load(base_path + 'test_singular_value.npy')[0, :200]
            print np.shape(test_singular_value)
            test_singular_value[np.abs(test_singular_value) <= thresh] = 1.0
            entropy = np.mean(np.log(np.abs(test_singular_value)), axis=(1,2))
            print entropy

            plt.plot(np.arange(len(entropy)), -entropy,  label='{}_layer_{}_batch_{}_alpha_{}/'.format(
                activation, layer_size, batch_size, alpha))
            #plt.show()

            """
            L = 199

            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 6)
            ax1 = plt.subplot(gs[0, :5])
            for i in range(25):
                A1 = np.sort(test_singular_value[L, i])
                ax1.plot(np.arange(len(A1)), A1, label='{}'.format(i))
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 6)
            ax1 = plt.subplot(gs[0, :5])
            for i in range(25):
                A1 = np.sort(test_singular_value[L, i])[::-1]
                A1[A1 <= 0.1 ** 5] = 1.0
                log_A1 = np.log(A1)
                entropy = np.cumsum(log_A1)

                ax1.plot(np.arange(len(entropy)), -entropy, label='{}'.format(i), )
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            ax1.set_yscale('log')
            plt.show()
            """
            """
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 6)
            ax1 = plt.subplot(gs[0, :5])
            for i in range(25):
                A1 = np.sort(test_singular_value[0, i])
                ax1.plot(np.arange(len(A1)), A1, label='{}'.format(i))
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 6)
            ax1 = plt.subplot(gs[0, :5])
            for i in range(25):
                A1 = np.sort(test_singular_value[50, i])
                ax1.plot(np.arange(len(A1)), A1, label='{}'.format(i))
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 6)
            ax1 = plt.subplot(gs[0, :5])
            for i in range(25):
                A1 = np.sort(test_singular_value[100, i])
                ax1.plot(np.arange(len(A1)), A1, label='{}'.format(i))
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 6)
            ax1 = plt.subplot(gs[0, :5])
            for i in range(25):
                A1 = np.sort(test_singular_value[150, i])
                ax1.plot(np.arange(len(A1)), A1, label='{}'.format(i))
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 6)
            ax1 = plt.subplot(gs[0, :5])
            for i in range(25):
                A1 = np.sort(test_singular_value[199, i])
                ax1.plot(np.arange(len(A1)), A1, label='{}'.format(i))
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.show()
            """
plt.legend()
#plt.ylim(3.6, 4.0)#plt.ylim(6.0, 6.1)
plt.show()

"""
### test_accuracy
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 6)
ax1 = plt.subplot(gs[0, :5])
for batch_size in batch_size_list:
    for activation in activation_list:
        for layer_size in layer_size_list:

            base_path = '/home/ishii/Desktop/research/practice/data/classification/mnist/batch_change/{}_layer_{}_batch_{}_alpha_{}/'.format(
                activation, layer_size, batch_size, alpha)

            test_accuracy = np.mean(np.load(base_path + 'test_accuracy.npy'), axis=0)
            _range = np.where((test_accuracy != 0))[0]

            ax1.plot(_range, test_accuracy[_range], alpha=0.5, label='{}_layer_{}_batch_{}_alpha_{}/'.format(
                activation, layer_size, batch_size, alpha))
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax1.set_ylim(0.96, 0.99)
plt.show()
"""
