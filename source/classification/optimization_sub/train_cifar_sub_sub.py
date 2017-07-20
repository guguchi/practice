# coding: utf-8

#from __future__ import print_function
import argparse
import sys

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG

from six.moves import xrange
import os, collections, six, math, cupy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from chainer import optimizers, iterators, cuda, Variable, initializers


def compute_classification_accuracy(model, x, t):
	xp = model.xp
	batches = xp.split(x, len(x) // 100)
	scores = None
	for batch in batches:
		p = F.softmax(model(batch, apply_softmax=False)).data
		scores = p if scores is None else xp.concatenate((scores, p), axis=0)
	return float(F.accuracy(scores, Variable(t)).data)


def cache_weights(model):cached_weights=[]forparaminmodel.params():withcuda.get_device(param.data):xp=cuda.get_array_module(param.data)cached_weights.append(xp.copy(param.data))return cached_weights


def restore_weights(model,cached_weights):for(i,param)inenumerate(model.params()):withcuda.get_device(param.data):param.data = cached_weights[i]


def arange_cifar(cifar_train, cifar_test):
	train_data, train_label = [], []
	test_data, test_label = [], []
	for data in cifar_train:
		train_data.append(data[0])
		train_label.append(data[1])
	for data in cifar_test:
		test_data.append(data[0])
		test_label.append(data[1])
	train_data = np.asanyarray(train_data, dtype=np.float32)
	test_data = np.asanyarray(test_data, dtype=np.float32)
	train_data = (train_data - np.mean(train_data)) / np.std(train_data)
	test_data = (test_data - np.mean(test_data)) / np.std(test_data)
	return (train_data, np.asanyarray(train_label, dtype=np.int32)), (test_data, np.asanyarray(test_label, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--minibatchsize', '-mb', type=int, default=16,
                        help='Number of images in each mini-mini-batch')
    parser.add_argument('--valid', '-v', type=int, default=10,
                        help='Number of validation in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

	if args.dataset == 'cifar10':
		print('Using CIFAR10 dataset.')
		class_labels = 10
		cifar_train, cifar_test = get_cifar10()
	elif args.dataset == 'cifar100':
		print('Using CIFAR100 dataset.')
		class_labels = 100
		cifar_train, cifar_test = get_cifar100()
	else:
		raise RuntimeError('Invalid dataset choice.')

	model = models.VGG.VGG16(class_labels)
    if args.gpu >= 0:
		chainer.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    cifar_train, cifar_test = arange_cifar(cifar_train, cifar_test)
    train_data, train_label = cifar_train
    test_data, test_label = cifar_test
    if args.gpu >= 0:
		train_data = cuda.to_gpu(train_data)
		train_label = cuda.to_gpu(train_label)
		test_data = cuda.to_gpu(test_data)
		test_label = cuda.to_gpu(test_label)

    train_loop = len(train_data) // args.minibatchsize
    train_indices = np.arange(len(train_data))

    for epoch in xrange(1, args.epoch):
        np.random.shuffle(train_indices)
        sum_loss = 0

        with chainer.using_config("train", True):

            for itr in xrange(1, train_loop + 1):

                batch_range = np.arange(itr * args.batchsize, min((itr + 1) * args.batchsize, len(train_data)))
                x = train_data[train_indices[batch_range]]
                t = train_label[train_indices[batch_range]]

                init_weights = cache_weights(model)weight_list=[]valid_accuracy=np.zeros(args.valid, dtype=np.float32)svalid_loss=np.zeros(args.valid,dtype=np.float32)

                for valid_iter in xrange(args.valid):
                    #restore_weights(model, init_weights)

                    train_indices = np.random.choice(args.batchsize, args.minibatchsize)valid_indices = np.ones(args.batchsize, dtype=bool)valid_indices[train_indices] = False

                    x_train = x[train_indices]t_train = t[train_indices]x_valid = x[valid_indices]t_valid = t[valid_indices]

					if model.xp is cuda.cupy:
						x_train = cuda.to_gpu(x_train)
						t_train = cuda.to_gpu(t_train)
						x_valid = cuda.to_gpu(x_valid)
						t_valid = cuda.to_gpu(t_valid)

                    logits = model(x_train)
                    loss = F.softmax_cross_entropy(logits, Variable(t_train))

                    optimizer.update(lossfun=lambda: loss)

                    accuracy_valid = compute_classification_accuracy(model, x_valid, t_valid)

                    update_weights = cache_weights(model)weight_list.append(update_weights)valid_accuracy[valid_iter] = accuracy_validvalid_loss[valid_iter] = float(loss.data)

                best_index = np.argmax(valid_accuracy)best_weight = weight_list[best_index]best_loss = valid_loss[best_index]restore_weights(model, best_weight)

                sum_loss += best_loss

        with chainer.using_config("train", False):
            accuracy_train = compute_classification_accuracy(model, train_data, train_label)
            accuracy_test = compute_classification_accuracy(model, test_data, test_label)

        sys.stdout.write("\r\033[2KEpoch {} - loss: {:.8f} - acc: {:.5f} (train), {:.5f} (test)\n".format(epoch, sum_loss / train_loop, accuracy_train, accuracy_test))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
