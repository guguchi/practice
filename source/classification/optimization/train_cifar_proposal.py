#!/usr/bin/env python
"""Convnet example using CIFAR10 or CIFAR100 dataset

This code is a custom loop version of train_cifar.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
from __future__ import print_function
import argparse
import sys

import numpy as np
import matplotlib as plt

import chainer
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers, cuda

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG


def cache_weights(model):
    cached_weights = {}
    for name, param in model.namedparams():
        with cuda.get_device(param.data):
            xp = cuda.get_array_module(param.data)
            cached_weights[name] = xp.copy(param.data)
    return cached_weights


def restore_weights(model, cached_weights):
    for (name, param) in model.namedparams():
        with cuda.get_device(param.data):
            if name not in cached_weights:
                raise Exception()
            param.data = cached_weights[name]


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
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    if args.test:
        train = train[:200]
        test = test[:200]

    train_count = len(train)
    test_count = len(test)

    print(train_count)
    print(test_count)

    model = L.Classifier(models.VGG.VGG(class_labels))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0
    iteration = 0

    thresh = int(float(args.batchsize) / args.minibatchsize)

    while train_iter.epoch < args.epoch * thresh:
        batch = train_iter.next()
        # Reduce learning rate by 0.5 every 25 epochs.
        if int(train_iter.epoch / thresh) % (25 * thresh) == 0 and train_iter.is_new_epoch:
            optimizer.lr *= 0.5
            print('Reducing learning rate to: ', optimizer.lr)

        x_array, t_array = convert.concat_examples(batch, args.gpu)

        if iteration == 0:
            x = chainer.Variable(x_array)
            t = chainer.Variable(t_array)
            optimizer.update(model, x, t)
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)

        else:
            init_weights = cache_weights(model)
            weight_list = []
            valid_accuracy = np.zeros(args.valid, dtype=np.float32)
            valid_loss = np.zeros(args.valid, dtype=np.float32)

            for valid_iter in xrange(args.valid):
                restore_weights(model, init_weights)

                train_indices = np.random.choice(args.batchsize, args.minibatchsize)
                valid_indices = np.ones(args.batchsize, dtype=bool)
                valid_indices[train_indices] = False

                x_train = x_array[train_indices]
                t_train = t_array[train_indices].astype(np.int32)
                x_valid = x_array[valid_indices]
                t_valid = t_array[valid_indices].astype(np.int32)

                x = chainer.Variable(x_train)
                t = chainer.Variable(t_train)
                optimizer.update(model, x, t)

                update_weights = cache_weights(model)

                x = chainer.Variable(x_valid)
                t = chainer.Variable(t_valid)
                loss = model(x, t)
                accuracy_valid = float(model.accuracy.data)

                weight_list.append(update_weights)
                valid_accuracy[valid_iter] = accuracy_valid
                valid_loss[valid_iter] = float(loss.data)

            best_index = np.argmax(valid_accuracy)
            best_weight = weight_list[best_index]
            best_loss = valid_loss[best_index]
            best_accuracy = valid_accuracy[best_index]
            restore_weights(model, best_weight)

            sum_loss += best_loss * len(t.data)
            sum_accuracy += best_accuracy * len(t.data)

        iteration += 1

        if train_iter.epoch % (1 * thresh) == 0 and train_iter.is_new_epoch:
            print('epoch: ', train_iter.epoch)
            print('train mean loss: {}, accuracy: {}'.format(
                sum_loss / train_count, sum_accuracy / train_count))
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            model.predictor.train = False
            for batch in test_iter:
                x_array, t_array = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            test_iter.reset()
            model.predictor.train = True
            print('test mean  loss: {}, accuracy: {}'.format(
                sum_loss / test_count, sum_accuracy / test_count))
            sum_accuracy = 0
            sum_loss = 0

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)


if __name__ == '__main__':
    main()
