# -*- coding: utf-8 -*-

import tensorflow as tf
"""
def batch_norm_wrapper(inputs, phase_train, decay=0.99):
    epsilon = 1e-5
    out_dim = inputs.get_shape()[-1]
    scale = tf.Variable(tf.ones([out_dim]))
    beta = tf.Variable(tf.zeros([out_dim]))
    pop_mean = tf.Variable(tf.zeros([out_dim]), trainable=False)
    pop_var = tf.Variable(tf.ones([out_dim]), trainable=False)

    if phase_train == None:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    rank = len(inputs.get_shape())
    axes = range(rank - 1)  # nn:[0], conv:[0,1,2]
    batch_mean, batch_var = tf.nn.moments(inputs, axes)

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.nn.batch_normalization(inputs, tf.identity(batch_mean), tf.identity(batch_var), beta, scale, epsilon)
    def average():
        train_mean = pop_mean.assign(ema.average(batch_mean))
        train_var = pop_var.assign(ema.average(batch_var))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, train_mean, train_var, beta, scale, epsilon)

    return tf.cond(phase_train, update, average)
"""
def batch_norm_wrapper(inputs, phase_train, scope_name, scope_reuse, decay=0.99):
    epsilon = 1e-5
    out_dim = inputs.get_shape()[-1]
    with tf.variable_scope(scope_name) as scope:

        scale = tf.get_variable('scale', shape=[out_dim],
                       initializer=tf.constant_initializer(1.0), trainable=True)
        beta = tf.get_variable('beta', shape=[out_dim],
                       initializer=tf.constant_initializer(0.0), trainable=True)
        pop_mean = tf.get_variable('pop_mean', shape=[out_dim],
                       initializer=tf.constant_initializer(0.0), trainable=False)
        pop_var = tf.get_variable('pop_var', shape=[out_dim],
                       initializer=tf.constant_initializer(1.0), trainable=False)

    if phase_train == None:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    rank = len(inputs.get_shape())
    axes = range(rank - 1)  # nn:[0], conv:[0,1,2]
    batch_mean, batch_var = tf.nn.moments(inputs, axes)
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def update():
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.nn.batch_normalization(inputs, tf.identity(batch_mean), tf.identity(batch_var), beta, scale, epsilon)
    def average():
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_mean = pop_mean.assign(ema.average(batch_mean))
            train_var = pop_var.assign(ema.average(batch_var))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, train_mean, train_var, beta, scale, epsilon)

    return tf.cond(phase_train, update, average)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5, stddev=0.02):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed
