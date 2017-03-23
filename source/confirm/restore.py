import tensorflow as tf
import numpy as np

w1 = tf.Variable(tf.zeros(shape=[5, 1]), name='w1')
w2 = tf.Variable(tf.zeros(shape=[1]), name='w2')

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        print "load " + last_model
        saver.restore(sess, last_model)

        print sess.run(w1)
        print sess.run(w2)

    else:
        init = tf.initialize_all_variables()
        sess.run(init)
