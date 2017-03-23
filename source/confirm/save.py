import tensorflow as tf
import numpy as np

def true_model(x):
    w = np.ones(5)
    return np.dot(x, w) + 2.0

X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.truncated_normal(shape=[5, 1]), name='w1')
w2 = tf.Variable(tf.truncated_normal(shape=[1]), name='w2')

y_pred = tf.matmul(X, w1) + w2

cost = tf.reduce_mean((Y-y_pred)**2)

opt_solver = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
step = 1000

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _iter in range(step):

    X_mb = np.random.randn(5, 5)
    Y_mb = true_model(X_mb)[:, np.newaxis]
    _, cost_curr = sess.run([opt_solver, cost], feed_dict={X: X_mb, Y: Y_mb})
    print sess.run(w1)
    print sess.run(w2)

save_path = saver.save(sess, 'model.ckpt')
print("Model saved in file: %s" % save_path)

sess.close()
