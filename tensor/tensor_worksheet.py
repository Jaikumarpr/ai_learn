# !/usr/bin/env python3

from __future__ import print_function
import tensorflow as tf


W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


linear_model = W * x + b

square_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(square_deltas)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])

sess.run([fixW, fixb])

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
