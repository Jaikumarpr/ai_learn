
# !/usr/bin/env python3

"""
1. tensor
2. computational graph
3. Session
4. Placeholder
"""
import tensorflow as tf


a = tf.constant([2], dtype=tf.float32)
b = tf.constant([3], dtype=tf.float32)

c = a + b

with tf.Session() as sess:

    print(sess.run(c))
