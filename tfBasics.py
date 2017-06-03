#I don't know why I have to do that
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1, x2)

print(result)

# Running the session

with tf.Session() as sess:
    output = sess.run(result)
    print(output)