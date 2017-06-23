import tensorflow as tf
import numpy as np


## session
sess = tf.Session()
hello = tf.constant("Hello...")

print(type(hello))
print(hello)


## running the Session

print(sess.run(hello))




## addition
a = tf.constant(10)
b = tf.constant(20)
add_result = tf.add(a, b)
print(sess.run(add_result))



## random variables from a binomial distribution
tf.random_normal



