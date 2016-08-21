

import numpy as np
import tensorflow as tf



sess = tf.Session()


# running tf.session with a variable
hello  = tf.constant("Hello...it's me")

print(type(sess.run(hello)))
print(type(hello))


# constant, variables, placeholders
a = tf.constant(1.3)
b = tf.constant(1.3)

print sess.run(a+b)
print sess.run(a) + sess.run(b)

print(a)
print(type(a))

# Operators are tensorflow variables
print("Addition operation as a variable...")
add = tf.add(a,b)
print(type(add))
print(sess.run(add))
print("add_out")
mul = tf.mul(a, b)
print("Multiplication...")
print(sess.run(mul))


print("Variables and Placeholders")
x = np.random.rand(1,20)
Input  = tf.placeholder(tf.float32, [None, 20])

print(Input)
print(type(Input))

# random values from a normal distribution
Weight = tf.Variable(tf.random_normal([20, 10], stddev=0.5))
Bias   = tf.Variable(tf.zeros([1, 10]))

print(type(Weight))
print(type(Bias))

init = tf.initialize_all_variables()
sess.run(init)

# We will be evaluating variables in a session - Weights, here
print("weight is ", Weight)
print(Weight.eval(sess).shape)
print(Weight.eval(sess))

print("Bias terms : ", Bias)
print(Bias.eval(sess))

X = np.random.rand(1, 20)

# X is a (1 x 20) matrix
# Weights is a (20 x 10) matrix

# Matrix multiplication (1 * 20) * (20 * 10) = (1 * 10)
oper = tf.matmul(Input, Weight) + Bias
val = sess.run(oper, feed_dict={Input:X})
print(val.shape)


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

a = 12
b =  13

add = tf.add(x, y)
add_out = sess.run(add, feed_dict={x:a, y:b})
print(add_out)
