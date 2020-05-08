import tensorflow as tf


def linear(inputs, num_outputs, bias=True):
	num_inputs = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[num_inputs, num_outputs],
		mean=0,
		stddev=0.1
	))
	layer = tf.matmul(inputs, weights)
	
	if bias:
		biases = tf.Variable(tf.zeros([num_outputs]))
		layer = tf.add(layer, biases)
	
	return layer

def conv2d(inputs, filters, kernel_size=5, stride=1, padding='VALID', bias=True):
	channels = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[kernel_size, kernel_size, channels, filters],
		mean=0,
		stddev=0.1
	))
	
	layer = tf.nn.conv2d(
		inputs,
		filter=weights,
		strides=[1, stride, stride, 1],
		padding=padding
	)
	
	if bias:
		biases = tf.Variable(tf.zeros([filters]))
		layer = tf.add(layer, biases)
	
	return layer


def maxpool2d(inputs, kernel_size=2, stride=None, padding='VALID'):
	stride = stride or kernel_size
	layer = tf.nn.max_pool2d(
		inputs,
		ksize=[1, kernel_size, kernel_size, 1],
		strides=[1, stride, stride, 1],
		padding=padding
	)
	
	return layer


def flatten(inputs):
	layer_shape = inputs.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer = tf.reshape(inputs, [-1, num_features])
	
	return layer
