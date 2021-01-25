import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def linear(inputs, num_outputs, bias=True):
	num_inputs = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.1))
	layer = tf.matmul(inputs, weights)
	
	if bias:
		biases = tf.Variable(tf.zeros([num_outputs]))
		layer = tf.add(layer, biases)
	
	return layer

def conv2d(inputs, filters, kernel_size=(5, 5), stride=1, padding='VALID', bias=True):
	channels = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(shape=[*kernel_size, channels, filters], mean=0, stddev=0.1))
	layer = tf.nn.conv2d(inputs, filter=weights, strides=[1, stride, stride, 1], padding=padding)
	
	if bias:
		biases = tf.Variable(tf.zeros([filters]))
		layer = tf.add(layer, biases)
	
	return layer


def maxpool2d(inputs, kernel_size=(2, 2), stride=None, padding='VALID'):
	stride = stride or kernel_size
	layer = tf.nn.max_pool2d(inputs, ksize=[1, *kernel_size, 1], strides=[1, *stride, 1], padding=padding)
	
	return layer


def gru(inputs, num_outputs, return_sequences=False):
	num_inputs = inputs.get_shape()[1]
	
	Wr = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.01))
	Wz = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.01))
	Wh = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.01))

	Ur = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.01))
	Uz = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.01))
	Uh = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.01))

	br = tf.Variable(tf.zeros([num_outputs]))
	bz = tf.Variable(tf.zeros([num_outputs]))
	bh = tf.Variable(tf.zeros([num_outputs]))
	
	def forward_pass(h_tm1, x_t):
		z_t = tf.sigmoid(tf.matmul(x_t, Wz) + tf.matmul(h_tm1, Uz) + bz)
		r_t = tf.sigmoid(tf.matmul(x_t, Wr) + tf.matmul(h_tm1, Ur) + br)
		h_proposal = tf.tanh(tf.matmul(x_t, Wh) + tf.matmul(tf.multiply(r_t, h_tm1), Uh) + bh)
		h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
		
		return h_t
	
	x_t = tf.transpose(inputs , [1, 0, 2])
	h_0 = tf.zeros(shape=[num_inputs, num_outputs])
	h_t = tf.scan(forward_pass, x_t, initializer=h_0)
	layer = tf.transpose(h_t, [1, 0, 2])
	
	if return_sequences:
		return layer
	else:
		return layer[:-1]


def flatten(inputs):
	layer_shape = inputs.get_shape()
	num_features = layer_shape[1:].num_elements()
	layer = tf.reshape(inputs, [-1, num_features])
	
	return layer
