'''
convolutional neural network
tensorflow 
cross entropy loss function
relu convolution activation function
max pooling
softmax fully connected activation function
adam optimizer
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def convolution_layer(x, num_inputs, filter_size, num_filters, strides=1, k=2):
	shape = [filter_size, filter_size, num_inputs, num_filters]
	weights = tf.Variable(tf.random_normal(shape))
	biases = tf.Variable(tf.random_normal([num_filters]))
	
	layer = tf.nn.conv2d(x, filter=weights, strides=[1, strides, strides, 1], padding='SAME') + biases
	layer = tf.nn.relu(layer)
	layer = tf.nn.max_pool(layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
	return layer

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	return tf.reshape(layer, [-1, num_features]), num_features

def fully_connected_layer(x, num_inputs, num_outputs, relu=True):
	weights = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
	biases = tf.Variable(tf.random_normal([num_outputs]))
	
	layer = tf.matmul(x, weights) + biases
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


class NeuralNetwork:
	def __init__(self):
		self.learning_rate = 0.001
		self.batch_size = 128
		
		self.num_input = 784
		self.num_classes = 10
		
		self.filter1_size = 5
		self.num_filter1 = 32
		
		self.filter2_size = 5
		self.num_filter2 = 64
		
		self.fc_size = 512
		
		self.load_data()
		self.build()
	
	def load_data(self):
		self.data = input_data.read_data_sets('data/MNIST/', one_hot=True)
	
	def build(self):
		self.x = tf.placeholder(tf.float32, [None, self.num_input])
		x_img = tf.reshape(self.x, shape=[-1, 28, 28, 1])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		
		conv_layer1 = convolution_layer(x_img, 1, self.filter1_size, self.num_filter1)
		conv_layer2 = convolution_layer(conv_layer1, self.num_filter1, self.filter2_size, self.num_filter2)
		flat_layer, num_features = flatten_layer(conv_layer2)
		fc_layer = fully_connected_layer(flat_layer, num_features, self.fc_size)
		
		logits = fully_connected_layer(fc_layer, self.fc_size, self.num_classes, relu=False)
		prediction = tf.nn.softmax(logits)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	def train(self, epochs = 500):
		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			
			for i in range(epochs):
				x_batch, y_batch = self.data.train.next_batch(self.batch_size)
				feed_dict = {self.x: x_batch, self.y: y_batch}
				
				session.run(self.optimizer, feed_dict=feed_dict)
				
				loss, acc = session.run([self.loss, self.accuracy], feed_dict=feed_dict)
				print(f'epoch {i + 1}: loss = {loss:.4f}, training accuracy = {acc:.4f}')
			print('training complete')
			
			feed_dict = {self.x: self.data.test.images, self.y: self.data.test.labels}
			acc = session.run(self.accuracy, feed_dict=feed_dict)
			print(f'test accuracy = {acc:.4f}')

if __name__ == '__main__':
	net = NeuralNetwork()
	net.train(500)
