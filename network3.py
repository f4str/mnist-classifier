"""
convolutional neural network
tensorflow 
cross entropy loss function
relu convolution activation function
max pooling
softmax fully connected activation function
adam optimizer
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def convolution_layer(x, num_inputs, num_filters, filter_size=5, strides=1, k=2):
	shape = [filter_size, filter_size, num_inputs, num_filters]
	weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
	biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
	layer = tf.nn.conv2d(x, filter=weights, strides=[1, strides, strides, 1], padding='SAME') + biases
	layer = tf.nn.max_pool(layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
	return tf.nn.relu(layer)


def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	return tf.reshape(layer, [-1, num_features]), num_features


def fully_connected_layer(x, num_inputs, num_outputs, relu=True):
	weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
	biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
	layer = tf.matmul(x, weights) + biases
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


class NeuralNetwork:
	def __init__(self):
		self.sess = tf.Session()
		
		self.learning_rate = 0.001
		self.batch_size = 128
		
		self.patience = 16
		
		self.num_input = 784
		self.num_classes = 10
		
		self.load_data()
		self.build()
	
	def load_data(self):
		mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
		self.training_data = mnist.train
		self.X_valid = mnist.validation.images
		self.y_valid = mnist.validation.labels
		self.X_test = mnist.test.images
		self.y_test = mnist.test.labels
	
	def build(self):
		self.x = tf.placeholder(tf.float32, [None, self.num_input])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		
		# Layer 0 = Reshape: 784 -> 28x28@1
		x_img = tf.reshape(self.x, shape=[-1, 28, 28, 1])
		# Layer 1 = Convolution + Pooling: 28x28@1 -> 14x14@16
		conv_layer1 = convolution_layer(x_img, num_inputs=1, num_filters=16)
		# Layer 2 = Convolution + Pooling: 14x14@16 -> 7x7@32
		conv_layer2 = convolution_layer(conv_layer1, num_inputs=16, num_filters=32)
		# Layer 3 = Flatten: 7x7@32 -> 1568
		flat_layer, num_features = flatten_layer(conv_layer2)
		# Layer 4 = 1568 -> 128
		fc_layer = fully_connected_layer(flat_layer, num_inputs=1568, num_outputs=128)
		# Layer 5 = Logits: 128 -> 10
		logits = fully_connected_layer(fc_layer, num_inputs=128, num_outputs=self.num_classes, relu=False)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.nn.softmax(logits)
	
	def train(self, epochs=200):
		self.sess.run(tf.global_variables_initializer())
		total_train_loss = []
		total_train_acc = []
		total_valid_loss = []
		total_valid_acc = []
		best_acc = 0
		no_acc_change = 0
		
		for e in range(epochs):
			x_batch, y_batch = self.training_data.next_batch(self.batch_size)
			feed_dict = {self.x: x_batch, self.y: y_batch}
			self.sess.run(self.optimizer, feed_dict=feed_dict)
			train_loss, train_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			
			feed_dict = {self.x: self.X_valid, self.y: self.y_valid}
			valid_loss, valid_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			
			total_train_loss.append(train_loss)
			total_train_acc.append(train_acc)
			total_valid_loss.append(valid_loss)
			total_valid_acc.append(valid_acc)
			
			print(f'epoch {e + 1}:',
				f'train loss = {train_loss:.4f},',
				f'train acc = {train_acc:.4f},',
				f'valid loss = {valid_loss:.4f},',
				f'valid acc = {valid_acc:.4f}'
			)
			
			if valid_acc > best_acc:
				best_acc = valid_acc
				no_acc_change = 0
			else:
				no_acc_change += 1
			
			if no_acc_change >= self.patience:
				print('early stopping')
				break
		
		print('training complete')
		
		feed_dict = {self.x: self.X_test, self.y: self.y_test}
		acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
		print(f'test accuracy = {acc:.4f}')
		
		return total_train_loss, total_train_acc, total_valid_loss, total_valid_acc


if __name__ == '__main__':
	net = NeuralNetwork()
	net.train(200)
