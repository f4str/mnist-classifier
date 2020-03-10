"""
feedforward neural network
tensorflow
cross entropy loss function
softmax activation function
adam optimizer
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def linear_layer(input, num_inputs, num_outputs, relu=True):
	weights = tf.Variable(tf.random.normal([num_inputs, num_outputs]))
	biases = tf.Variable(tf.zeros([num_outputs]))
	layer = tf.matmul(input, weights) + biases
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


class NeuralNetwork:
	def __init__(self):
		self.sess = tf.Session()
		
		self.learning_rate = 0.005
		self.batch_size = 128
		
		self.patience = 32
		
		self.n_hidden_1 = 256
		self.n_hidden_2 = 256
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
		
		layer1 = linear_layer(self.x, self.num_input, self.n_hidden_1)
		layer2 = linear_layer(layer1, self.n_hidden_1, self.n_hidden_2)
		logits = linear_layer(layer2, self.n_hidden_2, self.num_classes, relu=False)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.argmax(logits)
	
	def train(self, epochs=500):
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
	net.train(500)
