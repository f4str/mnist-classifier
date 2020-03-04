'''
feedforward neural network
tensorflow
cross entropy loss function
softmax activation function
adam optimizer
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def linear_layer(input, num_inputs, num_outputs, relu=True):
	weights = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
	biases = tf.Variable(tf.zeros([num_outputs]))
	layer = tf.matmul(input, weights) + biases
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


class NeuralNetwork:
	def __init__(self):
		self.learning_rate = 0.005
		self.batch_size = 128
		
		self.n_hidden_1 = 256
		self.n_hidden_2 = 256
		self.num_input = 784
		self.num_classes = 10
		
		self.load_data()
		self.build()
	
	def load_data(self):
		self.data = input_data.read_data_sets('data/MNIST/', one_hot=True)
	
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
