'''
feedforward neural network
tensorflow
cross entropy loss function
softmax activation function
adam optimizer
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class NeuralNetwork:
	def __init__(self):
		self.learning_rate = 0.1
		self.num_steps = 500
		self.batch_size = 128
		
		self.n_hidden_1 = 256
		self.n_hidden_2 = 256
		self.num_input = 784
		self.num_classes = 10
		
		self.weights = [
			tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1])),
			tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
			tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]))
		]
		self.biases = [
			tf.Variable(tf.random_normal([self.n_hidden_1])),
			tf.Variable(tf.random_normal([self.n_hidden_2])),
			tf.Variable(tf.random_normal([self.num_classes]))
		]
		
		self.load_data()
		self.build()
	
	def load_data(self):
		self.data = input_data.read_data_sets('data/MNIST/', one_hot=True)
	
	def build(self):
		self.x = tf.placeholder(tf.float32, [None, self.num_input])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		
		self.layer1 = tf.matmul(self.x, self.weights[0]) + self.biases[0]
		self.layer2 = tf.matmul(self.layer1, self.weights[1]) + self.biases[1]
		self.logits = tf.matmul(self.layer2, self.weights[2]) + self.biases[2]
		self.prediction = tf.nn.softmax(self.logits)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	def train(self):
		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			
			for i in range(self.num_steps):
				x_batch, y_batch = self.data.train.next_batch(self.batch_size)
				feed_dict = {self.x: x_batch, self.y: y_batch}
				
				session.run(self.optimizer, feed_dict=feed_dict)
				
				loss, acc = session.run([self.loss, self.accuracy], feed_dict=feed_dict)
				print(f'step {i + 1}: loss = {loss:.4f}, training accuracy = {acc:.4f}')
			print('training complete')
			
			feed_dict = {self.x: self.data.test.images, self.y: self.data.test.labels}
			acc = session.run(self.accuracy, feed_dict=feed_dict)
			print(f'test accuracy = {acc:.4f}')

if __name__ == '__main__':
	net = NeuralNetwork()
	net.train()
