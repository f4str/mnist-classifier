'''
feedforward neural network 
cross entropy cost function
sigmoid activation function
stochastic gradient descent
'''

import json
import os
import numpy as np

class NeuralNetwork:
	def __init__(self, sizes, training_rate = 0.5, stochastic = True, mini_batch_size = 32):
		self.sizes = sizes
		self.layers = len(sizes)
		self.weights = [np.random.randn(row, col) / np.sqrt(col) for row, col in zip(sizes[1:], sizes[:-1])]
		self.biases = [np.random.randn(row) for row in sizes[1:]]
		
		self.training_rate = training_rate
		self.stochastic = stochastic
		self.mini_batch_size = mini_batch_size
	
	def feedforward(self, a):
		for w, b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(w, a) + b)
		return a
	
	def predict(self, a):
		return np.argmax(self.feedforward(a))
	
	def train(self, training_data, epochs, test_data = None):
		training_cost = []
		test_cost = []
		training_accuracy = []
		test_accuracy = []
		
		for e in range(epochs):
			if self.stochastic:
				self.stochastic_gradient_descent(training_data)
			else:
				self.gradient_descent(training_data)
			
			if test_data:
				training_cost.append(self.total_cost(training_data))
				test_cost.append(self.total_cost(test_data, True))
				training_accuracy.append(self.accuracy(training_data))
				test_accuracy.append(self.accuracy(test_data, True))
			
			print(f'Epoch {e + 1}: complete')
		
		return (training_cost, test_cost, training_accuracy, test_accuracy)
	
	def gradient_descent(self, training_data):
		self.update_batch(training_data)
	
	def stochastic_gradient_descent(self, training_data):
		n = len(training_data)
		size = self.mini_batch_size
		
		np.random.shuffle(training_data)
		mini_batches = [training_data[i:i + size] for i in range(0, n, size)]
		for mini_batch in mini_batches:
			self.update_batch(mini_batch)
	
	def update_batch(self, batch):
		partial_w = [np.zeros(w.shape) for w in self.weights]
		partial_b = [np.zeros(b.shape) for b in self.biases]
		
		for x, y in batch:
			delta_partial_w, delta_partial_b = self.backpropagation(x, y)
			partial_w = [pw + dpw for pw, dpw in zip(partial_w, delta_partial_w)]
			partial_b = [pb + dpb for pb, dpb in zip(partial_b, delta_partial_b)]
		
		self.weights = [w - (self.training_rate / len(batch)) * pw for w, pw in zip(self.weights, partial_w)]
		self.biases = [b - (self.training_rate / len(batch)) * pb for b, pb in zip(self.biases, partial_b)]
	
	def backpropagation(self, x, y):
		partial_w = [np.zeros(w.shape) for w in self.weights]
		partial_b = [np.zeros(b.shape) for b in self.biases]
		
		# feedforward
		activation = x
		activations = [x]
		zs = []
		
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		# backward pass
		delta = (activations[-1] - y)
		partial_w[-1] = np.outer(delta, activations[-2])
		partial_b[-1] = delta
		
		for l in range(2, self.layers):
			delta = np.dot(delta, self.weights[-l + 1]) * sigmoid_derivative(zs[-l])
			partial_w[-l] = np.outer(delta, activations[-l - 1])
			partial_b[-l] = delta
		
		return (partial_w, partial_b)
	
	def total_cost(self, data, convert = False):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			if convert:
				y = convert_to_vector(y)
			cost += np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))) / len(data)
		return cost
	
	def accuracy(self, data, convert = False):
		if convert:
			results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
		else:
			results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
		
		accuracy = sum(int(x == y) for (x, y) in results)
		return accuracy
	
	def save(self, filename='network2.json'):
		data = {
			'sizes': self.sizes, 
			'weights': [w.tolist() for w in self.weights],
			'biases': [b.tolist() for b in self.biases],
			'training_rate': self.training_rate,
			'stochastic': self.stochastic,
			'mini_batch_size': self.mini_batch_size,
		}
		file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'networks', filename))
		f = open(file, 'w')
		json.dump(data, f)
		f.close()


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))

def convert_to_vector(y):
	v = np.zeros(10)
	v[y] = 1.0
	return v

def load(filename='network2.json'):
	file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'networks', filename))
	f = open(file, 'r')
	data = json.load(f)
	f.close()
	
	net = NeuralNetwork(data["sizes"])
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
	net.training_rate = data['training_rate']
	net.stochastic = data['stochastic']
	net.mini_batch_size = data['mini_batch_size']
	return net
