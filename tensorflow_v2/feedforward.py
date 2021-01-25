import time
import tensorflow as tf
import numpy as np


class FeedForward(tf.keras.Model):
	def __init__(self, learning_rate=0.001, early_stopping=True, patience=4):
		super().__init__()
		self.early_stopping = early_stopping
		self.patience = patience
		
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(512, activation='relu')
		self.dense2 = tf.keras.layers.Dense(128, activation='relu')
		self.dense3 = tf.keras.layers.Dense(10)
		
		self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		
	def call(self, x):
		# flatten: 28x28 -> 784
		x = self.flatten(x)
		# linear: 784 -> 512 + relu
		x = self.dense1(x)
		# linear: 512 -> 128 + relu
		x = self.dense2(x)
		# linear: 128 -> 10
		x = self.dense3(x)
		
		return x
	
	def fit(self, X, y, epochs=10, batch_size=128, validation_split=0.2, verbose=True):
		# split into training and validation sets
		dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), reshuffle_each_iteration=False)
		valid_size = int(validation_split * len(X))
		train_size = len(X) - valid_size
		
		train_dataset = dataset.skip(valid_size).shuffle(train_size, reshuffle_each_iteration=True).batch(batch_size)
		valid_dataset = dataset.take(valid_size).batch(batch_size)
		
		total_train_loss = []
		total_train_acc = []
		total_valid_loss = []
		total_valid_acc = []
		best_acc = 0
		no_acc_change = 0
		
		for e in range(epochs):
			if verbose:
				start = time.time()
				print(f'epoch {e + 1} / {epochs}:')
			
			# train on training data
			total = 0
			train_loss = 0
			train_acc = 0
			for images, labels in train_dataset:
				batch = len(images)
				
				with tf.GradientTape() as tape:
					predictions = self(images)
					loss = self.loss(labels, predictions)
				
				gradients = tape.gradient(loss, self.trainable_variables)
				self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
				acc = self.accuracy(labels, predictions)
				
				train_loss += loss.numpy() * batch
				train_acc += acc.numpy() * batch
				
				if verbose:
					current = time.time()
					total += batch
					print(f'[{total} / {train_size}] - {(current - start):.2f} s -', 
						f'train loss = {(train_loss / total):.4f},',
						f'train acc = {(train_acc / total):.4f}',
						end='\r'
					)
			
			train_loss /= train_size
			train_acc /= train_size
			total_train_loss.append(train_loss)
			total_train_acc.append(train_acc)
			
			# test on validation data
			valid_loss = 0
			valid_acc = 0
			for images, labels in valid_dataset:
				batch = len(images)
				predictions = self(images)
				loss = self.loss(labels, predictions)
				acc = self.accuracy(labels, predictions)
				
				valid_loss += loss.numpy() * batch
				valid_acc += acc.numpy() * batch
			
			valid_loss /= valid_size
			valid_acc /= valid_size
			total_valid_loss.append(valid_loss)
			total_valid_acc.append(valid_acc)
			
			if verbose:
				end = time.time()
				print(f'[{total} / {train_size}] - {(end - start):.2f} s -',
					f'train loss = {train_loss:.4f},',
					f'train acc = {train_acc:.4f},',
					f'valid loss = {valid_loss:.4f},',
					f'valid acc = {valid_acc:.4f}'
				)
			
			# early stopping
			if self.early_stopping:
				if valid_acc > best_acc:
					best_acc = valid_acc
					no_acc_change = 0
				else:
					no_acc_change += 1
				
				if no_acc_change >= self.patience:
					if verbose:
						print('early stopping')
					break
		
		return total_train_loss, total_train_acc, total_valid_loss, total_valid_acc
	
	def evaluate(self, X, y):
		predictions = self(X)
		loss = self.loss(y, predictions).numpy()
		acc = self.accuracy(y, predictions).numpy()
		
		return loss, acc
	
	def predict(self, X):
		outputs = self(X)
		predictions = tf.argmax(outputs, axis=1).numpy()
		
		return predictions


if __name__ == '__main__':
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_train = X_train / 255
	X_test = X_test / 255
	
	model = FeedForward()
	model.fit(X_train, y_train, epochs=10)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
