import tensorflow as tf
import numpy as np
import data_loader
import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class FeedForward:
	def __init__(self, learning_rate=0.001, early_stopping=True, patience=4):
		self.sess = tf.Session()
		self.early_stopping = early_stopping
		self.patience = patience
		
		self._build(learning_rate)
	
	def _build(self, learning_rate):
		# inputs
		self.X = tf.placeholder(tf.float32, [None, 28, 28])
		self.y = tf.placeholder(tf.int32, [None])
		one_hot_y = tf.one_hot(self.y, 10)
		
		# reshape: 28x28 -> 784
		reshaped = tf.reshape(self.X, shape=[-1, 784])
		# linear: 784 -> 512 + relu
		linear1 = layers.linear(reshaped, num_outputs=512)
		linear1 = tf.nn.relu(linear1)
		# linear: 512 -> 128 + relu
		linear2 = layers.linear(linear1, num_outputs=128)
		linear2 = tf.nn.relu(linear2)
		# linear: 128 -> 10
		logits = layers.linear(linear2, num_outputs=10)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
		self.loss = tf.reduce_mean(cross_entropy)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.train_op = optimizer.minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(one_hot_y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.argmax(logits, axis=1)
	
	def fit(self, X, y, epochs=100, batch_size=128, validation_split=0.2, verbose=True):
		# split into training and validation sets
		dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X))
		valid_size = int(validation_split * len(X))
		train_size = len(X) - valid_size
		
		train_dataset = dataset.skip(valid_size).shuffle(train_size, reshuffle_each_iteration=True).batch(batch_size)
		valid_dataset = dataset.take(valid_size).batch(batch_size)
		
		# create batch iterator
		train_iterator = train_dataset.make_initializable_iterator()
		valid_iterator = valid_dataset.make_initializable_iterator()
		
		X_train, y_train = train_iterator.get_next()
		X_valid, y_valid = valid_iterator.get_next()
		
		total_train_loss = []
		total_train_acc = []
		total_valid_loss = []
		total_valid_acc = []
		best_acc = 0
		no_acc_change = 0
		
		self.sess.run(tf.global_variables_initializer())
		
		for e in range(epochs):
			# initialize training batch iterator
			self.sess.run(train_iterator.initializer)
			
			# train on training data
			train_loss = 0
			train_acc = 0
			try:
				batch = 0
				while True:
					X_batch, y_batch = self.sess.run([X_train, y_train])
					size = len(X_batch)
					
					_, loss, acc = self.sess.run(
						[self.train_op, self.loss, self.accuracy], 
						feed_dict={self.X: X_batch, self.y: y_batch}
					)
					train_loss += loss * size
					train_acc += acc * size
					
					if verbose:
						batch += size
						print(f'epoch {e + 1}: {batch} / {train_size}', end='\r')
			except tf.errors.OutOfRangeError:
				pass
			
			train_loss /= train_size
			train_acc /= train_size
			total_train_loss.append(train_loss)
			total_train_acc.append(train_acc)
			
			# initialize validation batch iterator
			self.sess.run(valid_iterator.initializer)
			
			# test on validation data
			valid_loss = 0
			valid_acc = 0
			try:
				while True:
					X_batch, y_batch = self.sess.run([X_valid, y_valid])
					size = len(X_batch)
					
					loss, acc = self.sess.run(
						[self.loss, self.accuracy], 
						feed_dict={self.X: X_batch, self.y: y_batch}
					)
					valid_loss += loss * size
					valid_acc += acc * size
			except tf.errors.OutOfRangeError:
				pass
			
			valid_loss /= valid_size
			valid_acc /= valid_size
			total_valid_loss.append(valid_loss)
			total_valid_acc.append(valid_acc)
			
			if verbose:
				print(f'epoch {e + 1}:',
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
		loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.X: X, self.y: y})
		return loss, acc
	
	def predict(self, X):
		y_pred = self.sess.run(self.prediction, feed_dict={self.X: X})
		return y_pred


if __name__ == '__main__':
	(X_train, y_train), (X_test, y_test) = data_loader.load_data(normalize=False)
	
	model = FeedForward()
	model.fit(X_train, y_train, epochs=10)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
	
	y_pred = model.predict(X_test)
	print(y_pred)
	print(y_test)
	print(np.mean(y_pred == y_test))
