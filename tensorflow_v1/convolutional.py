import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import layers


class Convolutional:
	def __init__(self, learning_rate=0.001, early_stopping=True, patience=4):
		self.sess = tf.Session()
		self.early_stopping = early_stopping
		self.patience = patience
		
		self._build(learning_rate)
	
	def _build(self, learning_rate):
		# inputs
		self.X = tf.placeholder(tf.float32, [None, 28, 28])
		self.y = tf.placeholder(tf.int32, [None])
		
		# reshape: 28x28 -> 28x28@1
		reshaped = tf.reshape(self.X, [-1, 28, 28, 1])
		# convolution: 28x28@1 -> 24x24@16 + relu
		conv1 = tf.nn.relu(layers.conv2d(reshaped, 16, (5, 5)))
		# max pooling: 24x24@16 -> 12x12@16
		pool1 = layers.maxpool2d(conv1, (2, 2))
		# convolution: 12x12@16 -> 8x8@32 + relu
		conv2 = tf.nn.relu(layers.conv2d(pool1, 32, (5, 5)))
		# max pooling: 8x8@32 -> 4x4@32
		pool2 = layers.maxpool2d(conv2, (2, 2))
		# flatten: 4x4@32 -> 512
		flat = layers.flatten(pool2)
		# linear: 512 -> 128 + relu
		fc1 = tf.nn.relu(layers.linear(flat, 128))
		# linear: 128 -> 64 + relu
		fc2 = tf.nn.relu(layers.linear(fc1, 64))
		# linear: 64 -> 10
		logits = layers.linear(fc2, 10)
		# softmax cross entropy loss function
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
		
		self.loss = tf.reduce_mean(cross_entropy)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.train_op = optimizer.minimize(self.loss)
		
		self.prediction = tf.argmax(logits, axis=1, output_type=tf.dtypes.int32)
		correct_prediction = tf.equal(self.prediction, self.y)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	def fit(self, X, y, epochs=100, batch_size=128, validation_split=0.2, verbose=True):
		# shuffle input data
		p = np.random.permutation(len(X))
		X = np.array(X)[p]
		y = np.array(y)[p]
		
		# split into training and validation sets
		valid_size = int(validation_split * len(X))
		train_size = len(X) - valid_size
		
		dataset = tf.data.Dataset.from_tensor_slices((X, y))
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
			
			if verbose:
				start = time.time()
				print(f'epoch {e + 1} / {epochs}:')
			
			# train on training data
			total = 0
			train_loss = 0
			train_acc = 0
			try:
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
						current = time.time()
						total += size
						print(f'[{total} / {train_size}] - {(current - start):.2f} s -', 
							f'train loss = {(train_loss / total):.4f},',
							f'train acc = {(train_acc / total):.4f}',
							end='\r'
						)
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
		loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.X: X, self.y: y})
		return loss, acc
	
	def predict(self, X):
		y_pred = self.sess.run(self.prediction, feed_dict={self.X: X})
		return y_pred


if __name__ == '__main__':
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_train = X_train.astype(np.float32) / 255
	X_test = X_test.astype(np.float32) / 255
	
	model = Convolutional()
	model.fit(X_train, y_train, epochs=10)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
