import tensorflow as tf
import numpy as np


def load_data(normalize=False):
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	
	X_train = preprocess(X_train, normalize)
	X_test = preprocess(X_test, normalize)
	
	y_train = y_train.astype(np.float32)
	y_test = y_test.astype(np.float32)
	
	return (X_train, y_train), (X_test, y_test)


def preprocess(data, normalize):
	data = data.astype(np.float32)
	if normalize:
		for idx, img in enumerate(data):
			data[idx] = (img - np.mean(img)) / np.std(img)
		return data
	else:
		return data.astype(np.float32) / np.max(data)


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = load_data(normalize=False)
	print(f'X_train.shape = {X_train.shape}, X_train.dtype = {X_train.dtype}')
	print(f'X_train[0].shape = {X_train[0].shape}, X_train[0].dtype = {X_train[0].dtype}')
	print(f'X_train[0,0].shape = {X_train[0, 0].shape}, X_train[0,0].dtype = {X_train[0, 0].dtype}')
	print(f'X_train[0,0,0] = {X_train[0, 0, 0]}')
	print()
	print(f'y_train.shape = {y_train.shape}, y_train.dtype = {y_train.dtype}')
	print(f'y_train[0].shape = {y_train[0].shape}, y_train[0].dtype = {y_train[0].dtype}')
