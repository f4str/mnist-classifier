import torchvision
import numpy as np


def load_train(normalize=False):
	trainset = torchvision.datasets.MNIST('./data', transform=None, download=True, train=True)
	
	X_train = trainset.data.numpy()
	y_train = trainset.targets.numpy()
	
	X_train = preprocess(X_train, normalize)
	y_train = y_train.astype(np.float32)
	
	return X_train, y_train

def load_test(normalize=False):
	testset = torchvision.datasets.MNIST('./data', transform=None, download=True, train=False)
	
	X_test = testset.data.numpy()
	y_test = testset.targets.numpy()
	
	X_test = preprocess(X_test, normalize)
	y_test = y_test.astype(np.float32)
	
	return X_test, y_test


def preprocess(data, normalize):
	data = data.astype(np.float32)
	if normalize:
		for idx, img in enumerate(data):
			data[idx] = (img - np.mean(img)) / np.std(img)
		return data
	else:
		return data.astype(np.float32) / np.max(data)


if __name__ == "__main__":
	X_train, y_train = load_train(normalize=False)
	print(f'X_train.shape = {X_train.shape}, X_train.dtype = {X_train.dtype}')
	print(f'X_train[0].shape = {X_train[0].shape}, X_train[0].dtype = {X_train[0].dtype}')
	print(f'X_train[0,0].shape = {X_train[0, 0].shape}, X_train[0,0].dtype = {X_train[0, 0].dtype}')
	print(f'X_train[0,0,0] = {X_train[0, 0, 0]}')
	print()
	print(f'y_train.shape = {y_train.shape}, y_train.dtype = {y_train.dtype}')
	print(f'y_train[0] = {y_train[0]}')
