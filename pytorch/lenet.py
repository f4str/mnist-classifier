import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class LeNet(nn.Module):
	def __init__(self, lr=0.001, early_stopping=True, patience=4):
		super().__init__()
		self.early_stopping = early_stopping
		self.patience = patience
		
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
		self.linear1 = nn.Linear(in_features=1024, out_features=256)
		self.linear2 = nn.Linear(in_features=256, out_features=64)
		self.linear3 = nn.Linear(in_features=64, out_features=10)
		
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
	
	def forward(self, x):
		# reshape: 28x28 -> 1@28x28
		x = x.view(-1, 1, 28, 28)
		# convolution: 1@28x28 -> 32@24x24 + relu
		x = self.conv1(x)
		x = F.relu(x)
		# max pooling: 32@24x24 -> 32@12x12
		x = F.max_pool2d(x, kernel_size=2)
		# convolution: 32@12x12 -> 64@8x8
		x = self.conv2(x)
		x = F.relu(x)
		# max pooling: 64@8x8 -> 64@4x4
		x = F.max_pool2d(x, kernel_size=2)
		# flatten: 12x12@16 -> 1024
		x = x.view(-1, 1024)
		# linear: 1024 -> 256 + relu
		x = self.linear1(x)
		x = F.relu(x)
		# linear: 256 -> 64 + relu
		x = self.linear2(x)
		x = F.relu(x)
		# linear: 64 -> 10
		x = self.linear3(x)
		
		return x
	
	def fit(self, X, y, epochs=10, batch_size=128, validation_split=0.2, verbose=True):
		self.train()
		
		# split data into training and validation sets
		dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y))
		valid_size = int(validation_split * len(X))
		train_size = len(X) - valid_size
		train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
		
		# create batch iterators
		trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
		validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
		
		total_train_loss = []
		total_train_acc = []
		total_valid_loss = []
		total_valid_acc = []
		best_acc = 0
		no_acc_change = 0
		
		for e in range(epochs):
			if verbose:
				print(f'epoch {e + 1} / {epochs}:')
			
			# train on training data
			total = 0
			train_loss = 0
			train_acc = 0
			for data, labels in trainloader:
				self.optimizer.zero_grad()
				outputs = self(data)
				preds = outputs.argmax(dim=1)
				
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()
				
				train_acc += (preds == labels).sum().item()
				train_loss += loss.item() * len(data)
				
				if verbose:
					total += len(data)
					print(f'[{total} / {train_size}]', 
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
			with torch.no_grad():
				for data, labels in validloader:
					outputs = self(data)
					preds = outputs.argmax(dim=1)
					
					loss = self.criterion(outputs, labels)
					valid_acc += (preds == labels).sum().item()
					valid_loss += loss.item() * len(data)
			
			valid_loss /= valid_size
			valid_acc /= valid_size
			total_valid_loss.append(valid_loss)
			total_valid_acc.append(valid_acc)
			
			if verbose:
				print(f'[{total} / {train_size}]',
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
		self.eval()
		
		data = torch.Tensor(X)
		labels = torch.LongTensor(y)
		
		with torch.no_grad():
			outputs = self(data)
			preds = outputs.argmax(dim=1)
			
			loss = self.criterion(outputs, labels).item()
			acc = (preds == labels).double().mean().item()
		
		return loss, acc
	
	def predict(self, X):
		self.eval()
		
		with torch.no_grad():
			outputs = self(torch.Tensor(X))
			preds = outputs.argmax(dim=1).numpy()
		
		return preds


if __name__ == "__main__":
	trainset = torchvision.datasets.MNIST('./data', transform=None, download=True, train=True)
	X_train = trainset.data.numpy().astype(np.float32) / 255
	y_train = trainset.targets.numpy()
	
	testset = torchvision.datasets.MNIST('./data', transform=None, download=True, train=False)
	X_test = testset.data.numpy().astype(np.float32) / 255
	y_test = testset.targets.numpy()
	
	model = LeNet()
	model.fit(X_train, y_train, epochs=2)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
	
	y_pred = model.predict(X_test)
	print(y_pred)
	print(y_test)
	print(np.mean(y_pred == y_test))