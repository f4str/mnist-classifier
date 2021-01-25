import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class Recurrent(nn.Module):
	def __init__(self, lr=0.001, early_stopping=True, patience=4, cuda=True):
		super().__init__()
		self.early_stopping = early_stopping
		self.patience = patience
		self.device = torch.device('cuda:0' if cuda and torch.cuda.is_available() else 'cpu')
		
		self.hidden_size = 64
		self.num_layers = 2
		
		self.gru = nn.GRU(28, self.hidden_size, self.num_layers, batch_first=True)
		self.linear = nn.Linear(self.hidden_size, 10)
		
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		
		self.to(self.device)
	
	def forward(self, x):
		# init hidden state
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
		
		# gru: 28x28 -> 64x28 (x2)
		x, h = self.gru(x, h0)
		# linear: 64 -> 10
		x = self.linear(x[:, -1, :])
		
		return x
	
	def fit(self, X, y, epochs=10, batch_size=128, validation_split=0.2, verbose=True):
		self.train()
		
		# split data into training and validation sets
		dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
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
			start = time.time()
			if verbose:
				print(f'epoch {e + 1} / {epochs}:')
			
			# train on training data
			total = 0
			train_loss = 0
			train_acc = 0
			for data, labels in trainloader:
				data = data.to(self.device)
				labels = labels.to(self.device)
				
				self.optimizer.zero_grad()
				outputs = self(data)
				preds = outputs.argmax(dim=1)
				
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()
				
				train_acc += (preds == labels).sum().item()
				train_loss += loss.item() * len(data)
				
				current = time.time()
				if verbose:
					total += len(data)
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
			with torch.no_grad():
				for data, labels in validloader:
					data = data.to(self.device)
					labels = labels.to(self.device)
				
					outputs = self(data)
					preds = outputs.argmax(dim=1)
					
					loss = self.criterion(outputs, labels)
					valid_acc += (preds == labels).sum().item()
					valid_loss += loss.item() * len(data)
			
			valid_loss /= valid_size
			valid_acc /= valid_size
			total_valid_loss.append(valid_loss)
			total_valid_acc.append(valid_acc)
			
			end = time.time()
			if verbose:
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
		self.eval()
			
		with torch.no_grad():
			data = torch.FloatTensor(X).to(self.device)
			labels = torch.LongTensor(y).to(self.device)
			
			outputs = self(data)
			preds = outputs.argmax(dim=1)
			
			loss = self.criterion(outputs, labels).item()
			acc = (preds == labels).double().mean().item()
		
		return loss, acc
	
	def predict(self, X):
		self.eval()
		
		with torch.no_grad():
			data = torch.FloatTensor(X).to(self.device)
			outputs = self(data)
			preds = outputs.argmax(dim=1).cpu().numpy()
		
		return preds


if __name__ == "__main__":
	trainset = torchvision.datasets.MNIST('./data', transform=None, download=True, train=True)
	X_train = trainset.data.numpy() / 255
	y_train = trainset.targets.numpy()
	
	testset = torchvision.datasets.MNIST('./data', transform=None, download=True, train=False)
	X_test = testset.data.numpy() / 255
	y_test = testset.targets.numpy()
	
	model = Recurrent()
	model.fit(X_train, y_train, epochs=10)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
