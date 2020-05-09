import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader


class FeedForward(nn.Module):
	def __init__(self, lr=0.001, batch_size=128, early_stopping=True, patience=4):
		super().__init__()
		self.batch_size = batch_size
		self.early_stopping = early_stopping
		self.patience = patience
		
		self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
		self.linear1 = nn.Linear(in_features=2304, out_features=512)
		self.linear2 = nn.Linear(in_features=512, out_features=10)
		
		self.criterion = nn.CrossEntropyLoss(reduction='sum')
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
	
	def forward(self, x):
		# reshape: 28x28 -> 1@28x28
		x = x.view(-1, 1, 28, 28)
		# convolution: 1@28x28 -> 16@24x24 + relu
		x = self.conv(x)
		x = F.relu(x)
		# max pooling: 16@24x24 -> 16@12x12
		x = F.max_pool2d(x, kernel_size=2)
		# flatten: 16@12x12 -> 2304
		x = x.view(-1, 2304)
		# linear: 2304 -> 512 + relu
		x = self.linear1(x)
		x = F.relu(x)
		# linear: 512 -> 10
		x = self.linear2(x)
		
		return x
	
	def fit(self, X, y, epochs=10, validation_split=0.2, verbose=True):
		self.train()
		
		# split data into training and validation sets
		dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y))
		valid_size = int(validation_split * len(X))
		train_size = len(X) - valid_size
		train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
		
		# create batch iterators
		trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
		validloader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size)
		
		for e in range(epochs):
			# train on training data
			total = 0
			train_loss = 0
			train_acc = 0
			for data, labels in trainloader:
				self.optimizer.zero_grad()
				outputs = self(data)
				
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()
				
				pred = outputs.argmax(dim=1)
				train_acc += (pred == labels).sum().item()
				train_loss += loss.item()
				
				total += len(data)
				print(f'epoch {e + 1}: {total} / {train_size}', end='\r')
			
			train_loss /= train_size
			train_acc /= train_size
			
			# test on validation data
			valid_loss = 0
			valid_acc = 0
			with torch.no_grad():
				for data, labels in validloader:
					outputs = self(data)
					loss = self.criterion(outputs, labels)
					pred = outputs.argmax(dim=1)
					valid_acc += (pred == labels).sum().item()
					valid_loss += loss.item()
			
			valid_loss /= valid_size
			valid_acc /= valid_size
			
			print(f'epoch {e + 1}:',
				f'train loss = {train_loss:.4f},',
				f'train acc = {train_acc:.4f},',
				f'valid loss = {valid_loss:.4f},',
				f'valid acc = {valid_acc:.4f}'
			)
	
	def evaluate(self, X, y):
		self.eval()
		
		dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y))
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
		
		test_loss = 0
		test_acc = 0
		with torch.no_grad():
			for data, labels in dataloader:
				outputs = self(data)
				loss = self.criterion(outputs, labels)
				pred = outputs.argmax(dim=1)
				test_acc += (pred == labels).sum().item()
				test_loss += loss.item()
		
		test_loss /= len(X)
		test_acc /= len(X)
		
		return test_loss, test_acc
	
	def predict(self, X):
		self.eval()
		
		outputs = self(torch.Tensor(X))
		y = outputs.argmax(dim=1)
		
		return y.detach().numpy()


if __name__ == "__main__":
	X_train, y_train = data_loader.load_train(normalize=False)
	X_test, y_test = data_loader.load_test(normalize=False)
	
	model = FeedForward()
	model.fit(X_train, y_train, epochs=10)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
	
	y_pred = model.predict(X_test)
	print(y_pred)
	print(y_test)
	print(np.mean(y_pred == y_test))