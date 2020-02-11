import data_loader
import network2 as nn

epochs = 20

# load data
training_data = data_loader.load_training_data()
validation_data = data_loader.load_validation_data()
test_data = data_loader.load_test_data()
print('data loaded')

try:
	# load network if already exists
	net = nn.load()
	print('network loaded')
except Exception:
	# create new network otherwise
	net = nn.NeuralNetwork([784, 30, 10])
	print('network created')

# train network
net.train(training_data, epochs)
print('training complete')

# save network
net.save()
print('network saved')

# test network
net.test(test_data)
