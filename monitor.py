import numpy as np
import matplotlib.pyplot as plt
import data_loader
import network1, network2

training_data = data_loader.load_training_data()
validation_data = data_loader.load_validation_data()
test_data = data_loader.load_test_data()
print('data loaded')

nn = network2.NeuralNetwork([784, 30, 10])
fig, (ax1, ax2) = plt.subplots(1, 2)

epochs = np.arange(500)
training_cost, test_cost, training_accuracy, test_accuracy = nn.train(training_data[:500], len(epochs), test_data[:500])
print('training_complete')

ax1.set_title('Cost vs Epochs')
ax1.set(xlabel='Epochs', ylabel='Cost')
ax1.plot(epochs, training_cost, 'b', label='Training')
ax1.plot(epochs, test_cost, 'r', label='Testing')
ax1.legend()

ax2.set_title('Accuracy vs Epochs')
ax2.set(xlabel='Epochs', ylabel='Accuracy')
ax2.plot(epochs, training_accuracy, 'b', label='Training')
ax2.plot(epochs, test_accuracy, 'r', label='Testing')
ax2.legend()

plt.show()
