import numpy as np
import matplotlib.pyplot as plt
import network1

net = network1.NeuralNetwork()
net.patience = 10000

epochs = np.arange(500)
train_loss, train_acc, valid_loss, valid_acc = net.train(len(epochs))
print('training_complete')

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('Loss vs Epochs')
ax1.set(xlabel='Epochs', ylabel='Loss')
ax1.plot(epochs, train_loss, 'b', label='Training')
ax1.plot(epochs, valid_loss, 'r', label='Validation')
ax1.legend()

ax2.set_title('Accuracy vs Epochs')
ax2.set(xlabel='Epochs', ylabel='Accuracy')
ax2.plot(epochs, train_acc, 'b', label='Training')
ax2.plot(epochs, valid_acc, 'r', label='Validation')
ax2.legend()

plt.show()
