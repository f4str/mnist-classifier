import numpy as np
from tensorflow import keras
import data_loader


def convolutional():
	model = keras.Sequential()
	model.add(keras.layers.Reshape((28, 28, 1)))
	model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(units=512, activation='relu'))
	model.add(keras.layers.Dense(units=10, activation='softmax'))
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	return model


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = data_loader.load_data(normalize=False)
	
	model = convolutional()
	model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
