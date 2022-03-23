import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Activation, Conv2D, Flatten, Dense



class PlanRecognitionModel():
	def __init__(self, shape_input, dim_output):
		#Initialization of the neural network
		filters = shape_input[2]
		filterSize = 5
		self.model = Sequential(name="model")
		self.model.add(Conv2D(filters, filterSize, padding="same", input_shape=shape_input, activation="relu"))
		self.model.add(Dropout(0.1))
		#Hidden layers
		self.model.add(Conv2D(filters, filterSize, padding="same", activation="relu"))
		self.model.add(Dropout(0.1))

		self.model.add(Conv2D(filters, filterSize, padding="same", activation="relu"))
		self.model.add(Dropout(0.1))

		self.model.add(Conv2D(filters, filterSize, padding="same", activation="relu"))
		self.model.add(Dropout(0.1))

		self.model.add(Conv2D(filters, filterSize, padding="same", activation="relu"))
		self.model.add(Dropout(0.1))

		self.model.add(Conv2D(filters, filterSize, padding="same", activation="relu"))
		self.model.add(Dropout(0.1))

		self.model.add(Conv2D(1, filterSize, padding="same", activation="relu"))
		self.model.add(Dropout(0.1))

		#Output nodes
		self.model.add(Flatten())
		self.model.add(Dense(units=dim_output))
		self.model.add(Activation(tf.nn.softmax))


	def compile(self):
		self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	def fit(self, X, Y, epochs, batch):
		return self.model.fit(X, Y, epochs=epochs, batch_size=batch, validation_split=0.1)

	def evaluate(self, X, Y):
		return self.model.evaluate(X,Y)

	def predict(self,X):
		return self.model.predict(X)
