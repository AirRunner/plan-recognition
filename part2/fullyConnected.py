#Use keras functions. Browse documentation to find information.
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.core import Dropout


class PlanRecognitionModel:

	#Build the model architeture with keras layers
	def __init__(self, shape_input, dim_output):
		filter_n = 4
		filter_input = 4
		self.model = Sequential(name="model")
		for _ in range(6):
			self.model.add(Conv2D(filter_input, filter_n, padding="same", input_shape=8, strides=1, activation="relu"))
			self.model.add(Dropout(0.1))

		self.model.add(
			Conv2D(filter_input, filter_n, padding="same", input_shape=shape_input, strides=1, activation="relu"))
		self.model.add(Dropout(0.1))

		self.model.add(Flatten())

	#Compile the model with keras.
	def compile(self):
		self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	#Fit the model with keras
	def fit(self, X, Y, epochs, batch):
		return self.model.fit(X, Y, epochs=epochs, batch_size=batch, validation_split=0.1)

	#Evaluate the model with keras
	def evaluate(self, X, Y):
		return self.model.evaluate(X,Y)

	#Return a prediction with keras
	def predict(self,X):
		return self.model.predict(X)
