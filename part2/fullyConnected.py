#Use keras functions. Browse documentation to find information.
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Activation, Dense, Dropout


class PlanRecognitionModel:

	#Build the model architeture with keras layers
	def __init__(self, shape_input, dim_output):
		#Initialization of the neural network
		self.model = Sequential(name="model")
		self.model.add(Dense(units=20, input_shape=shape_input, activation="relu"))
		self.model.add(Dropout(0.1))
		self.model.add(Flatten())
		#Hidden layers
		self.model.add(Dense(units=20, activation="relu"))
		self.model.add(Dropout(0.1))
		self.model.add(Dense(units=10, activation="relu"))
		self.model.add(Dropout(0.1))
		#Output node
		self.model.add(Dense(units=dim_output, activation="relu"))
		self.model.add(Activation("softmax"))


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
