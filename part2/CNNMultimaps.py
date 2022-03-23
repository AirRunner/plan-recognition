import keras
from keras.layers import Flatten, Dropout, Dense, MaxPool2D, Conv2D, Reshape, Activation

from keras.models import Sequential, Model


class PlanRecognitionModel():
    def __init__(self, shape_input, dim_output):
        kernel_size = 8
        channels = shape_input[2]
        self.model = keras.Sequential([
            Conv2D(input_dim=channels, filters=channels, kernel_size=kernel_size, input_shape=shape_input,
                   padding="same", activation="relu"),
            Dropout(0.1),
            Conv2D(input_dim=channels, filters=channels, kernel_size=kernel_size, padding="same", activation="relu"),
            Dropout(0.1),
            Conv2D(input_dim=channels, filters=channels, kernel_size=kernel_size, padding="same", activation="relu"),
            Dropout(0.1),
            Conv2D(input_dim=channels, filters=channels, kernel_size=kernel_size, padding="same", activation="relu"),
            Dropout(0.1),
            Conv2D(input_dim=channels, filters=channels, kernel_size=kernel_size, padding="same", activation="relu"),
            Dropout(0.1),
            Conv2D(input_dim=channels, filters=channels, kernel_size=kernel_size, padding="same", activation="relu"),
            Dropout(0.1),
            Conv2D(input_dim=channels, filters=1, kernel_size=kernel_size, padding="same", activation="relu"),
            Dropout(0.1),
            Reshape((dim_output, dim_output)),
            Activation("softmax")
        ])

    def compile(self, loss="categorical_crossentropy"):
        self.model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    def fit(self, X, Y, epochs, batch):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch, validation_split=0.1)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return self.model.predict(X)
