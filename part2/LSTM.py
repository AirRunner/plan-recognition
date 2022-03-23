from keras.layers import Dense, Activation, LSTM
from keras.models import Sequential


class PlanRecognitionModel():
    def __init__(self, shape_input, dim_output, dropout = 0.1, recurrent_dropout = 0.1, units = 10):
        self.model = Sequential(name="model")
        self.model.add(LSTM(units=units, input_shape=shape_input, dropout=dropout, recurrent_dropout=recurrent_dropout))
        self.model.add(Dense(units=dim_output, activation="relu"))
        self.model.add(Activation("Softmax"))

    def compile(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def fit(self, X, Y):
        return self.model.fit(X, Y, epochs=7, batch_size=200, validation_split=0.1)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return self.model.predict(X)
