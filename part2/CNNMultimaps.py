from keras.layers import Flatten, Dropout, Dense, MaxPool2D, Conv2D, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.activations import softmax
import h5py,sys,glob
import numpy as np
from datetime import datetime as time
import random

np.set_printoptions(threshold=np.nan)

class PlanRecognitionModel():
	def __init__(self, shape_input, dim_output):
		#Add code here
		self.model = None

	def compile(self):
		#Add code here
		pass

	def fit(self, X, Y, epochs, batch):
		#Add code here
		return None

	def evaluate(self, X, Y):
		#Add code here
		return None

	def predict(self,X):
		#Add code here
		return None
