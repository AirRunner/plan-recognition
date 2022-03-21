from keras.layers import Flatten, Dropout, Dense, MaxPool2D, Conv2D, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.activations import softmax
import h5py,sys,glob
import numpy as np
from datetime import datetime as time
import random
import tensorflow as tf

#Use keras functions. Browse documentation to find information.
class PlanRecognitionModel():

	#Build the model architeture with keras layers
	def __init__(self, shape_input, dim_output):
		#Add code here
		self.model = None

	#Compile the model with keras.
	def compile(self):
		#Add code here
		pass

	#Fit the model with keras
	def fit(self, X, Y, epochs, batch):
		#Add code here
		return None

	#Evaluate the model with keras
	def evaluate(self, X, Y):
		#Add code here
		return None

	#Return a prediction with keras
	def predict(self,X):
		#Add code here
		return None
