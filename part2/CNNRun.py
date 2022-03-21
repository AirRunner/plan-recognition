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
import CNN

np.set_printoptions(threshold=np.nan)

#Load obsevations
print("Loading data...")
f = h5py.File("bitmapObs/EnigmaBitmapObs.h5", 'r')
trainX = f["bitmaps"][:]
trainY = f["labels"][:]
testX25 = f["bitmapsTest25"][:]
testY25 = f["labelsTest25"][:]
testX50 = f["bitmapsTest50"][:]
testY50 = f["labelsTest50"][:]
testX75 = f["bitmapsTest75"][:]
testY75 = f["labelsTest75"][:]
testX100 = f["bitmapsTest100"][:]
testY100 = f["labelsTest100"][:]
start = f["start"][:]
goals = f["goals"][:]
f.close()

print("Input shape : " + str(trainX.shape))
print("Output shape : " + str(trainY.shape))


print("####### Training shuffle #######")

#Random shuffle
perm = np.random.permutation(trainX.shape[0])
trainX,trainY = trainX[perm],trainY[perm]

model = CNN.PlanRecognitionModel(trainX[0].shape, trainY.shape[1])
model.compile()

tStart = time.now()
model.fit(trainX, trainY, 10, 50)
tEnd = time.now()
c = tEnd - tStart

print("Training time : " + str(c.microseconds) + " microseconds")

print("####### Testing shuffle #######")

perm = np.random.permutation(len(testX25))
testX,testY = np.array(testX25)[perm],np.array(testY25)[perm]
#Assess network accuracy score with test set
scores = model.evaluate(testX25,testY25)
if scores is not None:
	print("Accuracy, from 0% to 25% : " + str(scores[1]*100) + " %")
else:
	print("Evaluation function not implemented")

perm = np.random.permutation(len(testX50))
testX,testY = np.array(testX50)[perm],np.array(testY50)[perm]
#Assess network accuracy score with test set
scores = model.evaluate(testX50,testY50)
if scores is not None:
	print("Accuracy, from 0% to 50% : " + str(scores[1]*100) + " %")
else:
	print("Evaluation function not implemented")

perm = np.random.permutation(len(testX75))
testX,testY = np.array(testX75)[perm],np.array(testY75)[perm]
#Assess network accuracy score with test set
scores = model.evaluate(testX75,testY75)
if scores is not None:
	print("Accuracy, from 0% to 75% : " + str(scores[1]*100) + " %")
else:
	print("Evaluation function not implemented")

perm = np.random.permutation(len(testX100))
testX,testY = np.array(testX100)[perm],np.array(testY100)[perm]
#Assess network accuracy score with test set
scores = model.evaluate(testX100,testY100)
if scores is not None:
	print("Accuracy, from 0% to 100% : " + str(scores[1]*100) + " %")
else:
	print("Evaluation function not implemented")
	
print("###############################################")
print("")
