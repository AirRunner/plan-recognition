from keras.layers import Flatten, Dropout, Dense, MaxPool2D, Conv2D, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.activations import softmax
import h5py,sys,glob
import numpy as np
from datetime import datetime as time
import random
import CNNMultimaps

np.set_printoptions(threshold=np.nan)

#Load obsevations
print("Loading data...")
trainX = []
trainY = []
testX25 = []
testY25 = []
testX50 = []
testY50 = []
testX75 = []
testY75 = []
testX100 = []
testY100 = []
for map in sorted(glob.glob('bitmapToBitmapObs/*BitmapToBitmapObs.h5')):
	map = map[18:-20]
	print("Loading map : " + map)
	f = h5py.File("bitmapToBitmapObs/" + map + "BitmapToBitmapObs.h5", 'r')
	for b in f["bitmaps"][:]:
		trainX.append(b)
	for bl in f["bitmapsLabels"][:]:
		trainY.append(bl)

	for bt in f["bitmapsTest25"][:]:
		testX25.append(bt)
	for blt in f["bitmapsLabelsTest25"][:]:
		testY25.append(blt)

	for bt in f["bitmapsTest50"][:]:
		testX50.append(bt)
	for blt in f["bitmapsLabelsTest50"][:]:
		testY50.append(blt)

	for bt in f["bitmapsTest75"][:]:
		testX75.append(bt)
	for blt in f["bitmapsLabelsTest75"][:]:
		testY75.append(blt)

	for bt in f["bitmapsTest100"][:]:
		testX100.append(bt)
	for blt in f["bitmapsLabelsTest100"][:]:
		testY100.append(blt)
	f.close()

trainX = np.array(trainX)
trainY = np.array(trainY)
testX25 = np.array(testX25)
testY25 = np.array(testY25)
testX50 = np.array(testX50)
testY50 = np.array(testY50)
testX75 = np.array(testX75)
testY75 = np.array(testY75)
testX100 = np.array(testX100)
testY100 = np.array(testY100)

print("Input shape : " + str(trainX.shape))
print("Output shape : " + str(trainY.shape))

#Random shuffle
perm = np.random.permutation(trainX.shape[0])
trainX,trainY = trainX[perm],trainY[perm]

model = CNNMultimaps.PlanRecognitionModel(trainX[0].shape, trainY.shape[1])
model.compile()

tStart = time.now()
model.fit(trainX, trainY, 10, 10)
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
