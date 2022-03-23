import h5py
import numpy as np
from datetime import datetime as time
import random
import LSTM

# Load obsevations
print("Loading data...")
f = h5py.File("obs/EnigmaObs.h5", 'r')
obs = [o[:] for o in f["obs"].values()]
labels = [l[:] for l in f["labels"].values()]
start = f["start"][:]
goals = f["goals"][:]
f.close()

# Nb of subsets to generate in a path (ie nb of obsevation sequences)
nbSubsetPerPath = 500
# Part of the path to take into account for network training. For instance, if this value is set to [0,0.25], the samples will only be generated from the first 25% of the path
percentageSubPathTrain = [0, 1]
# Part of the path to take into account for network testing and M&S testing
percentageSubPathTest = [[0, 0.25], [0, 0.5], [0, 0.75], [0, 1]]
# Nb of observation to sample in a sequence
sizeOfSubset = 10

print("Processing data:")
trainX = []
trainY = []
testsX = []
testsY = []
# Training set
print("Training set...")
for i in range(len(obs)):
    print(str(i) + "/" + str(len(obs)) + "    ", end="\r")
    o = obs[i]
    for k in range(nbSubsetPerPath):
        # Train
        subPath = o[int(percentageSubPathTrain[0] * len(o)):int(percentageSubPathTrain[1] * len(o))]
        if (len(subPath) >= sizeOfSubset):
            sample = []
            for index in sorted(random.sample(range(len(subPath)), sizeOfSubset)):
                sample.append([subPath[index][0], subPath[index][1]])
            trainX.append(sample)
            trainY.append(labels[i])

# Testing set
print("Testing set...")
for interval in percentageSubPathTest:
    testX = []
    testY = []
    for i in range(len(obs)):
        print(str(interval) + " - " + str(i) + "/" + str(len(obs)) + "    ", end="\r")
        o = obs[i]
        for k in range(nbSubsetPerPath):
            subPath = o[int(interval[0] * len(o)):int(interval[1] * len(o))]
            if (len(subPath) >= sizeOfSubset):
                sample = []
                for index in sorted(random.sample(range(len(subPath)), sizeOfSubset)):
                    sample.append([subPath[index][0], subPath[index][1]])
                testX.append(sample)
                testY.append(labels[i])
    testsX.append(testX)
    testsY.append(testY)

trainX = np.array(trainX)
trainY = np.array(trainY)

print("Input shape : " + str(trainX.shape))
print("Output shape : " + str(trainY.shape))

print("####### Training shuffle #######")

# Random shuffle
perm = np.random.permutation(trainX.shape[0])
trainX, trainY = trainX[perm], trainY[perm]

for unit in [5, 10, 20, 50, 200]:
    model = LSTM.PlanRecognitionModel(trainX[0].shape, trainY.shape[1], units=unit)
    model.compile()

    tStart = time.now()
    model.fit(trainX, trainY)
    tEnd = time.now()
    c = tEnd - tStart

    print("Training time : " + str(c.microseconds) + " microseconds")

    print("####### Testing shuffle #######")
    result = []
    for i in range(len(percentageSubPathTest)):
        perm = np.random.permutation(len(testsX[i]))
        testX, testY = np.array(testsX[i])[perm], np.array(testsY[i])[perm]
        # Assess network accuracy score with test set
        scores = model.evaluate(testX, testY)
        if scores is not None:
            result.append(scores[1] * 100)
            print("Accuracy, from " + str(percentageSubPathTest[i][0] * 100) + "% to " + str(
                percentageSubPathTest[i][1] * 100) + "% : " + str(scores[1] * 100) + " %")
        else:
            print("Evaluation function not implemented")
    print(unit, result)
    print("###############################################")
    print("")
