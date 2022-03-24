import h5py
import numpy as np
from datetime import datetime as time
import random
import LSTM
import json
from os import path
from MS import World, Node

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
testsX = []
testsY = []

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

print("####### Testing shuffle #######")
m = "Enigma"
print("Computing accuracy for map " + m + "...\r",end='')
world = World(m)

result = []
for i in range(len(percentageSubPathTest)):
    perm = np.random.permutation(len(testsX[i]))
    testX, testY = np.array(testsX[i])[perm], np.array(testsY[i])[perm]
    testX = testX[:500]
    testX_shape = testX.shape[0]
    res = 0
    
    for j in range(testX_shape):
        last_position = testX[j, -1]
        probas = world.predictMastersSardina(Node(last_position))
        res += int(probas[np.argmax(testY[j])] == max(probas))
    
    result.append(res / testX_shape * 100)

print(result)


results_file = "results.json"
if path.isfile(results_file):
    with open(results_file, 'r') as jean_michel_fichier:
        result_dict = json.load(jean_michel_fichier)
else:
    result_dict = dict()

result_dict["MS"] = {
    "0-25": result[0],
    "0-50": result[1],
    "0-75": result[2],
    "0-100": result[3]
}

with open(results_file, 'w') as jean_michel_fichier:
    json.dump(result_dict, jean_michel_fichier, indent=4)

print("###############################################")
print("")
