import h5py, glob
import numpy as np
from datetime import datetime as time
import CNNMultimaps
from save import save_results

# Load obsevations
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

testsX = [np.array(testX25), np.array(testX50), np.array(testX75), np.array(testX100)]
testsY = [np.array(testY25), np.array(testY50), np.array(testY75), np.array(testY100)]

# Random shuffle
perm = np.random.permutation(len(trainX))
trainX, trainY = np.array(trainX)[perm], np.array(trainY)[perm]
print("Input shape : " + str(trainX.shape))
print("Output shape : " + str(trainY.shape))

model = CNNMultimaps.PlanRecognitionModel(trainX[0].shape, trainY.shape[1],alpha=10)
model.compile()

tStart = time.now()
model.fit(trainX, trainY, 10, 10)
tEnd = time.now()
c = tEnd - tStart

print("Training time : " + str(c.microseconds) + " microseconds")

print("####### Testing shuffle #######")
result = []
for i in range(4):
    perm = np.random.permutation(len(testsX[i]))
    # Assess network accuracy score with test set
    scores = model.evaluate(testsX[i], testsY[i])
    result.append(scores[1] * 100)
    if scores is not None:
        print("Accuracy, from 0% to " + str(25 * (i + 1)) + "% : " + str(scores[1] * 100) + " %")
    else:
        print("Evaluation function not implemented")

print("###############################################")
print(result)
print("")

save_results(result, "CNNMultimaps")
