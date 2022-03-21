import numpy as np
from MS import World

"""This part is the main code to run everything. Do not touch it."""

#The maps to use during computation.
maps = ["BigGameHunters", "CrashSites", "Desolation", "Brushfire", "Caldera", "LakeShore", "Enigma", "WinterConquest"]
#You algorithm should ouput these accuracies
ref_accuracy = [100, 90, 100, 100, 99, 99, 100, 100]

#Assess M&S accuracy score on different maps
print("")
print("Map             | Accuracy")
print("--------------------------")
for n in range(len(maps)):
    m = maps[n]
    print("Computing accuracy for map " + m + "...\r",end='')
    world = World(m)
    probabilities = []
    real_goals = []
    for i in range(len(world.observations)):
        probabilities.append(world.predictMastersSardina(world.observations[i]))
        real_goals.append(world.goals[np.argmax(world.labels[i])])
    accuracy = world.accuracy(probabilities, real_goals)
    if (accuracy == ref_accuracy[n]):
        print(m + ' '*(15-len(m)) + " | " + str(accuracy) + " % ---> Correct !                                             ")
    else:
        print(m + ' '*(15-len(m)) + " | " + str(accuracy) + " % ---> Wrong ! This accuracy should be " + str(ref_accuracy[n]) + " %")
print("")