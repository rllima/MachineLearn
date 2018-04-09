from random import randint
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

def generatePrototypes(trainSet, numberOfClasses,classes, numberOfPrototypes):
    prototypes = [[] for i in range(numberOfClasses)]
    numberOfInstances = trainSet.shape[0]
    final = []
    for i,item in enumerate(classes):
        prototypes[i] = [x for x in trainSet if x[-1] == item]
    proportions = []
    for i, item in enumerate(classes):
        classP = len(prototypes[i])
        proportions.append([math.ceil((classP/numberOfInstances) * numberOfPrototypes),item])
    proportions.sort()
    total = sum(x[0] for x in proportions)
    while(total > numberOfPrototypes):
        proportions[-1][0] -= 1
        total = sum(x[0] for x in proportions)
    while(total < numberOfPrototypes):
        proportions[-1][0] += 1
        total = sum(proportions)
    for i,item in enumerate(classes):
        size = [x[0] for x in proportions if x[1] == item]
        for j in range(size[0]):
            rand = randint(0, len(prototypes[i])-1)
            final.append(prototypes[i][rand])
    return final

        
        
            