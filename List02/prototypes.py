from random import *
import numpy as np
from sklearn.neighbors import NearestNeighbors

def generatePrototypes(trainSet, numberOfClasses,classes):
    prototypes = [[] for i in range(numberOfClasses)]
    numberAttributes = trainSet.shape[1]
    numberOfInstances = trainSet.shape[0]
    for instance in trainSet:
        
            
               

         
        
        
