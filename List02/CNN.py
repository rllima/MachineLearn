from random import *
import numpy as np
from sklearn.neighbors import NearestNeighbors

def CNN(trainSet):
    reduced = []
    print(randint(0,trainSet.shape[1]-1))