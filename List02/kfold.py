from sklearn.model_selection import StratifiedKFold
import numpy as np

def generateFolds(dataset, numberSplites):  
    kf = StratifiedKFold(n_splits=numberSplites)
    return kf
       
def getDefaultResults(dataset):
    classes = []
    size = dataset.shape[1]
    for  var in dataset:
        classes.append(var[size-1])
    classes = np.array(classes)
    return classes    

