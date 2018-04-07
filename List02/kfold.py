from sklearn.model_selection import StratifiedKFold
import numpy as np

def generateFolds(dataset, numberSplites):  
    kf = StratifiedKFold(n_splits=numberSplites)
    return kf
       
def getDefaultResults(dataset):
    classes = []
    for  var in dataset:
        classes.append(var[21])
    classes = np.array(classes)
    return classes    

