from sklearn.model_selection import StratifiedKFold
import numpy as np

def generateFolds(dataset, numberSplites):  
    kf = StratifiedKFold(n_splits=numberSplites,random_state=1)
    return kf
       
def getDefaultResults(dataset):
    classes = []
    size = dataset.shape[1]
    for  var in dataset:
        classes.append(var[-1])
    classes = np.array(classes)
    return classes    

