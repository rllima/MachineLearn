
import numpy as np
import reader as rd
import kfold as kfold
import prototypes


dataset = rd.readBase()
dataset = dataset.values
dataset = np.array(dataset)

Y = kfold.getDefaultResults(dataset)
numberOfClasses = len(set(Y))
classesValues = list(set(Y))
k = kfold.generateFolds(dataset,2)
for train_index, test_index in k.split(dataset,Y):
        X_train, X_test = dataset[train_index], dataset[test_index]
        print("TRAIN:", X_train.shape[0], "TEST:", X_test.shape[0])
        prototypes.generatePrototypes(X_train,numberOfClasses,classesValues)
       

