
import numpy as np
import reader as rd
import numbers
import kfold as kfold
import prototypes
from lvq import lvq1,lvq21,lvq3
import prepareData
import time
from sklearn.neighbors import KNeighborsClassifier

kfoldNumber = 2               #number of folds, for more or less, change this value.
numberOfPrototypes = 10       #number of prototypes
epochs = 5                    #number of epochs training
lvqs = [lvq1, lvq21,lvq3]                   
dataset = rd.readBase()
dataset = dataset.values
dataset = np.array(dataset)
Y = kfold.getDefaultResults(dataset)
numberOfClasses = len(set(Y))
classesValues = list(set(Y))


for l, lvq in enumerate(lvqs):
        print("LVQ " + str(l+1))
        k = kfold.generateFolds(dataset,kfoldNumber)
        totalAccuracy = 0.0
        for train_index, test_index in k.split(dataset,Y):
                X_train, X_test = dataset[train_index], dataset[test_index]
                prot = prototypes.generatePrototypes(X_train,numberOfClasses,classesValues,numberOfPrototypes)
                before = time.time()	
                LVQ_Prototypes = lvq(X_train, 0.1, prot,epochs)
                final = time.time() - before
                print('>Training time=%.3f' % (final))
                trainProtypes , resultProtypes, test, resulTest = prepareData.slipData(LVQ_Prototypes, X_train)
                for k in [1,3,5]:
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(trainProtypes, resultProtypes)
                        pred = knn.predict(test)
                        shot = 0
                        for i in range(len(pred)):
                                if pred[i] == resulTest[i]:
                                        shot += 1
                        partialAccuracy = (shot/len(pred))
                        totalAccuracy+=partialAccuracy
                        print('>Kneighbors=%d, >Partial-Accuracy=%.3f' % (k, partialAccuracy))
        print('>Final Resultas\n>Accuracy=%.3f' % (totalAccuracy/kfoldNumber))
                                     
                                


