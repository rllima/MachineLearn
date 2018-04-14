
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
numberOfPrototypes = 20      #number of prototypes
epochs = 10                   #number of epochs training
lvqs = [lvq1, lvq21, lvq3]                   
dataset = rd.readBase()
dataset = dataset.values
dataset = np.array(dataset)
Y = kfold.getDefaultResults(dataset)
numberOfClasses = len(set(Y))
classesValues = list(set(Y))
count = 1;
for lvq in lvqs:
        print("LVQ " + str(count))
        k = kfold.generateFolds(dataset,kfoldNumber)
        totalAccuracy = []
        for train_index, test_index in k.split(dataset,Y):
                X_train, X_test = dataset[train_index], dataset[test_index]
                prot = prototypes.generatePrototypes(X_train,numberOfClasses,classesValues,numberOfPrototypes)
                before = time.time()	
                LVQ_Prototypes = lvq(X_train, 0.3, prot,epochs)
                final = time.time() - before
                print('>Training time=%.3f' % (final))
                trainProtypes , resultProtypes, test, resulTest = prepareData.slipData(LVQ_Prototypes, X_test)
                for k in [1,3,5]:
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(trainProtypes, resultProtypes)
                        pred = knn.predict(test)
                        shot = 0
                        for i in range(len(pred)):
                                if pred[i] == resulTest[i]:
                                        shot += 1
                        partialAccuracy = (shot/len(pred))
                        totalAccuracy.append([k,partialAccuracy])
        count+=1
        print('>Final Results - KNN = 1\n>Accuracy=%.3f' % (sum([x[1] for x in totalAccuracy if x[0] == 1])/kfoldNumber))
        print('>Final Results - KNN = 3\n>Accuracy=%.3f' % (sum([x[1] for x in totalAccuracy if x[0] == 3])/kfoldNumber))
        print('>Final Results - KNN = 5\n>Accuracy=%.3f' % (sum([x[1] for x in totalAccuracy if x[0] == 5])/kfoldNumber))
                                
                                


