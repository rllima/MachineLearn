import numbers

def slipData(train, test):
    trainSet = []
    resultTrainSet = []
    testSet = []
    resultTestSet = []

    for item in train:
        trainSet.append([x for x in item if isinstance(x, numbers.Number)])
        resultTrainSet.append(item[-1])

    for item in test:
        testSet.append([x for x in item if isinstance(x, numbers.Number)])
        resultTestSet.append(item[-1])
    return trainSet,resultTrainSet,testSet,resultTestSet